"""Agent runner: per-agent serial runs, the global slot cap, partial results and result delivery."""

import asyncio
import json
from datetime import datetime, timezone

import pytest

import server.agents.execution_agent.agent as agent_module
import server.agents.execution_agent.runtime as runtime_module
import server.services.execution.runner as runner_module
from server.agents.execution_agent.runtime import ExecutionAgentRuntime, ExecutionResult
from server.services.agents import AgentCard, InMemoryDirectory
from server.services.agents.registry import SqliteAgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.runner import AgentRunner, current_run, format_result, side_effects_since

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def ok(agent_id: str, response: str = "ok") -> ExecutionResult:
    return ExecutionResult(agent_name=agent_id, agent_id=agent_id, success=True, response=response)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A runner over a temp log store and directory, with compaction off and deliveries captured."""
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    directory = SqliteAgentDirectory(tmp_path / "agents.db")
    directory.add_cards([AgentCard("alice", "Alice Reply", "Reply to Alice", ["alice"], created_at=NOW, last_active_at=NOW)])
    delivered = []

    async def no_compact(agent_id):
        return False

    def capture(result, *, origin, trigger_id=None):
        delivered.append((format_result(result), origin, trigger_id))

    monkeypatch.setattr(runner_module, "get_agent_directory", lambda: directory)
    monkeypatch.setattr(runner_module, "maybe_compact", no_compact)
    monkeypatch.setattr(runner_module, "deliver_result", capture)
    return logs, directory, delivered


def scripted_runtime(monkeypatch, execute):
    """Swap the EA runtime for one whose `execute(agent_id, instructions)` is the given coroutine."""

    class FakeRuntime:
        def __init__(self, agent_id, agent_name=None):
            self.agent_id = agent_id

        async def execute(self, instructions):
            return await execute(self.agent_id, instructions)

    monkeypatch.setattr(runner_module, "ExecutionAgentRuntime", FakeRuntime)


async def settle(times: int = 5):
    for _ in range(times):
        await asyncio.sleep(0)


# -- concurrency ----------------------------------------------------------------------------------------------


async def test_runs_on_one_agent_never_overlap_and_keep_fifo_order(env, monkeypatch):
    logs, _, _ = env
    events, gates = [], {name: asyncio.Event() for name in ("first", "second", "third")}

    async def execute(agent_id, instructions):
        events.append(f"start {instructions}")
        await gates[instructions].wait()
        events.append(f"end {instructions}")
        return ok(agent_id)

    scripted_runtime(monkeypatch, execute)
    runner = AgentRunner(max_concurrent=4, timeout_seconds=5, log_store=logs)
    tasks = [asyncio.create_task(runner.run("alice", name, origin="interaction")) for name in gates]
    await settle()
    assert events == ["start first"]  # the correction queues behind the run in progress
    gates["third"].set()
    gates["second"].set()
    await settle()
    assert events == ["start first"]
    gates["first"].set()
    await asyncio.gather(*tasks)
    assert events == ["start first", "end first", "start second", "end second", "start third", "end third"]


async def test_fan_out_never_exceeds_the_slot_cap(env, monkeypatch):
    logs, _, _ = env
    active, peak = [0], [0]

    async def execute(agent_id, instructions):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.01)
        active[0] -= 1
        return ok(agent_id)

    scripted_runtime(monkeypatch, execute)
    runner = AgentRunner(max_concurrent=2, timeout_seconds=5, log_store=logs)
    await asyncio.gather(*(runner.run(f"agent-{n}", "go", origin="interaction") for n in range(5)))
    assert peak[0] == 2


async def test_a_queued_run_holds_no_slot(env, monkeypatch):
    logs, _, _ = env
    started, release = [], asyncio.Event()

    async def execute(agent_id, instructions):
        started.append((agent_id, instructions))
        if agent_id == "alice":
            await release.wait()
        return ok(agent_id)

    scripted_runtime(monkeypatch, execute)
    runner = AgentRunner(max_concurrent=2, timeout_seconds=5, log_store=logs)
    first = asyncio.create_task(runner.run("alice", "one", origin="interaction"))
    queued = asyncio.create_task(runner.run("alice", "two", origin="interaction"))
    await settle()
    await asyncio.wait_for(runner.run("bob", "hi", origin="interaction"), 1)  # alice's queued run left a slot free
    assert started == [("alice", "one"), ("bob", "hi")]
    release.set()
    await asyncio.gather(first, queued)
    assert started[-1] == ("alice", "two")


async def test_the_run_context_names_the_agent_and_trigger(env, monkeypatch):
    logs, _, _ = env
    seen = []

    async def execute(agent_id, instructions):
        await asyncio.sleep(0)
        seen.append(current_run.get())
        return ok(agent_id)

    scripted_runtime(monkeypatch, execute)
    runner = AgentRunner(max_concurrent=2, timeout_seconds=5, log_store=logs)
    await runner.run("alice", "weekly report", origin="trigger", trigger_id=9)
    assert seen[0].agent_id == "alice" and seen[0].trigger_id == 9 and seen[0].origin == "trigger"
    assert current_run.get() is None


# -- status, compaction and delivery --------------------------------------------------------------------------


async def test_a_run_records_status_and_delivers_one_result(env, monkeypatch):
    logs, directory, delivered = env

    async def execute(agent_id, instructions):
        return ExecutionResult(agent_name="Alice Reply", agent_id=agent_id, success=True, response=" Sent the reply. ")

    scripted_runtime(monkeypatch, execute)
    result = await AgentRunner(log_store=logs).run("alice", "Reply", origin="trigger", trigger_id=3)
    assert result.success
    assert directory.get("alice").last_status == "Sent the reply."
    assert delivered == [("[SUCCESS] Alice Reply: Sent the reply.", "trigger", 3)]


async def test_each_result_is_delivered_on_its_own(env, monkeypatch):
    logs, _, delivered = env

    async def execute(agent_id, instructions):
        return ok(agent_id, f"{agent_id} done")

    scripted_runtime(monkeypatch, execute)
    runner = AgentRunner(log_store=logs)
    await asyncio.gather(runner.run("a", "go", origin="interaction"), runner.run("b", "go", origin="interaction"))
    assert sorted(item[0] for item in delivered) == ["[SUCCESS] a: a done", "[SUCCESS] b: b done"]


async def test_deliver_result_posts_the_payload_to_the_inbox(monkeypatch):
    received = []

    class FakeInbox:
        def post(self, kind, payload, agent_id=None):
            received.append((kind, payload, agent_id))

    monkeypatch.setattr(runner_module, "get_inbox", lambda: FakeInbox())
    failed = ExecutionResult(agent_name="Bob", agent_id="bob", success=False, response="No such thread")
    runner_module.deliver_result(failed, origin="interaction")
    assert received == [("agent_result", "[FAILED] Bob: No such thread", "bob")]


# -- partial results ------------------------------------------------------------------------------------------


def record_call(logs, agent_id, tool, args, result):
    logs.record_action(agent_id, f"Calling {tool} with: {json.dumps(args)}")
    logs.record_tool_response(agent_id, tool, json.dumps(result))


@pytest.fixture
def real_runtime(env, monkeypatch):
    """The real EA runtime over the temp log store with a scripted LLM and stub tools."""
    logs, _, _ = env

    class StubSettings:
        openrouter_api_key = "test"
        execution_agent_model = "test-model"

    tools = {
        "gmail_create_draft": lambda **kwargs: {"successful": True, "data": {"id": "r-123", "message": {"id": "m-1"}}},
        "gmail_list_drafts": lambda **kwargs: {"successful": True, "data": {"drafts": []}},
    }
    calls = []

    async def llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            name, args = "gmail_create_draft", {"recipient_email": "alice@x.com", "subject": "Hi", "body": "Hello"}
        else:
            name, args = "gmail_list_drafts", {}
        tool_call = {"id": f"c{len(calls)}", "function": {"name": name, "arguments": json.dumps(args)}}
        return {"choices": [{"message": {"content": "", "tool_calls": [tool_call]}}]}

    monkeypatch.setattr(runtime_module, "get_settings", lambda: StubSettings())
    monkeypatch.setattr(runtime_module, "get_tool_registry", lambda agent_id, agent_name=None: tools)
    monkeypatch.setattr(runtime_module, "get_tool_schemas", lambda: [])
    monkeypatch.setattr(runtime_module, "request_chat_completion", llm)
    monkeypatch.setattr(agent_module, "get_execution_agent_logs", lambda: logs)
    monkeypatch.setattr(agent_module, "get_agent_directory", lambda: InMemoryDirectory())
    return logs, calls


async def test_runtime_returns_a_partial_result_at_the_iteration_limit(real_runtime):
    logs, calls = real_runtime
    result = await ExecutionAgentRuntime("alice", "Alice Reply").execute("Draft a reply to Alice")
    assert len(calls) == ExecutionAgentRuntime.MAX_TOOL_ITERATIONS
    assert not result.success and result.partial
    assert "limit" in result.response and result.tools_executed[0] == "gmail_create_draft"


async def test_iteration_limit_lists_drafts_from_this_run_only(env, real_runtime):
    logs, _ = real_runtime
    _, directory, delivered = env
    record_call(logs, "alice", "gmail_create_draft", {"recipient_email": "old@x.com"},
                {"successful": True, "data": {"id": "r-old"}})

    result = await AgentRunner(log_store=logs).run("alice", "Draft a reply to Alice", origin="interaction")
    assert result.partial and not result.success
    assert result.side_effects == ["draft r-123 to alice@x.com"]
    assert "r-old" not in result.response
    payload = delivered[0][0]
    assert payload.startswith("[PARTIAL] Alice Reply: ")
    assert payload.endswith("Already done: draft r-123 to alice@x.com.")
    assert "Already done: draft r-123" in directory.get("alice").last_status


async def test_timeout_reports_side_effects_already_performed(env, monkeypatch):
    logs, _, delivered = env

    async def execute(agent_id, instructions):
        record_call(logs, agent_id, "createTrigger", {"payload": "Check in"}, {"trigger_id": 9, "status": "active"})
        record_call(logs, agent_id, "gmail_execute_draft", {"draft_id": "r-5"}, {"successful": True, "data": {}})
        await asyncio.sleep(10)

    scripted_runtime(monkeypatch, execute)
    result = await AgentRunner(timeout_seconds=0.05, log_store=logs).run("alice", "go", origin="interaction")
    assert result.partial and result.error == "Timeout"
    assert delivered[0][0] == (
        "[PARTIAL] Alice Reply: Timed out after 0.05 seconds before finishing. "
        "Already done: trigger 9 set; sent draft r-5."
    )


async def test_partial_with_no_side_effects_says_so(env, monkeypatch):
    logs, _, delivered = env

    async def execute(agent_id, instructions):
        await asyncio.sleep(10)

    scripted_runtime(monkeypatch, execute)
    await AgentRunner(timeout_seconds=0.01, log_store=logs).run("alice", "go", origin="interaction")
    assert delivered[0][0].endswith("Already done: nothing.")


def test_side_effects_skip_failed_and_read_only_calls(tmp_path):
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    record_call(logs, "a", "gmail_create_draft", {"recipient_email": "x@y.com"}, {"successful": True, "data": {"id": "r-1"}})
    start = len(list(logs.iter_entries("a")))
    record_call(logs, "a", "gmail_create_draft", {"recipient_email": "bad@y.com"}, {"error": "Gmail not connected"})
    record_call(logs, "a", "gmail_reply_to_thread", {"thread_id": "t-1", "recipient_email": "bo@y.com"},
                {"successful": False, "error": "boom"})
    logs.record_action("a", 'Calling gmail_forward_email with: {"message_id": "m-1", "recipient_email": "c@y.com"}')
    logs.record_tool_response("a", "gmail_forward_email", "Composio exploded")  # a raised tool logs its error text
    record_call(logs, "a", "gmail_list_drafts", {}, {"successful": True, "data": {}})
    record_call(logs, "a", "gmail_reply_to_thread", {"thread_id": "t-2", "recipient_email": "di@y.com"},
                {"successful": True, "data": {}})
    record_call(logs, "a", "updateTrigger", {"trigger_id": 4, "status": "paused"}, {"trigger_id": 4, "status": "paused"})
    assert side_effects_since(logs, "a", start) == ["replied to di@y.com on thread t-2", "trigger 4 paused"]
