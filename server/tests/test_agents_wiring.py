"""Interaction-agent wiring over the registry: tools, prompt rendering, last_routed and execution status."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

import server.agents.execution_agent.batch_manager as batch_module
import server.agents.interaction_agent.agent as agent_module
import server.agents.interaction_agent.runtime as runtime_module
import server.agents.interaction_agent.tools as tools
from server.agents.execution_agent.runtime import ExecutionResult
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.agents.interaction_agent.tools import ToolResult
from server.services.agents import AgentCard, TOOL_SCHEMAS
from server.services.agents.registry import SqliteAgentDirectory
from server.services.conversation.log import ConversationLog
from server.services.execution.log_store import ExecutionAgentLogStore

NOW = datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=5)


class StubBatchManager:
    def __init__(self):
        self.calls = []

    async def execute_agent(self, agent_id, instructions, agent_name=None, request_id=None):
        self.calls.append((agent_id, instructions, agent_name))
        return ExecutionResult(agent_name=agent_name or agent_id, agent_id=agent_id, success=True, response="ok")


@pytest.fixture
def env(tmp_path, monkeypatch):
    directory = SqliteAgentDirectory(tmp_path / "agents.db")
    directory.add_cards(
        [
            AgentCard("kwame-book-club", "Kwame Book Club", "Ask Kwame which dates work for book club", ["kwame"],
                      created_at=OLD, last_active_at=OLD, run_count=1),
            AgentCard("beatriz-trivia", "Beatriz Trivia Night", "Email Beatriz about trivia night", ["beatriz"],
                      created_at=OLD, last_active_at=OLD, run_count=1),
        ]
    )
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = StubBatchManager()
    monkeypatch.setattr(tools, "get_agent_directory", lambda: directory)
    monkeypatch.setattr(tools, "get_execution_agent_logs", lambda: logs)
    monkeypatch.setattr(tools, "_EXECUTION_BATCH_MANAGER", batch)
    return directory, logs, batch


async def settle():
    for _ in range(3):
        await asyncio.sleep(0)


async def test_send_reuses_by_id(env):
    directory, logs, batch = env
    result = tools.handle_tool_call(
        "send_message_to_agent", json.dumps({"instructions": "Any reply from Kwame?", "agent_id": "kwame-book-club"})
    )
    await settle()
    assert result.success
    assert result.payload == {
        "status": "submitted", "agent_id": "kwame-book-club", "agent_name": "Kwame Book Club", "new_agent_created": False,
    }
    assert batch.calls == [("kwame-book-club", "Any reply from Kwame?", "Kwame Book Club")]
    assert "Any reply from Kwame?" in logs.load_transcript("kwame-book-club")
    assert directory.get("kwame-book-club").run_count == 2


async def test_send_creates_new_agent(env):
    directory, logs, batch = env
    args = {"instructions": "Call Gary Whitman about the lease", "name": "Gary Whitman Lease", "purpose": "Lease renewal with Gary Whitman"}
    result = tools.handle_tool_call("send_message_to_agent", args)
    await settle()
    assert result.success and result.payload["new_agent_created"] is True
    assert result.payload["agent_id"] == "gary-whitman-lease"
    assert directory.get("gary-whitman-lease").purpose == "Lease renewal with Gary Whitman"
    assert batch.calls == [("gary-whitman-lease", args["instructions"], "Gary Whitman Lease")]
    assert "Gary Whitman" in logs.load_transcript("gary-whitman-lease")


async def test_possible_duplicate_sends_nothing(env):
    directory, logs, batch = env
    args = {"instructions": "Ask Kwame again", "name": "Kwame Book Club!", "purpose": "Book club dates with Kwame"}
    result = tools.handle_tool_call("send_message_to_agent", args)
    await settle()
    assert not result.success
    assert result.payload["status"] == "possible_duplicate"
    assert 'agent_id="kwame-book-club"' in result.payload["instructions"]
    assert batch.calls == [] and len(directory.list_cards()) == 2
    assert logs.list_agents() == []

    forced = tools.handle_tool_call("send_message_to_agent", {**args, "force_new": True})
    await settle()
    assert forced.success and forced.payload["agent_id"] == "kwame-book-club-2"
    assert [call[0] for call in batch.calls] == ["kwame-book-club-2"]


async def test_unknown_id_and_missing_fields_are_errors(env):
    _, _, batch = env
    unknown = tools.handle_tool_call("send_message_to_agent", {"instructions": "x", "agent_id": "kwame"})
    missing = tools.handle_tool_call("send_message_to_agent", {"instructions": "x", "name": "Only a name"})
    await settle()
    assert not unknown.success and "Unknown agent_id" in unknown.payload["error"]
    assert "kwame-book-club" in unknown.payload["nearest_agents"]
    assert not missing.success and "purpose" in missing.payload["error"]
    assert batch.calls == []


async def test_argument_errors_keep_handle_tool_call_behaviour(env):
    result = tools.handle_tool_call("send_message_to_agent", {"agent_name": "x", "instructions": "y"})
    assert not result.success and result.payload["error"].startswith("Missing required arguments:")
    assert "agent_name" in result.payload["error"]


def test_send_without_event_loop_is_an_error(env):
    _, _, batch = env
    result = tools.send_message_to_agent("x", agent_id="kwame-book-club")
    assert not result.success and result.payload == {"error": "No event loop available"}


def test_search_agents(env):
    result = tools.handle_tool_call("search_agents", {"query": "beatriz trivia"})
    assert result.success
    assert result.payload.startswith('<active_agents shown="')
    assert 'id="beatriz-trivia"' in result.payload


def test_tool_schemas_are_shared():
    assert tools.get_tool_schemas() is TOOL_SCHEMAS


# -- prepare_message_with_history ---------------------------------------------------------------------


class FakeLog:
    def __init__(self, texts):
        self.texts = texts

    def recent_texts(self, limit):
        return self.texts[-limit:]


def section(content, tag):
    start = content.index(f"<{tag}")
    end = content.index(f"</{tag}>") + len(f"</{tag}>")
    return content[start:end]


@pytest.fixture
def prompt_env(env, monkeypatch):
    directory = env[0]
    monkeypatch.setattr(agent_module, "get_agent_directory", lambda: directory)
    return directory


def test_prepare_renders_relevant_cards(prompt_env, monkeypatch):
    monkeypatch.setattr(agent_module, "get_conversation_log", lambda: FakeLog(["hi", "Draft for Beatriz?"]))
    content = agent_module.prepare_message_with_history("Draft for Beatriz?", "<user_message>hi</user_message>", now=NOW)[0]["content"]
    roster = section(content, "active_agents")
    assert roster.startswith('<active_agents shown="1" total="2">')
    assert 'id="beatriz-trivia"' in roster and "kwame" not in roster
    # History and the current turn render exactly as before.
    assert content.startswith("<conversation_history>\n<user_message>hi</user_message>\n</conversation_history>\n\n")
    assert content.endswith("\n\n<new_user_message>\nDraft for Beatriz?\n</new_user_message>")


def test_prepare_uses_recent_turns_and_last_routed(prompt_env, monkeypatch):
    # The vague follow-up matches nothing itself; the earlier turn about Kwame does.
    monkeypatch.setattr(agent_module, "get_conversation_log", lambda: FakeLog(["Ask Kwame about book club", "did he reply?"]))
    roster = section(agent_module.prepare_message_with_history("did he reply?", "", now=NOW)[0]["content"], "active_agents")
    assert 'id="kwame-book-club"' in roster

    monkeypatch.setattr(agent_module, "get_conversation_log", lambda: FakeLog([]))
    prompt_env.set_last_routed(["beatriz-trivia"])
    roster = section(agent_module.prepare_message_with_history("did she reply?", "", now=NOW)[0]["content"], "active_agents")
    assert 'id="beatriz-trivia"' in roster


def test_prepare_pins_agents_on_agent_messages(prompt_env, monkeypatch):
    monkeypatch.setattr(agent_module, "get_conversation_log", lambda: FakeLog([]))
    message = "[SUCCESS] Kwame Book Club: Kwame replied with Friday."
    agent_turn = agent_module.prepare_message_with_history(message, "", message_type="agent", now=NOW)[0]["content"]
    roster = section(agent_turn, "active_agents")
    assert roster.index('id="kwame-book-club"') < roster.index("</agent>")  # pinned first
    assert agent_turn.endswith(f"<new_agent_message>\n{message}\n</new_agent_message>")


def test_prepare_with_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "get_agent_directory", lambda: SqliteAgentDirectory(tmp_path / "a.db"))
    monkeypatch.setattr(agent_module, "get_conversation_log", lambda: FakeLog([]))
    content = agent_module.prepare_message_with_history("hello", "", now=NOW)[0]["content"]
    assert '<active_agents shown="0" total="0">None</active_agents>' in content


def test_recent_turn_texts_drop_the_latest_message(monkeypatch):
    monkeypatch.setattr(agent_module, "get_conversation_log", lambda: FakeLog(["a", "b", "c", "d", "latest"]))
    assert agent_module._recent_turn_texts("latest") == ["b", "c", "d"]
    assert agent_module._recent_turn_texts("not logged yet") == ["c", "d", "latest"]


def test_conversation_log_recent_texts(tmp_path):
    path = tmp_path / "conversation.log"
    path.write_text(
        '<user_message timestamp="t">one</user_message>\n'
        '<poke_reply timestamp="t">two</poke_reply>\n'
        '<wait timestamp="t">skip</wait>\n'
        '<agent_message timestamp="t">three\\nlines</agent_message>\n',
        encoding="utf-8",
    )
    log = ConversationLog(path)
    assert log.recent_texts(2) == ["two", "three\nlines"]
    assert log.recent_texts(0) == []


# -- runtime: last_routed after a user turn -----------------------------------------------------------


def bare_runtime():
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    replies = iter(
        [
            {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "send_message_to_agent", "arguments": "{}"}},
                {"id": "2", "function": {"name": "send_message_to_agent", "arguments": "{}"}},
                {"id": "3", "function": {"name": "send_message_to_agent", "arguments": "{}"}},
            ]}}]},
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    async def fake_llm(system_prompt, messages):
        return next(replies)

    runtime._make_llm_call = fake_llm
    return runtime


async def test_runtime_collects_dispatched_ids_and_sets_last_routed(tmp_path, monkeypatch):
    results = iter(
        [
            ToolResult(success=True, payload={"status": "submitted", "agent_id": "a"}),
            ToolResult(success=False, payload={"status": "possible_duplicate"}),
            ToolResult(success=True, payload={"status": "submitted", "agent_id": "b"}),
        ]
    )
    monkeypatch.setattr(runtime_module, "handle_tool_call", lambda name, args: next(results))
    directory = SqliteAgentDirectory(tmp_path / "agents.db")
    monkeypatch.setattr(runtime_module, "get_agent_directory", lambda: directory)

    runtime = bare_runtime()
    summary = await runtime._run_interaction_loop("system", [])
    assert summary.dispatched_ids == ["a", "b"]
    runtime._update_last_routed(summary)
    assert directory.last_routed() == ["a", "b"]

    runtime._update_last_routed(runtime_module._LoopSummary())  # nothing dispatched keeps the boost
    assert directory.last_routed() == ["a", "b"]


# -- execution side: status and the [SUCCESS] payload --------------------------------------------------


async def test_batch_manager_records_status_and_keeps_payload_format(tmp_path, monkeypatch):
    directory = SqliteAgentDirectory(tmp_path / "agents.db")
    card = directory.create("Email to Alice", "Reply to Alice", "", NOW)
    created = []

    class FakeRuntime:
        def __init__(self, agent_id, agent_name=None):
            created.append((agent_id, agent_name))

        async def execute(self, instructions):
            return ExecutionResult(agent_name="Email to Alice", agent_id=card.id, success=True, response="Sent the reply. " * 30)

    dispatched = []

    async def capture(self, payload):
        dispatched.append(payload)

    monkeypatch.setattr(batch_module, "get_agent_directory", lambda: directory)
    monkeypatch.setattr(batch_module, "ExecutionAgentRuntime", FakeRuntime)
    monkeypatch.setattr(batch_module.ExecutionBatchManager, "_dispatch_to_interaction_agent", capture)

    await batch_module.ExecutionBatchManager().execute_agent(card.id, "Reply to Alice")
    assert created == [(card.id, "Email to Alice")]  # display name resolved from the card
    assert directory.get(card.id).last_status == ("Sent the reply. " * 30).strip()[:200]
    assert dispatched == [f"[SUCCESS] Email to Alice: {('Sent the reply. ' * 30).strip()}"]
