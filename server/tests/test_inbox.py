"""Interaction inbox: durable store, debounce and coalescing, ordering, restart recovery, turn context."""

import asyncio
from datetime import datetime, timezone

import pytest

import server.agents.interaction_agent.runtime as ia_runtime_module
import server.services.conversation.chat_handler as chat_handler_module
import server.services.execution.runner as runner_module
from server.agents.execution_agent.runtime import ExecutionResult
from server.agents.interaction_agent.runtime import InteractionAgentRuntime, InteractionResult
from server.agents.interaction_agent.tools import ToolResult
from server.agents.interaction_agent.turn import TurnContext, current_turn
from server.models import ChatMessage, ChatRequest
from server.services.conversation.log import ConversationLog
from server.services.conversation.summarization.working_memory_log import WorkingMemoryLog
from server.services.inbox import InboxStore, InteractionInbox

DEBOUNCE = 0.05


class FakeLog:
    """Conversation log stand-in recording what was written, with the time of each user message."""

    def __init__(self):
        self.entries = []

    def record_user_message(self, content):
        self.entries.append(("user_message", content, datetime.now(timezone.utc)))


class FakeRuntime:
    """Records each turn; `gate` (when set) holds a turn open until released; `fail` makes turns raise."""

    def __init__(self):
        self.turns = []
        self.gate = None
        self.started = asyncio.Event()
        self.fail = False

    def factory(self):
        return self

    async def _turn(self, entry):
        self.turns.append(entry)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("model unavailable")
        return InteractionResult(success=True, response="ok")

    async def execute(self, user_message, *, user_message_at=None, recorded=()):
        return await self._turn(("user", user_message, user_message_at, tuple(recorded)))

    async def handle_agent_message(self, agent_message, *, origin="agent"):
        return await self._turn((origin, agent_message))


@pytest.fixture
async def inbox(tmp_path):
    """A started inbox over a temp store with a fake runtime and log; stopped after the test."""
    runtime = FakeRuntime()
    log = FakeLog()
    box = InteractionInbox(
        InboxStore(tmp_path / "inbox.db"),
        debounce_seconds=DEBOUNCE,
        runtime_factory=runtime.factory,
        conversation_log=log,
    )
    await box.start()
    yield box, runtime, log
    await box.stop()


async def drain(box, timeout=2.0):
    """Wait until nothing is pending or processing."""
    async def _idle():
        while box.store.unfinished():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_idle(), timeout)
    await asyncio.sleep(0.01)


# -- store -----------------------------------------------------------------------------------------------------


def test_store_round_trip_and_status(tmp_path):
    store = InboxStore(tmp_path / "inbox.db")
    first = store.insert("agent_result", "[SUCCESS] a: done", agent_id="a")
    second = store.insert("user", "hi")
    assert [item.id for item in store.pending()] == [first.id, second.id]
    assert first.agent_id == "a" and first.status == "pending"
    assert first.created_at.tzinfo is not None and first.created_at.utcoffset().total_seconds() == 0

    store.mark([first.id], "processing")
    assert [item.id for item in store.pending()] == [second.id]
    assert store.reset_processing() == 1
    store.mark([first.id, second.id], "done")
    assert store.pending() == [] and store.unfinished() == []


def test_store_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        InboxStore(tmp_path / "inbox.db").insert("email", "x")


def test_created_at_keeps_microseconds(tmp_path):
    store = InboxStore(tmp_path / "inbox.db")
    at = datetime(2026, 10, 28, 12, 0, 0, 123456, tzinfo=timezone.utc)
    item = store.insert("user", "hi", created_at=at)
    assert store.pending()[0].created_at == at == item.created_at


# -- consumer --------------------------------------------------------------------------------------------------


async def test_results_posted_within_the_debounce_make_one_turn(inbox):
    box, runtime, _ = inbox
    box.post("agent_result", "[SUCCESS] a: one", agent_id="a")
    await asyncio.sleep(DEBOUNCE / 5)
    box.post("agent_result", "[SUCCESS] b: two", agent_id="b")
    box.post("agent_result", "[FAILED] c: three", agent_id="c")
    await drain(box)
    assert runtime.turns == [("agent_result", "[SUCCESS] a: one\n[SUCCESS] b: two\n[FAILED] c: three")]


async def test_a_user_message_skips_the_debounce(tmp_path):
    runtime = FakeRuntime()
    box = InteractionInbox(
        InboxStore(tmp_path / "inbox.db"), debounce_seconds=30, runtime_factory=runtime.factory, conversation_log=FakeLog()
    )
    await box.start()
    try:
        box.post("user", "hello")
        await asyncio.wait_for(runtime.started.wait(), 1.0)
        assert runtime.turns[0][:2] == ("user", "hello")
    finally:
        await box.stop()


async def test_user_items_run_before_agent_items(inbox):
    box, runtime, _ = inbox
    box.post("watcher", "Important email watcher notification: x")
    box.post("user", "first")
    box.post("trigger_result", "[SUCCESS] t: fired", agent_id="t")
    box.post("user", "second")
    await drain(box)
    assert [turn[0] for turn in runtime.turns] == ["user", "agent"]
    assert runtime.turns[0][1] == "first\n\nsecond"
    assert runtime.turns[0][3] == ("first", "second")
    assert runtime.turns[1][1] == "Important email watcher notification: x\n[SUCCESS] t: fired"


async def test_user_turn_carries_the_newest_user_items_time(inbox):
    box, runtime, log = inbox
    box.post("user", "first")
    second = box.post("user", "second")
    await drain(box)
    user_message_at = runtime.turns[0][2]
    assert user_message_at == second.created_at
    assert user_message_at.tzinfo is not None and user_message_at.utcoffset().total_seconds() == 0
    # The log already holds each user message, written no later than the item's created_at.
    assert [entry[1] for entry in log.entries] == ["first", "second"]
    assert log.entries[-1][2] <= user_message_at


async def test_an_item_posted_during_a_turn_waits_for_the_next(inbox):
    box, runtime, _ = inbox
    runtime.gate = asyncio.Event()
    box.post("user", "one")
    await asyncio.wait_for(runtime.started.wait(), 1.0)
    box.post("user", "two")
    box.post("agent_result", "[SUCCESS] a: done", agent_id="a")
    await asyncio.sleep(DEBOUNCE * 2)
    assert len(runtime.turns) == 1
    runtime.gate.set()
    await drain(box)
    assert [turn[:2] for turn in runtime.turns] == [
        ("user", "one"),
        ("user", "two"),
        ("agent_result", "[SUCCESS] a: done"),
    ]


async def test_turn_failures_mark_items_done(inbox, caplog):
    box, runtime, _ = inbox
    runtime.fail = True
    box.post("user", "hi")
    box.post("agent_result", "[SUCCESS] a: done", agent_id="a")
    await drain(box)
    assert len(runtime.turns) == 2  # each batch tried once, never retried
    assert box.store.unfinished() == []


async def test_a_runtime_that_cannot_start_marks_items_done(tmp_path):
    def broken():
        raise ValueError("OpenRouter API key not configured.")

    box = InteractionInbox(
        InboxStore(tmp_path / "inbox.db"), debounce_seconds=DEBOUNCE, runtime_factory=broken, conversation_log=FakeLog()
    )
    await box.start()
    try:
        box.post("user", "hi")
        await drain(box)
        assert box.store.unfinished() == []
    finally:
        await box.stop()


async def test_restart_recovers_pending_and_processing_items(tmp_path):
    store = InboxStore(tmp_path / "inbox.db")
    stuck = store.insert("user", "was mid-turn")
    store.mark([stuck.id], "processing")
    store.insert("agent_result", "[SUCCESS] a: finished while down", agent_id="a")
    done = store.insert("user", "already handled")
    store.mark([done.id], "done")

    runtime = FakeRuntime()
    box = InteractionInbox(store, debounce_seconds=DEBOUNCE, runtime_factory=runtime.factory, conversation_log=FakeLog())
    await box.start()
    try:
        await drain(box)
    finally:
        await box.stop()
    assert [turn[:2] for turn in runtime.turns] == [
        ("user", "was mid-turn"),
        ("agent_result", "[SUCCESS] a: finished while down"),
    ]


async def test_post_before_start_is_kept_and_processed(tmp_path):
    runtime = FakeRuntime()
    box = InteractionInbox(
        InboxStore(tmp_path / "inbox.db"), debounce_seconds=DEBOUNCE, runtime_factory=runtime.factory, conversation_log=FakeLog()
    )
    box.post("watcher", "alert")
    await box.start()
    try:
        await drain(box)
    finally:
        await box.stop()
    assert runtime.turns == [("watcher", "alert")]


async def test_stop_leaves_no_task_running(tmp_path):
    box = InteractionInbox(
        InboxStore(tmp_path / "inbox.db"), debounce_seconds=DEBOUNCE, runtime_factory=FakeRuntime().factory, conversation_log=FakeLog()
    )
    await box.start()
    task = box._task
    await box.stop()
    assert task.done() and box._task is None


# -- producers -------------------------------------------------------------------------------------------------


class CapturingInbox:
    def __init__(self):
        self.posts = []

    def post(self, kind, payload, agent_id=None):
        self.posts.append((kind, payload, agent_id))


def test_runner_results_post_agent_or_trigger_results(monkeypatch):
    box = CapturingInbox()
    monkeypatch.setattr(runner_module, "get_inbox", lambda: box)
    failed = ExecutionResult(agent_name="Bob", agent_id="bob", success=False, response="No such thread")
    runner_module.deliver_result(failed, origin="interaction")
    fired = ExecutionResult(agent_name="Digest", agent_id="digest", success=True, response="Sent")
    runner_module.deliver_result(fired, origin="trigger", trigger_id=7)
    assert box.posts == [
        ("agent_result", "[FAILED] Bob: No such thread", "bob"),
        ("trigger_result", "[SUCCESS] Digest: Sent", "digest"),
    ]


async def test_chat_handler_posts_the_user_message(monkeypatch):
    box = CapturingInbox()
    monkeypatch.setattr(chat_handler_module, "get_inbox", lambda: box)

    class Settings:
        openrouter_api_key = "test"

    monkeypatch.setattr(chat_handler_module, "get_settings", lambda: Settings())
    request = ChatRequest(messages=[ChatMessage(role="user", content="  draft a reply to Ana  ")])
    response = await chat_handler_module.handle_chat_request(request)
    assert response.status_code == 202
    assert box.posts == [("user", "draft a reply to Ana", None)]


async def test_chat_handler_without_api_key_is_rejected(monkeypatch):
    box = CapturingInbox()
    monkeypatch.setattr(chat_handler_module, "get_inbox", lambda: box)

    class Settings:
        openrouter_api_key = None

    monkeypatch.setattr(chat_handler_module, "get_settings", lambda: Settings())
    response = await chat_handler_module.handle_chat_request(
        ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    )
    assert response.status_code == 400 and box.posts == []


# -- the runtime sets current_turn -----------------------------------------------------------------------------


class RuntimeLog:
    def __init__(self):
        self.recorded = []

    def load_transcript(self, omit_user_messages=()):
        self.omitted = tuple(omit_user_messages)
        return ""

    def record_user_message(self, content):
        self.recorded.append(("user_message", content))

    def record_agent_message(self, content):
        self.recorded.append(("agent_message", content))

    def record_reply(self, content):
        self.recorded.append(("poke_reply", content))


@pytest.fixture
def turn_runtime(monkeypatch):
    """The real IA runtime with a scripted LLM whose one tool call captures `current_turn`."""
    seen = []

    def tool(name, args):
        seen.append(current_turn.get())
        return ToolResult(success=True, payload={"ok": True})

    monkeypatch.setattr(ia_runtime_module, "handle_tool_call", tool)
    monkeypatch.setattr(ia_runtime_module, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(ia_runtime_module, "prepare_message_with_history", lambda text, transcript, message_type: [])
    monkeypatch.setattr(ia_runtime_module, "get_agent_directory", lambda: type("D", (), {"set_last_routed": lambda s, ids: None})())

    def make():
        runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
        runtime.settings = type("S", (), {"summarization_enabled": False})()
        runtime.conversation_log = RuntimeLog()
        replies = iter(
            [
                {"choices": [{"message": {"content": "", "tool_calls": [{"id": "1", "function": {"name": "probe", "arguments": "{}"}}]}}]},
                {"choices": [{"message": {"content": "done"}}]},
            ]
        )

        async def llm(system_prompt, messages):
            return next(replies)

        runtime._make_llm_call = llm
        return runtime

    return make, seen


async def test_user_turn_sets_current_turn_and_clears_it(turn_runtime):
    make, seen = turn_runtime
    at = datetime(2026, 10, 28, 12, 0, 0, 5, tzinfo=timezone.utc)
    runtime = make()
    result = await runtime.execute("yes, send it", user_message_at=at, recorded=["yes, send it"])
    assert result.success
    assert seen == [TurnContext(origin="user", user_message_at=at)]
    assert current_turn.get() is None
    # Already in the log: not recorded again, and left out of the history shown before the new message.
    assert runtime.conversation_log.recorded == [("poke_reply", "done")]
    assert runtime.conversation_log.omitted == ("yes, send it",)


async def test_direct_user_turn_records_then_stamps_utc_now(turn_runtime):
    make, seen = turn_runtime
    runtime = make()
    before = datetime.now(timezone.utc)
    await runtime.execute("hi")
    assert runtime.conversation_log.recorded[0] == ("user_message", "hi")
    assert seen[0].origin == "user" and seen[0].user_message_at >= before
    assert seen[0].user_message_at.utcoffset().total_seconds() == 0


async def test_agent_turn_sets_its_origin_and_no_user_time(turn_runtime):
    make, seen = turn_runtime
    await make().handle_agent_message("[SUCCESS] a: done", origin="trigger_result")
    assert seen == [TurnContext(origin="trigger_result", user_message_at=None)]
    assert current_turn.get() is None


async def test_failed_turn_still_clears_current_turn(turn_runtime):
    make, _ = turn_runtime
    runtime = make()

    async def broken(system_prompt, messages):
        raise RuntimeError("boom")

    runtime._make_llm_call = broken
    result = await runtime.handle_agent_message("[FAILED] a: x")
    assert not result.success and current_turn.get() is None


# -- transcript omission ---------------------------------------------------------------------------------------


def test_conversation_log_omits_the_last_matching_user_messages(tmp_path):
    log = ConversationLog(tmp_path / "conv.log")
    log._working_memory_log = WorkingMemoryLog(tmp_path / "wm.log")
    log.record_user_message("yes")
    log.record_reply("Here is the draft")
    log.record_user_message("yes")
    log.record_reply("late reply from another turn")
    transcript = log.load_transcript(omit_user_messages=["yes"])
    assert transcript.count("<user_message") == 1
    assert transcript.index("<user_message") < transcript.index("Here is the draft")
    assert "late reply" in transcript


def test_working_memory_omits_the_last_matching_user_messages(tmp_path):
    memory = WorkingMemoryLog(tmp_path / "wm.log")
    memory.append_entry("user_message", "a", "2026-10-28 12:00:00")
    memory.append_entry("poke_reply", "b", "2026-10-28 12:00:01")
    memory.append_entry("user_message", "a", "2026-10-28 12:00:02")
    rendered = memory.render_transcript(omit_user_messages=["a"])
    assert rendered.count("<user_message") == 1 and "12:00:00" in rendered and "12:00:02" not in rendered
