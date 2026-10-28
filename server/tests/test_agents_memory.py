"""Per-agent memory compaction: budget split, identifiers, compaction parsing, rendering and orchestration."""

import asyncio
import json
from datetime import datetime, timezone

import pytest

import server.agents.execution_agent.agent as agent_module
import server.agents.execution_agent.batch_manager as batch_module
import server.services.agents.memory as memory
from server.agents.execution_agent.agent import ExecutionAgent
from server.agents.execution_agent.runtime import ExecutionResult
from server.services.agents import AgentCard, InMemoryDirectory
from server.services.agents.memory import (
    COMPACTION_PROMPT,
    CompactionError,
    CompactionResult,
    Entry,
    compact,
    estimate_tokens,
    extract_identifiers,
    maybe_compact,
    render_entry,
    render_history,
    split_for_budget,
)
from server.services.execution.log_store import ExecutionAgentLogStore

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
THREAD = "18c9a1b2c3d4e5f6"
MESSAGE = "18c9a1b2c3d4e5f7"
DRAFT = "r-4815162342108815"


def entry(tag: str, text: str) -> Entry:
    return Entry(tag, text, "2026-10-28 09:00:00")


def sized(n: int, tokens: int) -> Entry:
    """An entry whose rendered size is `tokens - 1` tokens (the split counts +1 per entry for the newline)."""
    base = render_entry(entry("agent_request", ""))
    return entry("agent_request", f"{n}".ljust((tokens - 1) * 4 - len(base), "x"))


def card(agent_id: str = "gary-lease", purpose: str = "Negotiate Gary's lease renewal") -> AgentCard:
    return AgentCard(agent_id, "Gary Lease", purpose, ["gary"], created_at=NOW, last_active_at=NOW, run_count=3)


def fill(logs: ExecutionAgentLogStore, agent_id: str, runs: int, *, chars: int = 400) -> None:
    for run in range(runs):
        logs.record_request(agent_id, f"Run {run}: follow up with Gary about the lease. " + "x" * chars)
        logs.record_action(agent_id, f'Calling gmail_reply_to_thread with: {{"thread_id": "{THREAD}"}}')
        logs.record_tool_response(agent_id, "gmail_reply_to_thread", '{"successful": true}')
        logs.record_agent_response(agent_id, f"Replied to Gary (run {run}).")


def scripted(payload) -> tuple[list[tuple[str, str]], "memory.Complete"]:
    calls: list[tuple[str, str]] = []

    async def complete(system: str, user: str) -> str:
        calls.append((system, user))
        return payload if isinstance(payload, str) else json.dumps(payload)

    return calls, complete


GOOD = {"memory": "Gary agreed to $4,200. User wants formal replies.", "purpose": "Renew Gary's lease",
        "last_status": "Waiting on the signed copy\nfrom Gary"}


# -- split_for_budget ---------------------------------------------------------------------------------------


def test_estimate_tokens_is_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefg") == 1
    assert estimate_tokens("x" * 400) == 100


def test_split_keeps_newest_entries_that_fit():
    entries = [sized(i, 100) for i in range(10)]
    prefix, tail = split_for_budget(entries, 300)
    assert tail == entries[7:] and prefix == entries[:7]


def test_split_everything_fits_and_empty():
    entries = [sized(i, 10) for i in range(5)]
    assert split_for_budget(entries, 1000) == ([], entries)
    assert split_for_budget([], 1000) == ([], [])


def test_split_newest_entry_over_budget_leaves_empty_tail():
    entries = [sized(0, 10), sized(1, 500)]
    assert split_for_budget(entries, 100) == (entries, [])


def test_split_skips_entries_already_in_memory():
    entries = [sized(i, 100) for i in range(10)]
    prefix, tail = split_for_budget(entries, 300, upto=4)
    assert prefix == entries[4:7] and tail == entries[7:]
    # The tail never reaches back into the memory, and an out-of-range `upto` is clamped.
    assert split_for_budget(entries, 10_000, upto=8) == ([], entries[8:])
    assert split_for_budget(entries, 10_000, upto=50) == ([], [])


# -- identifiers ---------------------------------------------------------------------------------------------


def test_extract_identifiers_from_gmail_tool_payloads():
    args = {"recipient_email": "gary.whitman@example.com", "cc": ["priya.shah@lease-co.co.uk"],
            "subject": "Lease", "body": "Hi Gary", "thread_id": THREAD}
    texts = [
        "Draft a reply to Gary (gary.whitman@example.com) about the lease.",
        f"Calling gmail_create_draft with: {json.dumps(args)[:200]}",
        "gmail_create_draft: " + json.dumps(
            {"data": {"response_data": {"id": DRAFT, "message": {"id": MESSAGE, "threadId": THREAD,
                                                                   "labelIds": ["DRAFT"]}}},
             "error": None, "successful": True}
        ),
        f"GMAIL_SEND_DRAFT succeeded | args={json.dumps({'draft_id': DRAFT})}",
        f"Calling gmail_forward_email with: message_id={MESSAGE} recipient_email=landlord@example.org",
        "Tool said: draft_id: r-99887766 and thread_id: None",
    ]
    assert extract_identifiers(texts) == [
        "gary.whitman@example.com", "priya.shah@lease-co.co.uk", THREAD, DRAFT, MESSAGE,
        "landlord@example.org", "r-99887766",
    ]


def test_extract_identifiers_handles_escaped_json_and_camel_case():
    nested = json.dumps({"content": json.dumps({"draftId": "r1234567890", "messageId": "msg-abc-1"})})
    assert extract_identifiers([nested]) == ["r1234567890", "msg-abc-1"]


def test_extract_identifiers_dedupes_case_insensitively_and_ignores_noise():
    texts = [
        "Email Gary@Example.com, then gary@example.com again.",
        "Booked for 2026-10-28 at 1700, ref 12345, price $4,200, sha deadbeefcafebabe.",
        "The local part 1234567890abcdef1@example.com is an email, not a thread id.",
    ]
    assert extract_identifiers(texts) == ["Gary@Example.com", "1234567890abcdef1@example.com"]


# -- compact -------------------------------------------------------------------------------------------------


def test_compaction_prompt_asks_for_json_and_says_identifiers_are_kept():
    assert '{"memory": "...", "purpose": "...", "last_status": "..."}' in COMPACTION_PROMPT
    assert "300 words" in COMPACTION_PROMPT and "preserved separately" in COMPACTION_PROMPT
    for kept in ("people", "decisions", "commitments", "preferences", "open loops"):
        assert kept in COMPACTION_PROMPT


async def test_compact_parses_json_and_sends_prefix_memory_and_purpose():
    calls, complete = scripted(GOOD)
    prefix = [entry("agent_request", "Ask Gary about the lease")]
    result = await compact(prefix, "Earlier: Gary is the landlord.", "Lease with Gary", complete=complete)
    assert result == CompactionResult(GOOD["memory"], "Renew Gary's lease", "Waiting on the signed copy from Gary")
    system, user = calls[0]
    assert system == COMPACTION_PROMPT
    assert "Purpose: Lease with Gary" in user and "Earlier: Gary is the landlord." in user
    assert render_entry(prefix[0]) in user


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Here is the memory: " + json.dumps(GOOD) + " Done.",
    ],
)
async def test_compact_tolerates_fences_and_surrounding_text(raw):
    _, complete = scripted(raw)
    assert (await compact([], "", "p", complete=complete)).memory == GOOD["memory"]


async def test_compact_keeps_old_purpose_when_missing():
    _, complete = scripted({"memory": "m"})
    assert await compact([], "", "Old purpose", complete=complete) == CompactionResult("m", "Old purpose", "")


@pytest.mark.parametrize(
    "raw",
    ["not json", "{broken", "[1, 2]", json.dumps({"memory": "   "}), json.dumps({"memory": 3}),
     json.dumps({"purpose": "x"}), ""],
)
async def test_compact_rejects_invalid_or_empty_output(raw):
    _, complete = scripted(raw)
    with pytest.raises(CompactionError):
        await compact([], "", "p", complete=complete)


async def test_compact_wraps_call_failures():
    async def boom(system, user):
        raise RuntimeError("503")

    with pytest.raises(CompactionError, match="503"):
        await compact([], "", "p", complete=boom)


# -- render ---------------------------------------------------------------------------------------------------


def test_render_history_format():
    tail = [entry("agent_request", "Send it <now> & thanks"), Entry("agent_response", "Sent.")]
    assert render_history("Gary & Priya agreed.", ["gary@example.com", THREAD], tail) == (
        "<agent_memory>Gary &amp; Priya agreed.</agent_memory>\n"
        f"<agent_identifiers>gary@example.com, {THREAD}</agent_identifiers>\n"
        '<agent_request timestamp="2026-10-28 09:00:00">Send it &lt;now&gt; &amp; thanks</agent_request>\n'
        "<agent_response>Sent.</agent_response>"
    )


def test_rendered_tail_matches_load_transcript(tmp_path):
    logs = ExecutionAgentLogStore(tmp_path)
    fill(logs, "a", 2, chars=5)
    entries = memory.load_entries(logs, "a")
    assert memory.render_entries(entries) == logs.load_transcript("a")


# -- log store ------------------------------------------------------------------------------------------------


def test_memory_entries_are_skipped_by_transcript_and_iteration(tmp_path):
    logs = ExecutionAgentLogStore(tmp_path)
    logs.record_request("a", "first")
    assert logs.latest_memory("a") is None
    logs.record_memory("a", "Line one\nLine two with </agent_memory> & \"quotes\"", ["x@y.com"], 1)
    logs.record_agent_response("a", "done")
    logs.record_memory("a", "newer", ["x@y.com", THREAD], 2)

    assert [tag for tag, _, _ in logs.iter_entries("a")] == ["agent_request", "agent_response"]
    assert "agent_memory" not in logs.load_transcript("a") and "newer" not in logs.load_transcript("a")
    assert [tag for tag, _, _ in logs.load_recent("a")] == ["agent_request", "agent_response"]
    latest = logs.latest_memory("a")
    assert (latest.memory, latest.identifiers, latest.upto) == ("newer", ["x@y.com", THREAD], 2)
    raw = (tmp_path / "a.log").read_text(encoding="utf-8")
    assert raw.count("<agent_memory ") == 2 and 'upto="2"' in raw


def test_latest_memory_round_trips_newlines(tmp_path):
    logs = ExecutionAgentLogStore(tmp_path)
    logs.record_memory("a", "Line one\nLine two", [], 0)
    assert logs.latest_memory("a").memory == "Line one\nLine two"


# -- execution agent prompt -----------------------------------------------------------------------------------


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    directory = InMemoryDirectory([card()])
    monkeypatch.setattr(agent_module, "get_execution_agent_logs", lambda: logs)
    monkeypatch.setattr(agent_module, "get_agent_directory", lambda: directory)
    return logs, directory


def test_system_prompt_uses_card_purpose_falling_back_to_name(agent_env):
    _, directory = agent_env
    assert "Purpose: Negotiate Gary's lease renewal" in ExecutionAgent("gary-lease", "Gary Lease").build_system_prompt()
    directory.update_card("gary-lease", purpose="  ")
    assert "Purpose: Gary Lease" in ExecutionAgent("gary-lease", "Gary Lease").build_system_prompt()
    assert "Purpose: Unknown Agent" in ExecutionAgent("missing", "Unknown Agent").build_system_prompt()


def test_system_prompt_without_memory_is_the_full_transcript(agent_env):
    logs, _ = agent_env
    fill(logs, "gary-lease", 3, chars=5)
    prompt = ExecutionAgent("gary-lease", "Gary Lease").build_system_prompt_with_history()
    assert prompt.endswith("# Execution History\n\n" + logs.load_transcript("gary-lease"))
    assert "<agent_memory>" not in prompt


def test_system_prompt_with_memory_renders_memory_identifiers_and_entries_after_it(agent_env):
    logs, _ = agent_env
    fill(logs, "gary-lease", 3, chars=5)
    logs.record_memory("gary-lease", "Gary agreed to $4,200.", ["gary@example.com", THREAD], 8)
    entries = memory.load_entries(logs, "gary-lease")
    prompt = ExecutionAgent("gary-lease", "Gary Lease").build_system_prompt_with_history()
    expected = render_history("Gary agreed to $4,200.", ["gary@example.com", THREAD], entries[8:])
    assert prompt.endswith("# Execution History\n\n" + expected)
    assert "Run 0" not in prompt and "Run 1" not in prompt and "Run 2" in prompt


# -- maybe_compact --------------------------------------------------------------------------------------------


@pytest.fixture
def compact_env(tmp_path):
    memory._LOCKS.clear()
    return ExecutionAgentLogStore(tmp_path / "logs"), InMemoryDirectory([card()])


async def test_maybe_compact_skips_under_budget(compact_env):
    logs, directory = compact_env
    fill(logs, "gary-lease", 3)
    calls, complete = scripted(GOOD)
    assert not await maybe_compact("gary-lease", log_store=logs, directory=directory, complete=complete,
                                   budget_tokens=4000)
    assert calls == [] and logs.latest_memory("gary-lease") is None


async def test_maybe_compact_records_memory_and_refreshes_card(compact_env):
    logs, directory = compact_env
    logs.record_request("gary-lease", "Email gary.whitman@example.com about the lease")
    fill(logs, "gary-lease", 12)
    entries = memory.load_entries(logs, "gary-lease")
    prefix, tail = split_for_budget(entries, 600)
    calls, complete = scripted(GOOD)

    assert await maybe_compact("gary-lease", log_store=logs, directory=directory, complete=complete,
                               budget_tokens=600)
    saved = logs.latest_memory("gary-lease")
    assert saved.memory == GOOD["memory"] and saved.upto == len(prefix)
    assert saved.identifiers == ["gary.whitman@example.com", THREAD]
    assert "Purpose: Negotiate Gary's lease renewal" in calls[0][1]
    refreshed = directory.get("gary-lease")
    assert refreshed.purpose == "Renew Gary's lease" and refreshed.last_status == "Waiting on the signed copy from Gary"
    assert memory.load_entries(logs, "gary-lease") == entries  # raw entries untouched

    # The next compaction folds only what came after the memory, into the existing memory and identifiers.
    fill(logs, "gary-lease", 12)
    calls2, complete2 = scripted({"memory": "Even newer.", "purpose": "p", "last_status": "s"})
    assert await maybe_compact("gary-lease", log_store=logs, directory=directory, complete=complete2,
                               budget_tokens=600)
    assert GOOD["memory"] in calls2[0][1] and "Email gary.whitman" not in calls2[0][1]
    second = logs.latest_memory("gary-lease")
    assert second.upto > saved.upto and second.identifiers == ["gary.whitman@example.com", THREAD]


async def test_maybe_compact_keeps_full_history_on_failure(compact_env):
    logs, directory = compact_env
    fill(logs, "gary-lease", 12)
    transcript = logs.load_transcript("gary-lease")
    _, bad = scripted("not json")
    assert not await maybe_compact("gary-lease", log_store=logs, directory=directory, complete=bad, budget_tokens=600)
    assert logs.latest_memory("gary-lease") is None and logs.load_transcript("gary-lease") == transcript
    assert directory.get("gary-lease").purpose == "Negotiate Gary's lease renewal"

    # It retries after the next run.
    _, good = scripted(GOOD)
    assert await maybe_compact("gary-lease", log_store=logs, directory=directory, complete=good, budget_tokens=600)


async def test_maybe_compact_never_raises(compact_env):
    logs, directory = compact_env
    fill(logs, "gary-lease", 12)

    class BrokenDirectory:
        def get(self, agent_id):
            raise RuntimeError("db locked")

    assert not await maybe_compact("gary-lease", log_store=logs, directory=BrokenDirectory(),
                                   complete=scripted(GOOD)[1], budget_tokens=600)


async def test_maybe_compact_takes_the_per_agent_lock(compact_env):
    logs, directory = compact_env
    fill(logs, "gary-lease", 12)
    seen_locked = []

    async def complete(system, user):
        seen_locked.append(memory.compaction_lock("gary-lease").locked())
        await asyncio.sleep(0)
        return json.dumps(GOOD)

    first, second = await asyncio.gather(
        maybe_compact("gary-lease", log_store=logs, directory=directory, complete=complete, budget_tokens=600),
        maybe_compact("gary-lease", log_store=logs, directory=directory, complete=complete, budget_tokens=600),
    )
    # The second waits for the first, then finds nothing left to compact.
    assert (first, second) == (True, False) and seen_locked == [True]
    assert memory.compaction_lock("gary-lease") is not memory.compaction_lock("other")


async def test_maybe_compact_default_complete_uses_summarizer_model(compact_env, monkeypatch):
    logs, directory = compact_env
    fill(logs, "gary-lease", 12)
    requests = []

    async def fake_request(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps(GOOD)}}]}

    monkeypatch.setattr(memory, "request_chat_completion", fake_request)
    assert await maybe_compact("gary-lease", log_store=logs, directory=directory, budget_tokens=600)
    assert requests[0]["model"] == memory.get_settings().summarizer_model
    assert requests[0]["system"] == COMPACTION_PROMPT


async def test_maybe_compact_defaults_budget_to_setting(compact_env):
    logs, directory = compact_env
    assert memory.get_settings().history_token_budget == 4000
    fill(logs, "gary-lease", 12)  # well under 4000 tokens
    calls, complete = scripted(GOOD)
    assert not await maybe_compact("gary-lease", log_store=logs, directory=directory, complete=complete)
    assert calls == []


# -- batch manager hook ---------------------------------------------------------------------------------------


async def test_batch_manager_schedules_compaction_after_the_run(monkeypatch):
    directory = InMemoryDirectory([card()])
    order = []
    started = asyncio.Event()
    release = asyncio.Event()

    class FakeRuntime:
        def __init__(self, agent_id, agent_name=None):
            pass

        async def execute(self, instructions):
            order.append("run")
            return ExecutionResult(agent_name="Gary Lease", agent_id="gary-lease", success=True, response="ok")

    async def slow_compact(agent_id):
        order.append(("compact", agent_id))
        started.set()
        await release.wait()
        raise RuntimeError("boom")  # a failing compaction never reaches the run

    async def capture(self, payload):
        order.append("dispatch")

    monkeypatch.setattr(batch_module, "get_agent_directory", lambda: directory)
    monkeypatch.setattr(batch_module, "ExecutionAgentRuntime", FakeRuntime)
    monkeypatch.setattr(batch_module, "maybe_compact", slow_compact)
    monkeypatch.setattr(batch_module.ExecutionBatchManager, "_dispatch_to_interaction_agent", capture)

    result = await batch_module.ExecutionBatchManager().execute_agent("gary-lease", "Follow up")
    assert result.success and order == ["run", "dispatch"]  # returned without waiting on compaction
    await asyncio.wait_for(started.wait(), 1)
    assert order[-1] == ("compact", "gary-lease")
    release.set()
    for _ in range(3):
        await asyncio.sleep(0)
    assert not batch_module._COMPACTION_TASKS
