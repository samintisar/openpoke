import asyncio
import json
from datetime import datetime, timezone

import pytest

import server.agents.interaction_agent.runtime as runtime_module
import server.agents.interaction_agent.second_look_gate as gate_module
from server.agents.interaction_agent.runtime import InteractionAgentRuntime, _LoopSummary, _ToolCall
from server.agents.interaction_agent.second_look_gate import gate
from server.agents.interaction_agent.tools import ToolResult
from server.agents.interaction_agent.turn import TurnContext, current_turn
from server.services.agents.cards import AgentCard, extract_entities
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.second_look import SecondLookConfig

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
GAME_NIGHT = "ask Kwame what he thinks we should do for game night"


def _card(agent_id, name, purpose):
    return AgentCard(id=agent_id, name=name, purpose=purpose, entities=extract_entities([name, purpose]),
                     created_at=NOW, last_active_at=NOW)


def _dir():
    return InMemoryDirectory([_card("kwame-book", "Kwame Book Club Dates", "Ask Kwame which dates work for book club.")])


def _verdict(payload):
    async def complete(system, user):
        return json.dumps(payload)
    return complete


async def test_new_subject_holds_a_reuse():
    args, held, record = await gate({"agent_id": "kwame-book", "instructions": "x"},
                                    message=GAME_NIGHT, context=[],
                                    directory=_dir(), now=NOW, complete=_verdict({"decision": "new", "reason": "r"}))
    assert held["status"] == "held_by_second_look" and record["trigger"] == "reuse" and record["effect"] == "hold"
    assert record["decision"] == "new" and record["why"] == "r"


async def test_no_trigger_means_no_call():
    async def never(system, user):
        raise AssertionError("called")
    args, held, record = await gate({"agent_id": "kwame-book", "instructions": "x"},
                                    message="ask Kwame about the book club dates", context=[], directory=_dir(),
                                    now=NOW, complete=never)
    assert held is None and record is None and args["agent_id"] == "kwame-book"


async def test_failures_let_the_decision_stand():
    async def broken(system, user):
        raise RuntimeError("down")
    args, held, record = await gate({"agent_id": "kwame-book", "instructions": "x"},
                                    message=GAME_NIGHT, context=[],
                                    directory=_dir(), now=NOW, complete=broken)
    assert held is None and args["agent_id"] == "kwame-book" and record["effect"] == "error"
    assert "down" in record["error"]


async def test_timeout_uses_the_config_and_lets_the_decision_stand():
    async def slow(system, user):
        await asyncio.sleep(5)
        return "{}"
    args, held, record = await gate({"agent_id": "kwame-book", "instructions": "x"},
                                    message=GAME_NIGHT, context=[], directory=_dir(), now=NOW, complete=slow,
                                    cfg=SecondLookConfig(timeout_s=0.01))
    assert held is None and args["agent_id"] == "kwame-book" and record["effect"] == "error"


async def test_create_redirects_to_the_owner():
    args, held, record = await gate({"name": "Kwame Book Club Check", "purpose": "Ask Kwame again about book club",
                                     "instructions": "x", "force_new": True},
                                    message="any word from Kwame on the book club dates?", context=[],
                                    directory=_dir(), now=NOW,
                                    complete=_verdict({"decision": "reuse", "agent_id": "kwame-book", "reason": "r"}))
    assert held is None and args == {"agent_id": "kwame-book", "instructions": "x"} and record["effect"] == "redirect"
    assert record["trigger"] == "create"


async def test_create_the_duplicate_guard_would_catch_is_left_to_it():
    async def never(system, user):
        raise AssertionError("called")
    arguments = {"name": "Kwame Book Club Dates", "purpose": "Ask Kwame which dates work for book club.",
                 "instructions": "x"}
    args, held, record = await gate(dict(arguments), message="ask Kwame about book club dates", context=[],
                                    directory=_dir(), now=NOW, complete=never)
    assert args == arguments and held is None and record is None


async def test_unknown_ids_and_malformed_calls_pass_through():
    async def never(system, user):
        raise AssertionError("called")
    for arguments in ({"agent_id": "nobody", "instructions": "x"}, {"instructions": "x"}):
        args, held, record = await gate(dict(arguments), message=GAME_NIGHT, context=[], directory=_dir(), now=NOW,
                                        complete=never)
        assert args == arguments and held is None and record is None


# -- runtime wiring ------------------------------------------------------------------------------------------


def _runtime(calls, log=None):
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    runtime.conversation_log = log
    replies = iter([
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": str(i), "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(
                call if isinstance(call, tuple) else ("send_message_to_agent", call) for call in calls)
        ]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ])

    async def fake_llm(system_prompt, messages):
        return next(replies)

    runtime._make_llm_call = fake_llm
    return runtime


@pytest.fixture
def wired(monkeypatch):
    seen = {"gate": [], "tools": []}

    async def fake_gate(arguments, *, message, context, directory, now, complete, **kwargs):
        seen["gate"].append((dict(arguments), message, list(context)))
        seen["gate_kwargs"] = kwargs
        return arguments, {"status": "held_by_second_look"}, {"trigger": "reuse", "effect": "hold"}

    def fake_tool(name, args):
        seen["tools"].append(args)
        return ToolResult(success=True, payload={"status": "submitted", "agent_id": args.get("agent_id")})

    monkeypatch.setattr(runtime_module, "second_look_gate", fake_gate)
    monkeypatch.setattr(runtime_module, "handle_tool_call", fake_tool)
    monkeypatch.setattr(runtime_module, "get_agent_directory", _dir)
    return seen


async def _loop(runtime, turn):
    token = current_turn.set(turn)
    try:
        messages: list = []
        summary = await runtime._run_interaction_loop("system", messages)
        return summary, messages
    finally:
        current_turn.reset(token)


async def test_runtime_holds_once_per_user_turn(wired):
    calls = [{"agent_id": "kwame-book", "instructions": "x"}, {"agent_id": "kwame-book", "instructions": "y"}]
    summary, messages = await _loop(_runtime(calls), TurnContext(origin="user", user_message_at=NOW,
                                                                 latest_text=GAME_NIGHT, context=("earlier",)))
    assert [g[1] for g in wired["gate"]] == [GAME_NIGHT]  # one look per turn
    assert wired["gate"][0][2] == ["earlier"]  # the turn's context snapshot
    assert wired["gate_kwargs"].get("guard", True) is True  # production keeps the duplicate guard
    assert wired["tools"] == [{"agent_id": "kwame-book", "instructions": "y"}]
    held = json.loads(messages[1]["content"])
    assert held["status"] == "error" and held["error"]["status"] == "held_by_second_look"
    assert summary.dispatched_ids == ["kwame-book"] and summary.second_look_done


async def test_runtime_never_looks_on_agent_turns(wired):
    calls = [{"agent_id": "kwame-book", "instructions": "x"}]
    summary, _ = await _loop(_runtime(calls), TurnContext(origin="agent_result", latest_text="report"))
    assert wired["gate"] == [] and wired["tools"] == calls


async def test_runtime_redirect_rewrites_the_call(wired, monkeypatch):
    async def redirect(arguments, **kwargs):
        return {"agent_id": "other", "instructions": arguments["instructions"]}, None, {"effect": "redirect"}

    monkeypatch.setattr(runtime_module, "second_look_gate", redirect)
    calls = [{"name": "N", "purpose": "P", "instructions": "x"}]
    summary, messages = await _loop(_runtime(calls), TurnContext(origin="user", latest_text="m"))
    assert wired["tools"] == [{"agent_id": "other", "instructions": "x"}]
    assert json.loads(messages[1]["content"])["arguments"] == {"agent_id": "other", "instructions": "x"}


async def test_runtime_gate_crash_keeps_the_call(wired, monkeypatch):
    async def crash(arguments, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_module, "second_look_gate", crash)
    calls = [{"agent_id": "kwame-book", "instructions": "x"}]
    summary, _ = await _loop(_runtime(calls), TurnContext(origin="user", latest_text="m"))
    assert wired["tools"] == calls and summary.dispatched_ids == ["kwame-book"]


async def test_runtime_no_trigger_leaves_the_look_available(wired, monkeypatch):
    looks = []

    async def quiet(arguments, **kwargs):
        looks.append(arguments)
        return arguments, None, None

    monkeypatch.setattr(runtime_module, "second_look_gate", quiet)
    calls = [{"agent_id": "kwame-book", "instructions": "x"}, {"agent_id": "kwame-book", "instructions": "y"}]
    summary, _ = await _loop(_runtime(calls), TurnContext(origin="user", latest_text="m"))
    assert len(looks) == 2 and not summary.second_look_done and len(wired["tools"]) == 2


async def test_production_complete_uses_the_cheap_model(monkeypatch):
    sent = {}

    async def fake_request(**kwargs):
        sent.update(kwargs)
        return {"choices": [{"message": {"content": " {\"decision\": \"new\"} "}}]}

    monkeypatch.setattr(gate_module, "request_chat_completion", fake_request)
    assert await gate_module.openrouter_complete("sys", "usr") == '{"decision": "new"}'
    assert sent["model"] == gate_module.get_settings().summarizer_model and sent["system"] == "sys"


async def test_runtime_context_is_snapshotted_before_the_turn_writes(wired, monkeypatch, tmp_path):
    """The context is the turns before the message, even after the IA acks mid-loop (a poke_reply)."""
    from server.services.conversation.log import ConversationLog

    log = ConversationLog(tmp_path / "log.txt")
    for i in range(4):
        log.record_user_message(f"user {i}")
        log.record_reply(f"reply {i}")
    log.record_user_message("already posted")  # recorded by the inbox before the turn

    def fake_tool(name, args):
        if name == "send_message_to_user":
            log.record_reply(args["message"])
        wired["tools"].append(args)
        return ToolResult(success=True, payload={"status": "submitted", "agent_id": args.get("agent_id")})

    monkeypatch.setattr(runtime_module, "handle_tool_call", fake_tool)
    monkeypatch.setattr(runtime_module, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(runtime_module, "prepare_message_with_history", lambda text, transcript, message_type: [])
    calls = [("send_message_to_user", {"message": "on it"}), {"agent_id": "kwame-book", "instructions": "x"}]
    runtime = _runtime(calls, log)
    runtime.settings = type("S", (), {"summarization_enabled": False})()
    result = await runtime.execute("already posted", recorded=["already posted"])
    assert result.success
    assert wired["gate"][0][2] == ["user 1", "reply 1", "user 2", "reply 2", "user 3", "reply 3"]

    log.clear()
    log.record_user_message("before")
    log.record_reply("ack")
    wired["gate"].clear()
    runtime = _runtime(calls, log)
    runtime.settings = type("S", (), {"summarization_enabled": False})()
    await runtime.execute("fresh message")  # recorded by execute itself
    assert wired["gate"][0][2] == ["before", "ack"]


async def test_noguard_looks_at_an_obvious_duplicate():
    arguments = {"name": "Kwame Book Club Dates", "purpose": "Ask Kwame which dates work for book club.",
                 "instructions": "x"}
    args, held, record = await gate(dict(arguments), message="ask Kwame about book club dates", context=[],
                                    directory=_dir(), now=NOW, guard=False,
                                    complete=_verdict({"decision": "reuse", "agent_id": "kwame-book", "reason": "r"}))
    assert record is not None and record["trigger"] == "create" and record["effect"] == "redirect"
    assert args == {"agent_id": "kwame-book", "instructions": "x"} and held is None
