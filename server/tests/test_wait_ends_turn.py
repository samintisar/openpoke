import json

import pytest

import server.agents.interaction_agent.runtime as runtime_module
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.agents.interaction_agent.tools import ToolResult
from server.agents.interaction_agent.turn import TurnContext, current_turn


def _runtime(rounds):
    """An agent that makes one round of tool calls per entry, then answers; counts its LLM calls."""
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    replies = iter([
        *({"choices": [{"message": {"content": "", "tool_calls": [
            {"id": f"{r}-{i}", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)]}}]} for r, calls in enumerate(rounds)),
        {"choices": [{"message": {"content": "done"}}]},
    ])
    runtime.llm_calls = 0

    async def fake_llm(system_prompt, messages):
        runtime.llm_calls += 1
        return next(replies)

    runtime._make_llm_call = fake_llm
    return runtime


@pytest.fixture
def executed(monkeypatch):
    seen = []

    def fake_tool(name, args):
        seen.append(name)
        return ToolResult(success=True, payload={"status": "ok"})

    monkeypatch.setattr(runtime_module, "handle_tool_call", fake_tool)
    return seen


async def _loop(runtime):
    token = current_turn.set(TurnContext(origin="user", latest_text="hi"))
    try:
        return await runtime._run_interaction_loop("system", [])
    finally:
        current_turn.reset(token)


WAIT = ("wait", {"reason": "already answered"})


async def test_a_round_of_only_wait_ends_the_turn(executed):
    # Sonnet 5.5 repeated wait until the 8-iteration limit raised; one wait now ends the turn.
    runtime = _runtime([[WAIT]] * 10)
    summary = await _loop(runtime)
    assert executed == ["wait"]
    assert runtime.llm_calls == 1
    assert summary.tool_names == ["wait"]


async def test_wait_after_a_reply_ends_the_turn(executed):
    runtime = _runtime([[("send_message_to_user", {"message": "on it"})], [WAIT], [WAIT]])
    await _loop(runtime)
    assert executed == ["send_message_to_user", "wait"]
    assert runtime.llm_calls == 2


async def test_wait_beside_another_tool_does_not_end_the_turn(executed):
    runtime = _runtime([[("send_message_to_user", {"message": "on it"}), WAIT]])
    await _loop(runtime)
    assert executed == ["send_message_to_user", "wait"]
    assert runtime.llm_calls == 2  # the model still sees the round's results and answers
