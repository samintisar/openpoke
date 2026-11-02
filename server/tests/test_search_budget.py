import json
from datetime import datetime, timezone

import pytest

import server.agents.interaction_agent.runtime as runtime_module
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.agents.interaction_agent.tools import ToolResult
from server.agents.interaction_agent.turn import TurnContext, current_turn
from server.services.agents.dispatch import SEARCH_BUDGET, SEARCH_BUDGET_EXHAUSTED

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def _runtime(rounds):
    """An agent that makes one round of tool calls per entry, then answers."""
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    replies = iter([
        *({"choices": [{"message": {"content": "", "tool_calls": [
            {"id": f"{r}-{i}", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)]}}]} for r, calls in enumerate(rounds)),
        {"choices": [{"message": {"content": "done"}}]},
    ])

    async def fake_llm(system_prompt, messages):
        return next(replies)

    runtime._make_llm_call = fake_llm
    return runtime


@pytest.fixture
def executed(monkeypatch):
    seen = []

    def fake_tool(name, args):
        seen.append((name, args))
        return ToolResult(success=True, payload="cards")

    monkeypatch.setattr(runtime_module, "handle_tool_call", fake_tool)
    return seen


async def _loop(runtime):
    token = current_turn.set(TurnContext(origin="agent_result", latest_text="report"))
    try:
        messages: list = []
        summary = await runtime._run_interaction_loop("system", messages)
        return summary, [m for m in messages if m["role"] == "tool"]
    finally:
        current_turn.reset(token)


def _searches(n):
    return [[("search_agents", {"query": f"q{i}"})] for i in range(n)]


async def test_fourth_search_in_a_turn_is_refused(executed):
    assert SEARCH_BUDGET == 3
    summary, tool_messages = await _loop(_runtime(_searches(4)))
    assert [name for name, _ in executed] == ["search_agents"] * 3  # the fourth never ran
    results = [json.loads(m["content"]) for m in tool_messages]
    assert [r["status"] for r in results] == ["success"] * 3 + ["error"]
    assert results[3]["error"] == SEARCH_BUDGET_EXHAUSTED


async def test_refusal_also_applies_within_one_round(executed):
    calls = [[("search_agents", {"query": f"q{i}"}) for i in range(5)]]
    _, tool_messages = await _loop(_runtime(calls))
    assert len(executed) == 3
    assert [json.loads(m["content"])["status"] for m in tool_messages] == ["success"] * 3 + ["error"] * 2


async def test_other_tools_are_unaffected_after_the_budget(executed):
    rounds = _searches(4) + [[("send_message_to_agent", {"agent_id": "a", "instructions": "x"})]]
    await _loop(_runtime(rounds))
    assert [name for name, _ in executed][-1] == "send_message_to_agent"


async def test_budget_resets_next_turn(executed):
    await _loop(_runtime(_searches(4)))
    await _loop(_runtime(_searches(3)))
    assert len(executed) == 6
