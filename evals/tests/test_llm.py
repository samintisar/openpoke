import json

import httpx
import pytest

from evals.llm import Budget, BudgetExceeded, OpenRouterLLM, ScriptedLLM, cache_key, estimate_cost, tool_call

KW = dict(model="m", system="s", messages=[{"role": "user", "content": "x"}], tools=[], sample=0)


def _ok(cost=0.01):
    return {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": cost},
    }


def _llm(tmp_path, handler, cap=1.0):
    return OpenRouterLLM(
        api_key="k", cache_dir=tmp_path, budget=Budget(cap),
        transport=httpx.MockTransport(handler), retry_delay_s=0,
    )


async def test_identical_call_is_served_from_cache(tmp_path):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_ok())

    llm = _llm(tmp_path, handler)
    first = await llm.complete(**KW)
    second = await llm.complete(**KW)
    await llm.aclose()

    assert len(sent) == 1
    assert first["_meta"]["cached"] is False and second["_meta"]["cached"] is True
    assert llm.budget.spent_usd == pytest.approx(0.01)
    assert sent[0]["usage"] == {"include": True}
    assert sent[0]["messages"][0] == {"role": "system", "content": "s"}


async def test_each_sample_is_a_separate_call(tmp_path):
    sent = []
    llm = _llm(tmp_path, lambda r: sent.append(1) or httpx.Response(200, json=_ok()))
    await llm.complete(**KW)
    await llm.complete(**{**KW, "sample": 1})
    await llm.aclose()
    assert len(sent) == 2


async def test_budget_cap_blocks_uncached_calls(tmp_path):
    llm = _llm(tmp_path, lambda r: httpx.Response(200, json=_ok()), cap=0.0)
    with pytest.raises(BudgetExceeded):
        await llm.complete(**KW)
    await llm.aclose()


async def test_retries_rate_limit_then_succeeds(tmp_path):
    responses = [httpx.Response(429, json={}), httpx.Response(200, json=_ok())]
    llm = _llm(tmp_path, lambda r: responses.pop(0))
    data = await llm.complete(**KW)
    await llm.aclose()
    assert data["choices"][0]["message"]["content"] == "hi"


def test_estimate_cost_when_usage_has_no_cost():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    assert estimate_cost("anthropic/claude-sonnet-4", usage) == pytest.approx(3.0)


def test_cache_key_depends_on_every_input():
    base = cache_key(**KW)
    assert base == cache_key(**KW)
    assert base != cache_key(**{**KW, "model": "other"})
    assert base != cache_key(**{**KW, "messages": [{"role": "user", "content": "y"}]})


async def test_scripted_llm_returns_tool_calls():
    llm = ScriptedLLM(lambda messages, sample: {"content": "", "tool_calls": [tool_call("wait", reason="r")]})
    data = await llm.complete(**KW)
    call = data["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "wait"
    assert json.loads(call["function"]["arguments"]) == {"reason": "r"}
