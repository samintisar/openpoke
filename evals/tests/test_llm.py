import json

import httpx
import pytest

from evals.llm import (
    Budget,
    BudgetExceeded,
    OpenRouterLLM,
    RetryableResponseError,
    ScriptedLLM,
    cache_key,
    estimate_cost,
    fake_llm,
    tool_call,
)

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


def _llm_with_sleeps(tmp_path, handler, sleeps, max_retries=3):
    async def fake_sleep(seconds):
        sleeps.append(seconds)

    return OpenRouterLLM(
        api_key="k", cache_dir=tmp_path, budget=Budget(1.0), transport=httpx.MockTransport(handler),
        retry_delay_s=0, max_retries=max_retries, sleep=fake_sleep,
    )


async def test_malformed_200_is_charged_not_cached_and_retried(tmp_path):
    bad = {"choices": [{"message": None}], "usage": {"cost": 0.02}}
    responses = [httpx.Response(200, json=bad), httpx.Response(200, json=_ok())]
    llm = _llm(tmp_path, lambda r: responses.pop(0))
    data = await llm.complete(**KW)
    await llm.aclose()
    assert data["choices"][0]["message"]["content"] == "hi"
    assert llm.budget.spent_usd == pytest.approx(0.03)
    assert data["_meta"]["cost"] == pytest.approx(0.03)


async def test_finish_reason_error_is_not_cached(tmp_path):
    bad = {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "error"}], "usage": {}}
    llm = _llm(tmp_path, lambda r: httpx.Response(200, json=bad))
    llm.max_retries = 1
    with pytest.raises(RetryableResponseError):
        await llm.complete(**KW)
    await llm.aclose()
    assert not list(tmp_path.glob("*.json"))


async def test_missing_choices_is_rejected(tmp_path):
    llm = _llm(tmp_path, lambda r: httpx.Response(200, json={"usage": {"cost": 0.5}}))
    llm.max_retries = 0
    with pytest.raises(RetryableResponseError):
        await llm.complete(**KW)
    await llm.aclose()
    assert llm.budget.spent_usd == pytest.approx(0.5)
    assert not list(tmp_path.glob("*.json"))


async def test_transport_errors_are_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("slow", request=request)
        if len(calls) == 2:
            raise httpx.RemoteProtocolError("bad", request=request)
        return httpx.Response(200, json=_ok())

    sleeps = []
    llm = _llm_with_sleeps(tmp_path, handler, sleeps)
    data = await llm.complete(**KW)
    await llm.aclose()
    assert data["choices"][0]["message"]["content"] == "hi"
    assert len(calls) == 3 and len(sleeps) == 2


async def test_transport_error_gives_up_after_retry_budget(tmp_path):
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    llm = _llm_with_sleeps(tmp_path, handler, [], max_retries=2)
    with pytest.raises(httpx.ConnectError):
        await llm.complete(**KW)
    await llm.aclose()


async def test_408_is_retried(tmp_path):
    responses = [httpx.Response(408), httpx.Response(200, json=_ok())]
    llm = _llm(tmp_path, lambda r: responses.pop(0))
    assert (await llm.complete(**KW))["choices"][0]["message"]["content"] == "hi"
    await llm.aclose()


async def test_200_with_retryable_error_code_is_retried(tmp_path):
    responses = [httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}}), httpx.Response(200, json=_ok())]
    llm = _llm(tmp_path, lambda r: responses.pop(0))
    assert (await llm.complete(**KW))["choices"][0]["message"]["content"] == "hi"
    await llm.aclose()


async def test_200_with_fatal_error_code_raises(tmp_path):
    llm = _llm(tmp_path, lambda r: httpx.Response(200, json={"error": {"code": 400, "message": "bad"}}))
    with pytest.raises(RuntimeError, match="OpenRouter error"):
        await llm.complete(**KW)
    await llm.aclose()


async def test_retry_after_is_honored_and_capped(tmp_path):
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503, headers={"Retry-After": "600"}),
        httpx.Response(429, headers={"Retry-After": "soon"}),
        httpx.Response(200, json=_ok()),
    ]
    sleeps = []
    llm = _llm_with_sleeps(tmp_path, lambda r: responses.pop(0), sleeps)
    await llm.complete(**KW)
    await llm.aclose()
    assert sleeps == [7.0, 60.0, 0]


async def test_cache_is_written_atomically_and_corrupt_entries_are_misses(tmp_path):
    sent = []
    llm = _llm(tmp_path, lambda r: sent.append(1) or httpx.Response(200, json=_ok()))
    await llm.complete(**KW)
    files = list(tmp_path.iterdir())
    assert [f.name for f in files] == [f"{cache_key(**KW)}.json"]
    files[0].write_text('{"choices": [', encoding="utf-8")
    again = await llm.complete(**KW)
    await llm.aclose()
    assert len(sent) == 2 and again["_meta"]["cached"] is False
    assert json.loads(files[0].read_text(encoding="utf-8"))["choices"]


def test_cache_key_covers_the_sent_payload():
    assert cache_key(**KW) != cache_key(**{**KW, "system": "other"})
    assert cache_key(**KW) != cache_key(**{**KW, "tools": [{"type": "function"}]})
    assert cache_key(**KW) != cache_key(**{**KW, "sample": 1})
    assert cache_key(**KW) == cache_key(**dict(reversed(list(KW.items()))))


def test_tool_call_ids_are_deterministic():
    assert tool_call("wait", reason="a")["id"] == "call_wait_0"
    assert tool_call("wait", reason="a")["id"] == "call_wait_0"
    assert tool_call("wait", index=3, reason="a")["id"] == "call_wait_3"


async def test_scripted_llm_numbers_tool_call_ids_within_a_message():
    calls = [tool_call("wait", reason="a"), tool_call("wait", reason="b"), tool_call("send_message_to_user", message="m")]
    llm = ScriptedLLM(lambda messages, sample: {"content": "", "tool_calls": calls})
    data = await llm.complete(**KW)
    ids = [call["id"] for call in data["choices"][0]["message"]["tool_calls"]]
    assert ids == ["call_wait_0", "call_wait_1", "call_send_message_to_user_2"]


async def test_scripted_llm_keeps_custom_ids():
    raw = {"id": "raw_x", "type": "function", "function": {"name": "wait", "arguments": "{}"}}
    llm = ScriptedLLM(lambda messages, sample: {"content": "", "tool_calls": [raw]})
    data = await llm.complete(**KW)
    assert data["choices"][0]["message"]["tool_calls"][0]["id"] == "raw_x"


async def test_fake_llm_ids_are_numbered_per_message():
    messages = [{"role": "user", "content": '<agent name="A" />'}]
    data = await fake_llm().complete(**{**KW, "messages": messages})
    ids = [call["id"] for call in data["choices"][0]["message"]["tool_calls"]]
    assert ids == ["call_send_message_to_user_0", "call_send_message_to_agent_1"]


async def test_non_json_200_is_retried(tmp_path):
    responses = [httpx.Response(200, text="<html>gateway</html>"), httpx.Response(200, json=_ok())]
    sleeps = []
    llm = _llm_with_sleeps(tmp_path, lambda r: responses.pop(0), sleeps)
    data = await llm.complete(**KW)
    await llm.aclose()
    assert data["choices"][0]["message"]["content"] == "hi"
    assert len(sleeps) == 1


async def test_non_json_200_on_last_attempt_raises(tmp_path):
    llm = _llm_with_sleeps(tmp_path, lambda r: httpx.Response(200, text="nope"), [], max_retries=1)
    with pytest.raises(RetryableResponseError, match="not JSON"):
        await llm.complete(**KW)
    await llm.aclose()


async def test_cache_write_failure_still_returns_paid_data(tmp_path, monkeypatch, caplog):
    import evals.llm as llm_module

    def broken(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(llm_module, "_write_cache", broken)
    llm = _llm(tmp_path, lambda r: httpx.Response(200, json=_ok()))
    with caplog.at_level("WARNING"):
        data = await llm.complete(**KW)
    await llm.aclose()
    assert data["choices"][0]["message"]["content"] == "hi"
    assert llm.budget.spent_usd == pytest.approx(0.01)
    assert "disk full" in caplog.text
