import json

import httpx
import pytest

from server.services.agents.jev import (
    JEV_URL,
    JevClient,
    JevError,
    build_body,
    cost_usd,
    parse_choice,
    parse_noul,
)

KEY = "test-key-not-real"
CRITERIA = {"a": "agent A", "b": "agent B", "new": "a new matter"}


def _ok_choice(qid="route", pick="a"):
    return {"model": "jev-1.13.0",
            "answers": {qid: {"type": "choice", "choice": pick,
                              "probabilities": {"a": 0.7, "b": 0.2, "new": 0.1}, "confidence": 0.7}},
            "usage": {"input_tokens": 1000, "output_tokens": 3}}


def _client(handler, **kw):
    return JevClient(api_key=KEY, transport=httpx.MockTransport(handler), **kw)


async def test_choice_sends_the_documented_body_and_parses_the_answer():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_ok_choice())

    client = _client(handler)
    pick, probs, conf = await client.choice("state text", "route", "Which agent?", CRITERIA)
    await client.aclose()
    assert (pick, conf) == ("a", 0.7) and probs["b"] == 0.2
    [request] = seen
    assert str(request.url) == JEV_URL
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {
        "model": "jev-latest", "state": "state text",
        "questions": {"route": {"type": "choice", "instructions": "Which agent?", "criteria": CRITERIA}}}


async def test_noul_parses_a_probability():
    def handler(request):
        body = json.loads(request.content)
        assert body["questions"]["q"] == {"type": "noul", "instructions": "same matter?"}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.83}},
                                         "usage": {"input_tokens": 10}})

    client = _client(handler)
    assert await client.noul({"message": "hi"}, "same matter?") == 0.83
    await client.aclose()


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.Response(401, text="bad key"),
    httpx.Response(200, text="<html>"),
    httpx.Response(200, json={"model": "x"}),
    httpx.Response(200, json={"model": "x", "answers": {}}),
    httpx.Response(200, json={"answers": {"route": {"type": "choice", "choice": "zzz", "probabilities": {}}}}),
    httpx.Response(200, json={"answers": {"route": {"type": "choice", "choice": "a", "confidence": 7}}}),
    httpx.Response(200, json={"answers": {"route": {"type": "noul", "noul": 0.5}}}),
    httpx.Response(200, json={"answers": {"route": "a"}}),
])
async def test_every_bad_response_is_a_jev_error(response):
    client = _client(lambda request: response)
    with pytest.raises(JevError):
        await client.choice("s", "route", "q", CRITERIA)
    await client.aclose()


async def test_http_error_carries_the_body_but_never_the_key():
    client = _client(lambda request: httpx.Response(429, text="slow down"))
    with pytest.raises(JevError) as info:
        await client.ask("s", {"q": {"type": "noul", "instructions": "x"}})
    assert "429" in str(info.value) and "slow down" in str(info.value) and KEY not in str(info.value)
    assert KEY not in repr(client)
    await client.aclose()


async def test_transport_errors_and_timeouts_are_jev_errors():
    def refuse(request):
        raise httpx.ConnectError("no route")

    def slow(request):
        raise httpx.ReadTimeout("slow")

    for handler in (refuse, slow):
        client = _client(handler)
        with pytest.raises(JevError):
            await client.noul("s", "x")
        await client.aclose()


async def test_no_key_and_bad_questions_are_jev_errors(monkeypatch):
    client = JevClient(api_key="", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_ok_choice())))
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        await client.noul("s", "x")
    await client.aclose()
    client = _client(lambda r: httpx.Response(200, json=_ok_choice()))
    with pytest.raises(JevError):
        await client.ask("s", {})
    with pytest.raises(JevError):
        await client.ask("s", {"q": {"type": "maybe"}})
    with pytest.raises(JevError):
        await client.ask("s", {"q": {"type": "choice", "instructions": "x", "criteria": {}}})
    with pytest.raises(JevError):
        await client.ask("s", {"q": {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(256)}}})
    await client.aclose()


async def test_unexpected_exceptions_are_wrapped():
    def explode(request):
        raise RuntimeError("anything")

    client = _client(explode)
    with pytest.raises(JevError):
        await client.noul("s", "x")
    await client.aclose()


def test_parsers_and_cost():
    assert parse_choice(_ok_choice(), "route", set(CRITERIA))[0] == "a"
    with pytest.raises(JevError):
        parse_choice(_ok_choice(pick="b"), "route", {"a"})
    with pytest.raises(JevError):
        parse_noul({"answers": {"q": {"type": "noul", "noul": "high"}}}, "q")
    assert cost_usd(_ok_choice()) == pytest.approx(1000 * 0.042 / 1e6)
    assert cost_usd({}) == 0.0
    assert build_body("m", {"k": 1}, {"q": {"type": "noul", "instructions": "x"}})["state"] == {"k": 1}
