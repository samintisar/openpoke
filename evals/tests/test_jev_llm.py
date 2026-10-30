import pytest

from evals.jev_llm import CACHE_PREFIX, CachedJev, ScriptedJev, body_key, estimate_tokens
from evals.llm import Budget, BudgetExceeded
from server.services.agents.jev import JevError, build_body

Q = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "A", "b": "B"}}}


def _pick_b(state, qid, question):
    return {"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}, "confidence": 0.9}


async def test_second_identical_request_is_served_from_the_cache(tmp_path):
    fake = ScriptedJev(_pick_b, version="jev-1.13.0")
    jev = CachedJev(fake, cache_dir=tmp_path, budget=Budget(1.0))
    first = await jev.ask("state", Q)
    second = await jev.ask("state", Q)
    assert len(fake.requests) == 1
    assert first["_meta"]["cached"] is False and second["_meta"]["cached"] is True
    assert [p.name for p in tmp_path.iterdir()] == [f"{CACHE_PREFIX}{body_key(build_body('jev-latest', 'state', Q))}.json"]
    meter = jev.meter()
    assert meter["calls"] == 1 and meter["cached_calls"] == 1
    assert meter["models"] == {"jev-1.13.0": 2}
    expected = estimate_tokens(fake.requests[0]) * 0.042 / 1e6
    assert jev.cost_usd == pytest.approx(expected) and jev.budget.spent_usd == pytest.approx(expected)
    assert await jev.choice("state", "q", "which?", {"a": "A", "b": "B"}) == ("b", {"a": 0.1, "b": 0.9}, 0.9)
    assert len(fake.requests) == 1  # same body as Q: cached


async def test_a_different_body_is_a_different_entry(tmp_path):
    fake = ScriptedJev(_pick_b)
    jev = CachedJev(fake, cache_dir=tmp_path, budget=Budget(1.0))
    await jev.ask("state", Q)
    await jev.ask("other state", Q)
    assert len(fake.requests) == 2


async def test_budget_is_checked_before_uncached_calls_only(tmp_path):
    fake = ScriptedJev(_pick_b)
    jev = CachedJev(fake, cache_dir=tmp_path, budget=Budget(0.0))
    with pytest.raises(BudgetExceeded):
        await jev.ask("state", Q)
    assert fake.requests == []
    warm = CachedJev(fake, cache_dir=tmp_path, budget=Budget(1.0))
    await warm.ask("state", Q)
    cold = CachedJev(fake, cache_dir=tmp_path, budget=Budget(0.0))
    assert (await cold.ask("state", Q))["_meta"]["cached"] is True  # a cached answer needs no budget


async def test_noul_and_bad_answers(tmp_path):
    fake = ScriptedJev(lambda s, qid, q: {"type": "noul", "noul": 0.25})
    jev = CachedJev(fake, cache_dir=tmp_path, budget=Budget(1.0))
    assert await jev.noul("s", "same matter?") == 0.25
    bad = CachedJev(ScriptedJev(lambda s, qid, q: {"type": "choice", "choice": "zzz"}), cache_dir=tmp_path,
                    budget=Budget(1.0))
    with pytest.raises(JevError):
        await bad.choice("s", "q", "which?", {"a": "A"})
