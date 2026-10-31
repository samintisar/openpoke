"""The eval's Jev reviewer switch (Plan 6c J5). Scripted Jev only: no live calls."""

import json

import pytest

from evals.cases import Case
from evals.grading import Decision
from evals.jev_llm import ScriptedJev
from evals.layers.routing import run_routing_case
from evals.llm import Budget, RunAborted, ScriptedLLM, tool_call
from evals.systems import get_system
from evals.systems.full_cards import FullCardsSystem
from evals.systems.retrieved import RetrievedSystem
from evals.worlds.model import AgentSpec, World
from server.services.agents.jev import JEV_USD_PER_M_INPUT

GAME_NIGHT = "ask Kwame what he thinks we should do for game night"
SEND = {"agent_id": "kwame-book", "instructions": "x"}


def _world():
    agent = AgentSpec(id="kwame-book", name="Kwame Book Club Dates", purpose="Ask Kwame which dates work for book club.",
                      entities=[], last_active_days=1.0, created_days=5.0)
    return World(seed=1, agents=[agent])


def _case(message=GAME_NIGHT):
    return Case(id="c", kind="same_person_new_topic", world_seed=1, message=message, gold_targets=["NEW"], context=[],
                message_type="user")


def _jev(choice="new", confidence=0.9):
    def script(state, qid, question):
        options = list(question["criteria"])
        rest = (1 - confidence) / (len(options) - 1)
        return {"type": "choice", "choice": choice, "confidence": confidence,
                "probabilities": {o: confidence if o == choice else rest for o in options}}
    return ScriptedJev(script, version="jev-v7")


def _llm(tmp_path, budget=1.0):
    """A scripted run client with the cache directory and budget a real run's client carries."""
    def never(messages, sample):
        raise AssertionError("the LLM reviewer was called")
    llm = ScriptedLLM(never)
    llm.cache_dir, llm.budget = tmp_path, Budget(budget)
    return llm


def _system(jev):
    return RetrievedSystem(second_look_reviewer="jev", jev_client=jev)


async def _send(system, tmp_path, llm=None):
    turn = system.new_turn(_world(), _case())
    turn.bind_llm(llm or _llm(tmp_path), model="m", sample=0)
    decision = Decision()
    result = await turn.handle("send_message_to_agent", dict(SEND), decision)
    return result, decision


def test_the_switch_defaults_to_jev_and_validates():
    assert RetrievedSystem().second_look_reviewer == "jev"
    assert RetrievedSystem().new_turn(_world(), _case()).second_look_cfg.reviewer == "jev"
    assert RetrievedSystem(second_look_reviewer="llm").new_turn(_world(), _case()).second_look_cfg.reviewer == "llm"
    with pytest.raises(ValueError):
        RetrievedSystem(second_look_reviewer="oracle")


def test_the_switch_is_forwarded_through_options():
    system = get_system("retrieved", {"second_look_reviewer": "jev"})
    assert system.second_look_reviewer == "jev"
    for name in ("retrieved_nolife", "retrieved_noguard", "retrieved_nosearch"):
        assert get_system(name, {"second_look_reviewer": "jev"}).second_look_reviewer == "jev"
    assert FullCardsSystem().second_look is False  # frozen: no second look of either kind


async def test_confident_new_holds_and_jev_is_accounted(tmp_path):
    jev = _jev("new", 0.9)
    result, decision = await _send(_system(jev), tmp_path)
    assert result["status"] == "error" and "force_new=true" in result["error"]["instructions"]
    assert decision.dispatched_ids == [] and len(jev.requests) == 1
    record, = decision.second_looks
    assert record["reviewer"] == "jev" and record["effect"] == "hold" and record["jev_model"] == "jev-v7"
    tokens = record["jev_input_tokens"]
    cost = tokens * JEV_USD_PER_M_INPUT / 1e6
    assert tokens > 0 and record["jev_cost_usd"] == pytest.approx(cost)
    assert decision.cost == pytest.approx(cost) and decision.fresh_cost == pytest.approx(cost)
    assert decision.prompt_tokens == tokens and decision.cached_calls == 0
    assert decision.cache_keys and decision.cache_keys[0].startswith("jev-")


async def test_a_cached_answer_is_free_but_attributed(tmp_path):
    llm = _llm(tmp_path)
    system = _system(_jev("new", 0.9))
    await _send(system, tmp_path, llm)
    _, decision = await _send(system, tmp_path, llm)  # same body: the cache answers
    assert decision.cached_calls == 1 and decision.fresh_cost == 0 and decision.cost > 0
    assert decision.second_looks[0]["jev_cached"] is True
    assert system._jev_for(llm).meter()["cached_calls"] == 1  # one cached client per run client


@pytest.mark.parametrize("choice,confidence", [("new", 0.3), ("ask", 0.9), ("kwame-book", 0.9)])
async def test_other_answers_leave_the_dispatch(tmp_path, choice, confidence):
    result, decision = await _send(_system(_jev(choice, confidence)), tmp_path)
    assert result["status"] == "success" and decision.dispatched_ids == ["kwame-book"]
    assert decision.second_looks[0]["effect"] == "proceed"


async def test_a_spent_budget_skips_jev_and_lets_the_dispatch_stand(tmp_path):
    """Like the LLM reviewer: the failed look is an error record; the IA's next call meets the same cap."""
    jev = _jev()
    result, decision = await _send(_system(jev), tmp_path, _llm(tmp_path, budget=0.0))
    assert jev.requests == [] and result["status"] == "success"
    assert decision.second_looks[0]["effect"] == "error" and decision.fresh_cost == 0


async def test_no_key_aborts_the_run(tmp_path, monkeypatch):
    import evals.systems.retrieved as retrieved
    monkeypatch.setattr(retrieved, "api_key_from_env", lambda: None)
    with pytest.raises(RunAborted):
        await _send(RetrievedSystem(second_look_reviewer="jev"), tmp_path)


async def test_routing_layer_runs_the_jev_reviewer(tmp_path):
    def script(messages, sample):
        if messages[-1]["role"] == "tool":
            return {"content": "Done."}
        return {"content": "", "tool_calls": [tool_call("send_message_to_agent", **SEND)]}

    llm = ScriptedLLM(script)
    llm.cache_dir, llm.budget = tmp_path, Budget(1.0)
    decision = await run_routing_case(_case(), _world(), _system(_jev()), llm, model="m", sample=0)
    assert decision.second_looks[0]["effect"] == "hold" and decision.second_looks[0]["reviewer"] == "jev"
    held = [c for c in decision.tool_calls if c.get("name") == "send_message_to_agent"]
    assert held and decision.dispatched_ids == []


async def test_offline_run_clients_get_an_offline_jev_that_never_holds(tmp_path, monkeypatch):
    """--fake-llm and test fakes (no cache dir or budget) need no key and never reach the network."""
    import evals.systems.retrieved as retrieved

    def no_key():
        raise AssertionError("an offline run looked for the key")
    monkeypatch.setattr(retrieved, "api_key_from_env", no_key)
    def script(messages, sample):
        if messages[-1]["role"] == "tool":
            return {"content": "Done."}
        return {"content": "", "tool_calls": [tool_call("send_message_to_agent", **SEND)]}

    decision = await run_routing_case(_case(), _world(), RetrievedSystem(), ScriptedLLM(script), model="m", sample=0)
    record, = decision.second_looks
    assert record["reviewer"] == "jev" and record["jev_model"] == "jev-offline" and record["effect"] == "proceed"
    assert decision.dispatched_ids == ["kwame-book"] and decision.cost == 0 and decision.prompt_tokens == 0
    assert not list(tmp_path.iterdir())


async def test_an_unbound_jev_turn_never_looks():
    """A turn without its system's Jev never falls back to production's client."""
    from evals.systems.retrieved import RetrievedTurn
    world = _world()
    turn = RetrievedTurn(world, RetrievedSystem().cards(world), RetrievedSystem().cfg, case=_case(), second_look=True)
    turn.bind_llm(ScriptedLLM(lambda m, s: {"content": ""}), model="m", sample=0)
    decision = Decision()
    result = await turn.handle("send_message_to_agent", dict(SEND), decision)
    assert result["status"] == "success" and decision.second_looks == []
