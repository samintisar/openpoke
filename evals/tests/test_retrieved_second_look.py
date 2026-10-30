import json

from evals.cases import Case
from evals.grading import Decision
from evals.layers.routing import run_routing_case
from evals.llm import ScriptedLLM, tool_call
from evals.systems.full_cards import FullCardsSystem
from evals.systems.retrieved import RetrievedSystem
from evals.worlds.model import AgentSpec, World

GAME_NIGHT = "ask Kwame what he thinks we should do for game night"
SEND = {"agent_id": "kwame-book", "instructions": "x"}


def _world():
    agent = AgentSpec(id="kwame-book", name="Kwame Book Club Dates", purpose="Ask Kwame which dates work for book club.",
                      entities=[], last_active_days=1.0, created_days=5.0)
    return World(seed=1, agents=[agent])


def _case(message, message_type="user"):
    return Case(id="c", kind="same_person_new_topic", world_seed=1, message=message, gold_targets=["NEW"], context=[],
                message_type=message_type)


def _verdict_llm(payload, meta=None):
    llm = ScriptedLLM(lambda m, s: {"content": json.dumps(payload)})
    if meta is None:
        return llm
    inner = llm.complete

    async def complete(**kwargs):
        response = await inner(**kwargs)
        response["_meta"] = {**response["_meta"], **meta}
        return response

    llm.complete = complete
    return llm


async def test_second_look_holds_and_is_accounted():
    world = _world()
    turn = RetrievedSystem().new_turn(world, _case(GAME_NIGHT))
    turn.bind_llm(_verdict_llm({"decision": "new", "reason": "r"}), model="m", sample=0)
    decision = Decision()
    result = await turn.handle("send_message_to_agent", dict(SEND), decision)
    assert result["status"] == "error" and result["error"]["status"] == "held_by_second_look"
    assert decision.dispatched_ids == [] and decision.second_looks[0]["effect"] == "hold"
    again = await turn.handle("send_message_to_agent", dict(SEND), decision)
    assert again["status"] == "success" and len(decision.second_looks) == 1  # one look per turn
    assert decision.dispatched_ids == ["kwame-book"]


async def test_second_look_cost_lands_on_the_decision():
    turn = RetrievedSystem().new_turn(_world(), _case(GAME_NIGHT))
    meta = {"cost": 0.25, "prompt_tokens": 100, "completion_tokens": 7, "cached_tokens": 40, "latency_s": 0.5,
            "cache_key": "k1"}
    turn.bind_llm(_verdict_llm({"decision": "new", "reason": "r"}, meta), model="m", sample=0)
    decision = Decision()
    await turn.handle("send_message_to_agent", dict(SEND), decision)
    assert decision.cost == 0.25 and decision.fresh_cost == 0.25 and decision.cached_calls == 0
    assert (decision.prompt_tokens, decision.completion_tokens, decision.cached_prompt_tokens) == (100, 7, 40)
    assert decision.cache_keys == ["k1"] and decision.latency_s == 0.5


async def test_a_broken_verdict_lets_the_dispatch_stand():
    turn = RetrievedSystem().new_turn(_world(), _case(GAME_NIGHT))
    turn.bind_llm(ScriptedLLM(lambda m, s: {"content": "not json"}), model="m", sample=0)
    decision = Decision()
    result = await turn.handle("send_message_to_agent", dict(SEND), decision)
    assert result["status"] == "success" and decision.dispatched_ids == ["kwame-book"]
    assert decision.second_looks[0]["effect"] == "error"


async def test_switch_off_never_looks():
    turn = RetrievedSystem(second_look=False).new_turn(_world(), _case("ask Kwame about game night"))
    turn.bind_llm(_verdict_llm({"decision": "new", "reason": "r"}), model="m", sample=0)
    decision = Decision()
    await turn.handle("send_message_to_agent", dict(SEND), decision)
    assert decision.dispatched_ids == ["kwame-book"] and decision.second_looks == []


async def test_agent_messages_and_unbound_turns_never_look():
    decision = Decision()
    turn = RetrievedSystem().new_turn(_world(), _case(GAME_NIGHT, message_type="agent"))
    turn.bind_llm(_verdict_llm({"decision": "new", "reason": "r"}), model="m", sample=0)
    await turn.handle("send_message_to_agent", dict(SEND), decision)
    unbound = RetrievedSystem().new_turn(_world(), _case(GAME_NIGHT))
    await unbound.handle("send_message_to_agent", dict(SEND), decision)
    assert decision.dispatched_ids == ["kwame-book"] and decision.second_looks == []


def test_full_cards_stays_frozen_without_the_second_look():
    assert FullCardsSystem().second_look is False and RetrievedSystem().second_look is True


async def test_routing_layer_binds_the_run_client():
    def script(messages, sample):
        first = messages[0]["content"]
        if first.startswith("<recent_conversation>"):  # the second-look call
            return {"content": json.dumps({"decision": "new", "reason": "game night is new"})}
        if messages[-1]["role"] == "tool":
            return {"content": "Done."}
        return {"content": "", "tool_calls": [tool_call("send_message_to_agent", **SEND)]}

    decision = await run_routing_case(_case(GAME_NIGHT), _world(), RetrievedSystem(), ScriptedLLM(script),
                                      model="m", sample=0)
    assert decision.second_looks[0]["effect"] == "hold" and decision.dispatched_ids == []
