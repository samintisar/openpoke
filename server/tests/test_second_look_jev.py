"""Jev as the second look's reviewer, split-only (Plan 6c J5). Fake Jev only: no live calls."""

import asyncio
from datetime import datetime, timezone

import pytest

import server.agents.interaction_agent.second_look_gate as gate_module
from server.agents.interaction_agent.second_look_gate import gate
from server.services.agents.cards import AgentCard, extract_entities
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.jev import JevError
from server.services.agents.jev_review import (
    ROUTE_INSTRUCTIONS,
    ROUTE_QID,
    JevVerdict,
    jev_state,
    route_criteria,
    split_only_effect,
)
from server.services.agents.second_look import DEFAULT_SECOND_LOOK, SecondLookConfig

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
GAME_NIGHT = "ask Kwame what he thinks we should do for game night"
JEV = SecondLookConfig(reviewer="jev")


def _card(agent_id, name, purpose):
    return AgentCard(id=agent_id, name=name, purpose=purpose, entities=extract_entities([name, purpose]),
                     created_at=NOW, last_active_at=NOW)


def _dir():
    return InMemoryDirectory([_card("kwame-book", "Kwame Book Club Dates", "Ask Kwame which dates work for book club.")])


class FakeJev:
    model = "jev-latest"

    def __init__(self, choice="new", confidence=0.9, error=None, delay=0.0):
        self.choice, self.confidence, self.error, self.delay = choice, confidence, error, delay
        self.asked = []

    async def ask(self, state, questions):
        self.asked.append((state, questions))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        options = list(questions[ROUTE_QID]["criteria"])
        rest = (1 - self.confidence) / (len(options) - 1)
        probs = {o: (self.confidence if o == self.choice else rest) for o in options}
        return {"model": "jev-v7", "usage": {"input_tokens": 1000},
                "answers": {ROUTE_QID: {"type": "choice", "choice": self.choice, "probabilities": probs,
                                        "confidence": self.confidence}}}


async def _never(system, user):
    raise AssertionError("the LLM reviewer was called")


async def _look(jev, cfg=JEV, arguments=None, message=GAME_NIGHT):
    return await gate(dict(arguments or {"agent_id": "kwame-book", "instructions": "x"}), message=message,
                      context=["earlier"], directory=_dir(), now=NOW, complete=_never, cfg=cfg, jev=jev)


def test_jev_split_only_is_the_default_reviewer():
    assert DEFAULT_SECOND_LOOK.reviewer == "jev" and DEFAULT_SECOND_LOOK.jev_min_confidence == 0.4


async def test_confident_new_on_a_reuse_holds_with_force_new():
    jev = FakeJev("new", 0.42)
    args, held, record = await _look(jev)
    assert held["status"] == "held_by_second_look" and "force_new=true" in held["instructions"]
    assert args["agent_id"] == "kwame-book"
    assert record["trigger"] == "reuse" and record["effect"] == "hold" and record["reviewer"] == "jev"
    assert record["decision"] == "new" and record["agent_id"] is None and record["jev_choice"] == "new"
    assert record["jev_confidence"] == 0.42 and record["jev_model"] == "jev-v7"
    assert record["jev_top3"][0] == ["new", 0.42] and len(record["jev_top3"]) == 3
    assert record["jev_input_tokens"] == 1000 and record["jev_cost_usd"] == pytest.approx(1000 * 0.042 / 1e6)


async def test_the_question_is_the_experiments_question():
    jev = FakeJev("new", 0.9)
    await _look(jev)
    (state, questions), = jev.asked
    cards = _dir().list_cards()
    assert state == jev_state(GAME_NIGHT, ["earlier"], cards, NOW, 1)  # no proposal block
    assert questions == {ROUTE_QID: {"type": "choice", "instructions": ROUTE_INSTRUCTIONS,
                                     "criteria": route_criteria(cards)}}


@pytest.mark.parametrize("jev", [FakeJev("new", 0.39), FakeJev("ask", 0.95), FakeJev("kwame-book", 0.9)])
async def test_anything_else_leaves_the_reuse(jev):
    args, held, record = await _look(jev)
    assert held is None and args == {"agent_id": "kwame-book", "instructions": "x"}
    assert record["effect"] == "proceed" and record["jev_choice"] == jev.choice


async def test_a_card_choice_is_recorded_as_reuse():
    _, _, record = await _look(FakeJev("kwame-book", 0.9))
    assert record["decision"] == "reuse" and record["agent_id"] == "kwame-book"


async def test_the_threshold_is_configurable():
    _, held, _ = await _look(FakeJev("new", 0.39), cfg=SecondLookConfig(reviewer="jev", jev_min_confidence=0.3))
    assert held is not None


@pytest.mark.parametrize("jev", [FakeJev(error=JevError("HTTP 500")), FakeJev(error=RuntimeError("boom")),
                                 FakeJev("not-an-option", 0.9)])
async def test_errors_leave_the_decision(jev):
    args, held, record = await _look(jev)
    assert held is None and args["agent_id"] == "kwame-book"
    assert record["effect"] == "error" and record["reviewer"] == "jev" and record["error"]


async def test_timeout_leaves_the_decision():
    args, held, record = await _look(FakeJev(delay=5), cfg=SecondLookConfig(reviewer="jev", timeout_s=0.01))
    assert held is None and record["effect"] == "error" and "timed out" in record["error"]


async def test_creates_are_never_reviewed_by_jev():
    jev = FakeJev("new", 0.9)
    arguments = {"name": "Kwame Book Club Check", "purpose": "Ask Kwame again about book club", "instructions": "x",
                 "force_new": True}
    args, held, record = await _look(jev, cfg=SecondLookConfig(reviewer="jev", create=True), arguments=arguments,
                                     message="any word from Kwame on the book club dates?")
    assert jev.asked == [] and args == arguments and held is None and record is None


async def test_no_trigger_no_jev_call():
    jev = FakeJev()
    args, held, record = await _look(jev, message="ask Kwame about the book club dates")
    assert jev.asked == [] and record is None and held is None


async def test_production_uses_a_jev_client_from_the_env(monkeypatch):
    made = []

    class Client(FakeJev):
        def __init__(self, **kwargs):
            super().__init__("new", 0.8)
            made.append(kwargs)

    monkeypatch.setattr(gate_module, "JevClient", Client)
    monkeypatch.setattr(gate_module, "api_key_from_env", lambda: "test-key")
    _, held, record = await _look(None)
    assert held is not None and record["effect"] == "hold"
    _, _, _ = await _look(None)
    assert len(made) == 1  # one client per process
    assert made[0]["timeout_s"] == JEV.timeout_s


def test_split_only_effect_ignores_creates():
    verdict = JevVerdict("new", {"new": 0.9}, 0.9, "v")
    assert split_only_effect(verdict, proposal="create", target_id=None, min_confidence=0.4).action == "proceed"


async def test_production_without_a_key_lets_every_call_stand_and_warns_once(monkeypatch):
    warnings = []
    monkeypatch.setattr(gate_module.logger, "warning", lambda message, *a, **k: warnings.append(message))
    for _ in range(3):
        args, held, record = await gate({"agent_id": "kwame-book", "instructions": "x"}, message=GAME_NIGHT,
                                        context=[], directory=_dir(), now=NOW, complete=_never)
        assert held is None and args["agent_id"] == "kwame-book"
        assert record["effect"] == "error" and record["jev_unavailable"] and record["reviewer"] == "jev"
    assert len(warnings) == 1 and "TYPESAFE_API_KEY" in warnings[0]


async def test_runtime_logs_a_missing_key_at_debug(monkeypatch):
    import server.agents.interaction_agent.runtime as runtime_module
    from server.agents.interaction_agent.runtime import InteractionAgentRuntime, _LoopSummary, _ToolCall
    from server.agents.interaction_agent.turn import TurnContext, current_turn

    seen = []
    monkeypatch.setattr(runtime_module, "get_agent_directory", _dir)
    monkeypatch.setattr(runtime_module.logger, "warning", lambda message, *a, **k: seen.append(("warning", message)))
    monkeypatch.setattr(runtime_module.logger, "debug", lambda message, *a, **k: seen.append(("debug", message)))
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    token = current_turn.set(TurnContext(origin="user", latest_text=GAME_NIGHT))
    try:
        for _ in range(2):  # two turns
            call = _ToolCall(identifier="0", name="send_message_to_agent",
                             arguments={"agent_id": "kwame-book", "instructions": "x"})
            call, held = await runtime._second_look(call, _LoopSummary())
            assert held is None and call.arguments["agent_id"] == "kwame-book"
    finally:
        current_turn.reset(token)
    assert [level for level, _ in seen] == ["warning", "debug", "debug"]  # the gate's one warning, then quiet
