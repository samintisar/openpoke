from datetime import datetime, timezone

from evals.second_look_calibration import (
    Proposal,
    choose,
    first_trigger,
    proposals_for,
    sweep,
    trigger_reason,
)
from server.services.agents.cards import AgentCard, extract_entities
from server.services.agents.second_look import SecondLookConfig, create_trigger, reuse_trigger

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def _card(agent_id, name, purpose, summary=""):
    return AgentCard(id=agent_id, name=name, purpose=purpose, summary=summary,
                     entities=extract_entities([name, purpose, summary]), created_at=NOW, last_active_at=NOW)


KWAME_BOOK = _card("kwame-book", "Kwame Book Club Dates", "Ask Kwame which dates work for book club.")
CARL = _card("carl-remodel", "Carl Remodel Urgency Check", "Ask Carl how urgent the remodel is.",
             "Carl Jensen (contractor): bathroom remodel; waiting on the start date.")
PHOENIX = _card("phoenix-trip", "Phoenix Trip Planning", "Plan the Phoenix trip, including a rental car.")
RUTH = _card("ruth", "Ruth Thanksgiving", "Ask Ruth how to help with Thanksgiving.")
CARDS = [KWAME_BOOK, CARL, PHOENIX, RUTH]


def _reuse(agent_id):
    return {"name": "send_message_to_agent", "arguments": {"instructions": "x", "agent_id": agent_id}}


def _create(name, purpose, **extra):
    return {"name": "send_message_to_agent",
            "arguments": {"instructions": "x", "name": name, "purpose": purpose, **extra}}


# --- trigger_reason / first_trigger: pure thresholds over config-independent features ---------------------------

def test_trigger_reason_applies_each_threshold():
    cfg = SecondLookConfig(min_score=0.5, reuse_coverage=0.5, name_coverage=0.8)
    assert trigger_reason(Proposal("create", best_score=0.6, evidence=True), cfg) == "create"
    assert trigger_reason(Proposal("create", best_score=0.4, evidence=True), cfg) is None
    assert trigger_reason(Proposal("create", best_score=0.9, evidence=False), cfg) is None
    assert trigger_reason(Proposal("create", best_score=0.9, evidence=True, guarded=True), cfg) is None
    assert trigger_reason(Proposal("reuse", shared_person=True, coverage=1.0), cfg) == "shared_person"
    assert trigger_reason(Proposal("reuse", named_coverage=0.7, coverage=0.9), cfg) == "unmentioned_name"
    assert trigger_reason(Proposal("reuse", named_coverage=0.85, coverage=0.4), cfg) == "low_coverage"
    assert trigger_reason(Proposal("reuse", named_coverage=0.85, coverage=0.6), cfg) is None
    assert trigger_reason(Proposal("reuse", coverage=None), cfg) is None
    assert trigger_reason(Proposal("none"), cfg) is None


def test_first_trigger_is_the_one_look_per_turn():
    cfg = SecondLookConfig()
    calls = [Proposal("reuse", coverage=0.9), Proposal("reuse", coverage=0.1), Proposal("reuse", shared_person=True)]
    assert first_trigger(calls, cfg) == "low_coverage"
    assert first_trigger([], cfg) is None


# --- proposals_for: replays a row's dispatches with the gate's inputs ----------------------------------------------

def test_proposals_match_the_production_triggers():
    msg = "ask Kwame what he thinks we should do for game night"
    row = {"tool_calls": [_reuse("kwame-book")]}
    [prop] = proposals_for(row, msg, CARDS, NOW)
    for cfg in (SecondLookConfig(), SecondLookConfig(reuse_coverage=0.1), SecondLookConfig(name_coverage=0.0)):
        assert (trigger_reason(prop, cfg) is not None) == bool(reuse_trigger(msg, KWAME_BOOK, CARDS, NOW, cfg))
    assert trigger_reason(prop, SecondLookConfig()) == "low_coverage"

    sedona = "book a rental car in Sedona"
    [prop] = proposals_for({"tool_calls": [_reuse("phoenix-trip")]}, sedona, CARDS, NOW)
    assert trigger_reason(prop, SecondLookConfig()) == "unmentioned_name"
    assert trigger_reason(prop, SecondLookConfig(name_coverage=0.0, reuse_coverage=0.0)) is None

    remodel = "any update from my contractor on the bathroom remodel?"
    [prop] = proposals_for({"tool_calls": [_create("Contractor Update", "Get an update from the contractor.")]},
                           remodel, CARDS, NOW)
    for min_score in (0.3, 0.5, 0.9, 1.1):
        cfg = SecondLookConfig(min_score=min_score)
        assert (trigger_reason(prop, cfg) == "create") == bool(create_trigger(remodel, CARDS, NOW, cfg))


def test_proposals_skip_errored_calls_and_unknown_ids_and_see_in_turn_creates():
    row = {"tool_calls": [
        {"name": "send_message_to_agent", "arguments": {"agent_id": "kwame-book"}, "error": "missing instructions"},
        {"name": "send_message_to_user", "arguments": {"message": "ok"}},
        _reuse("no-such-agent"),
        _create("Kwame Game Night", "Plan game night with Kwame.", force_new=True),
        _reuse("kwame-game-night"),
    ]}
    props = proposals_for(row, "ask Kwame about game night", CARDS, NOW)
    assert [p.kind for p in props] == ["none", "create", "reuse"]
    assert props[1].guarded is False  # force_new skips the duplicate guard
    assert props[2].target == "kwame-game-night"  # the agent created earlier in the turn is in the directory


def test_a_create_the_duplicate_guard_catches_is_guarded():
    row = {"tool_calls": [_create("Kwame Book Club Dates", "Ask Kwame which dates work for book club.")]}
    [prop] = proposals_for(row, "ask Kwame about book club dates", CARDS, NOW)
    assert prop.guarded and trigger_reason(prop, SecondLookConfig(min_score=0.0)) is None
    [prop] = proposals_for(row, "ask Kwame about book club dates", CARDS, NOW, guard=False)
    assert not prop.guarded


# --- sweep / choose ------------------------------------------------------------------------------------------------

def _item(outcome, size, *props, user=True):
    return {"outcome": outcome, "n_agents": size, "user": user, "proposals": list(props)}


ITEMS = [
    _item("missed_create", "lived-200", Proposal("reuse", coverage=0.2)),
    _item("correct", "lived-200", Proposal("reuse", coverage=0.45)),
    _item("correct", "lived-200", Proposal("reuse", coverage=1.0)),
    _item("duplicate_create", "lived-1000", Proposal("create", best_score=0.7, evidence=True)),
    _item("correct", "lived-1000", Proposal("reuse", coverage=0.2), user=False),  # agent message: never looks
]


def test_sweep_counts_rates_reasons_and_failures():
    [row] = sweep(ITEMS, [SecondLookConfig(min_score=0.6, reuse_coverage=0.5)])
    assert row["min_score"] == 0.6 and row["reuse_coverage"] == 0.5 and row["name_coverage"] == 0.8
    assert row["rows"] == 5 and row["triggered"] == 3 and row["trigger_rate"] == 0.6
    assert row["by_size"]["lived-200"]["trigger_rate"] == round(2 / 3, 4)
    assert row["by_size"]["lived-1000"]["trigger_rate"] == 0.5
    assert row["by_reason"] == {"create": 1, "low_coverage": 2}
    assert row["failures"] == 2 and row["failures_covered"] == 2
    assert row["precision"] == round(2 / 3, 4) and row["recall"] == 1.0
    assert row["by_outcome"]["missed_create"] == {"failures": 1, "covered": 1}
    assert row["by_reason_failures"] == {"create": 1, "low_coverage": 1}


def test_sweep_restricts_to_sizes():
    [row] = sweep(ITEMS, [SecondLookConfig(min_score=0.8, reuse_coverage=0.3)], sizes=("lived-200",))
    assert row["rows"] == 3 and row["triggered"] == 1 and row["failures_covered"] == 1


def test_choose_takes_the_most_recall_under_the_cap_at_every_size():
    table = [
        {"min_score": 0.5, "reuse_coverage": 0.5, "name_coverage": 0.8, "failures_covered": 9, "trigger_rate": 0.2,
         "by_size": {"a": {"trigger_rate": 0.2}}},
        {"min_score": 0.6, "reuse_coverage": 0.4, "name_coverage": 0.8, "failures_covered": 7, "trigger_rate": 0.1,
         "by_size": {"a": {"trigger_rate": 0.16}}},
        {"min_score": 0.7, "reuse_coverage": 0.3, "name_coverage": 0.8, "failures_covered": 5, "trigger_rate": 0.1,
         "by_size": {"a": {"trigger_rate": 0.12}}},
        {"min_score": 0.7, "reuse_coverage": 0.3, "name_coverage": 0.6, "failures_covered": 5, "trigger_rate": 0.08,
         "by_size": {"a": {"trigger_rate": 0.09}}},
    ]
    assert choose(table, cap=0.15) is table[3]  # ties on recall go to the lower rate
    assert choose(table, cap=0.01) is table[3]  # nothing under the cap: the lowest rate
