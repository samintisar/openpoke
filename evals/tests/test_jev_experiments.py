import json
from datetime import datetime, timezone

import pytest

import evals.jev_experiments as jx
from evals.cases import Case
from evals.grading import CORRECT, DUPLICATE_CREATE, MISSED_CREATE
from evals.jev_llm import ScriptedJev
from server.services.agents.cards import AgentCard, extract_entities

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


def _create(name, purpose):
    return {"name": "send_message_to_agent", "arguments": {"instructions": "x", "name": name, "purpose": purpose,
                                                           "force_new": True}}


def _turn(case_id, kind, message, gold, outcome, calls, size="lived-200", message_type="user"):
    case = Case(id=case_id, kind=kind, world_seed=1, message=message, gold_targets=gold, message_type=message_type,
                context=[("user_message", "earlier"), ("poke_reply", "ok")])
    row = {"case_id": case_id, "n_agents": size, "sample": 0, "kind": kind, "outcome": outcome,
           "clarified": False, "tool_calls": calls}
    return jx.build_turn(row, case, CARDS, CARDS, NOW)


def _turns():
    game = _turn("game", "same_person_new_topic", "ask Kwame what he thinks we should do for game night",
                 ["NEW"], MISSED_CREATE, [_reuse("kwame-book")])
    dup = _turn("dup", "paraphrase", "ask my contractor Carl when the bathroom remodel can start",
                ["carl-remodel"], DUPLICATE_CREATE,
                [_create("Contractor Start Date", "Ask the contractor when the bathroom remodel starts.")])
    ruth = _turn("ruth", "follow_up", "ask Ruth how I can help with Thanksgiving", ["ruth"], CORRECT,
                 [_reuse("ruth")], size="lived-1000")
    agent = _turn("agent", "agent_result", "Phoenix Trip Planning: car booked", ["phoenix-trip"], CORRECT,
                  [_reuse("phoenix-trip")], message_type="agent")
    return [game, dup, ruth, agent]


def test_replay_builds_the_review_and_duplicate_questions():
    game, dup, ruth, _ = _turns()
    [call] = game.calls
    assert call.proposal.kind == "reuse" and call.ia_choice == "kwame-book"
    assert set(call.review_criteria) >= {"kwame-book", "new", "ask"}
    assert "Kwame Book Club Dates" in call.review_criteria["kwame-book"]
    assert "<new_message>\nask Kwame" in call.review_state and "<active_agents" in call.review_state
    assert "<proposed_decision>" not in call.review_state  # j1-j3 do not show the IA's proposal
    [create] = dup.calls
    assert create.ia_choice == "new"
    assert set(create.dup_criteria) - {"none"} <= {c.id for c in CARDS} and len(create.dup_criteria) <= 4
    assert "<proposed_decision>\nCreate a new agent named 'Contractor Start Date'" in create.dup_state
    assert "summary: Carl Jensen" in dup.router_state  # cards render with summaries, as production does
    assert ruth.calls[0].dup_state is None


def test_plan_shapes_per_experiment():
    turns = _turns()
    j1 = jx.plan("j1", turns)
    assert [r.key for r in j1] == [(0, 0)]  # only the call the calibrated trigger looks at
    body = j1[0].body()
    assert body["model"] == "jev-latest" and body["questions"]["route"]["type"] == "choice"
    assert list(body["questions"]["route"]["criteria"])[-2:] == ["new", "ask"]
    j2 = jx.plan("j2", turns)
    assert [r.key for r in j2] == [(0, 0), (1, 0), (2, 0)]  # every user call at 200/1,000; the agent turn is out
    assert j2[0].body() == j1[0].body()  # j1 and j2 share the cache
    j3 = jx.plan("j3", turns)
    assert [r.key for r in j3] == [(0, -1), (1, -1), (2, -1), (3, -1)]
    assert set(j3[0].body()["questions"]["route"]["criteria"]) == {c.id for c in CARDS} | {"new", "ask"}
    j4 = jx.plan("j4", turns)
    assert [r.key for r in j4] == [(1, 0)]
    assert list(j4[0].body()["questions"]["covered_by"]["criteria"])[-1] == "none"
    est = jx.estimate(j2 + j2)
    assert est["requests"] == 6 and est["unique_requests"] == 3 and est["est_cost_usd"] >= 0


def _answers(table):
    return {key: (choice, probs, conf) for key, (choice, probs, conf) in table.items()}


def test_j1_scores_the_verdict_and_its_effect():
    turns = _turns()
    report = jx.score_j1(turns, _answers({(0, 0): ("new", {"new": 0.8}, 0.8)}))
    [row] = report["rows"]
    assert row["verdict_outcome"] == CORRECT and row["effect"] == "hold_new"
    assert (row["before"], row["after"], row["change"]) == (MISSED_CREATE, CORRECT, "fixed")
    assert report["fixed"] == 1 and report["net"] == 1 and report["regrade_mismatch"] == 0
    asked = jx.score_j1(turns, _answers({(0, 0): ("ask", {}, 0.5)}))["rows"][0]
    assert asked["effect"] == "hold_ask" and asked["after"] != CORRECT  # clarifying is wrong on a new task
    kept = jx.score_j1(turns, _answers({(0, 0): ("kwame-book", {}, 0.9)}))["rows"][0]
    assert kept["effect"] == "proceed" and kept["change"] == "same"
    moved = jx.score_j1(turns, _answers({(0, 0): ("ruth", {}, 0.9)}))["rows"][0]
    assert moved["effect"] == "redirect" and moved["verdict_outcome"] != CORRECT


def test_j2_flags_disagreement_or_low_confidence():
    turns = _turns()
    answers = _answers({(0, 0): ("new", {}, 0.9), (1, 0): ("new", {}, 0.55), (2, 0): ("ruth", {}, 0.95)})
    report = jx.score_j2(turns, answers, confidences=[0.0, 0.6, 1.01])
    by_t = {line["confidence"]: line for line in report["table"]}
    assert report["calls"] == 3 and report["agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert by_t[0.0]["flagged"] == 1 and by_t[0.0]["failures_covered"] == 1 and by_t[0.0]["precision"] == 1.0
    assert by_t[0.6]["flagged"] == 2 and by_t[0.6]["recall"] == 1.0
    assert by_t[1.01]["flagged"] == 3 and by_t[1.01]["rows"] == 4
    assert by_t[0.0]["by_outcome"][MISSED_CREATE] == {"failures": 1, "covered": 1}
    assert "trigger_rate" in report["calibrated_trigger"]


def test_j3_grades_jev_alone_against_the_ia():
    turns = _turns()
    answers = _answers({(0, -1): ("new", {}, 0.9), (1, -1): ("new", {}, 0.9), (2, -1): ("ruth", {}, 0.9),
                        (3, -1): ("ask", {}, 0.4)})
    report = jx.score_j3(turns, answers)
    overall = next(l for l in report["table"] if (l["kind"], l["size"]) == ("all", "all"))
    assert overall["rows"] == 4 and overall["jev_pass"] == 0.5 and overall["ia_pass"] == 0.5
    assert report["jev_outcomes"][DUPLICATE_CREATE] == 1


def test_j4_precision_and_recall_on_duplicate_creates():
    turns = _turns()
    report = jx.score_j4(turns, _answers({(1, 0): ("carl-remodel", {"carl-remodel": 0.7, "none": 0.3}, 0.7)}),
                         thresholds=[0.5, 0.8])
    lines = {l["rule"]: l for l in report["table"]}
    assert report["duplicate_create_rows"] == 1
    assert lines["P(covered) >= 0.5"]["recall"] == 1.0 and lines["P(covered) >= 0.8"]["flagged"] == 0
    assert lines["choice != none"]["precision"] == 1.0


async def test_ask_all_with_the_fake_client_keeps_errors_out():
    turns = _turns()
    requests = jx.plan("j2", turns)
    fake = ScriptedJev(lambda s, qid, q: ({"type": "choice", "choice": "new", "probabilities": {"new": 1.0},
                                           "confidence": 1.0} if "game night" in s else {"type": "choice", "choice": "?"}))
    answers, errors = await jx.ask_all(fake, requests)
    assert list(answers) == [(0, 0)] and len(errors) == 2
    assert all(body["model"] == "jev-latest" for body in fake.requests)


def _result(tmp_path, life, split="dev"):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"meta": {"life": life, "split": split}, "routing_rows": []}), encoding="utf-8")
    return path


def test_life_two_is_refused_before_any_world_is_read(tmp_path, monkeypatch):
    def no_worlds(*a, **k):
        raise AssertionError("load_v2 must not be called")

    monkeypatch.setattr("evals.worlds.lived.bench_v2.load_v2", no_worlds)
    for life, split in ((2, "test"), (2, "dev"), (1, "test")):
        with pytest.raises(ValueError, match="life 1 dev"):
            jx.load_turns(_result(tmp_path, life, split))
        with pytest.raises(ValueError):
            jx.main(["j3", "--result", str(_result(tmp_path, life, split)), "--dry-run"])


def test_dry_run_prints_one_body_and_calls_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jx, "load_turns", lambda path, sizes, limit: ({}, _turns()))
    monkeypatch.setattr(jx, "_make_jev", lambda budget: pytest.fail("no client in a dry run"))
    assert jx.main(["j2", "--result", str(_result(tmp_path, 1)), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "'unique_requests': 3" in out and '"model": "jev-latest"' in out and "Authorization" not in out


def test_without_a_key_the_run_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(jx, "load_turns", lambda path, sizes, limit: ({}, _turns()))
    monkeypatch.setattr(jx, "_make_jev", lambda budget: None)
    assert jx.main(["j1", "--result", str(_result(tmp_path, 1))]) == 2
