import hashlib
import json

import pytest

from evals.llm import ScriptedLLM
from evals.systems.cards_from_world import card_for
from evals.systems.full_cards import FullCardsSystem
from evals.systems.retrieved import RetrievedSystem
from evals.worlds.lived import summaries
from evals.worlds.model import AgentSpec, World
from server.services.agents.retrieval import _entity_terms, content_tokens


def _agent(agent_id="trent", history=None):
    history = history or [["Ask Trent about permits.", ""], ["How is the bathroom remodel going with Trent?", ""]]
    return {"id": agent_id, "name": "Trent - Permit Stuff", "purpose": history[0][0], "entities": [],
            "last_active_days": 1.0, "created_days": 9.0, "history": history}


def _world_file(tmp_path, seed=1, sizes=None):
    sizes = sizes or [{"label": "lived-10", "size": 10, "day": 9.0, "agents": [_agent()], "thread_agents": {},
                       "cases": []}]
    path = tmp_path / f"world-s{seed}.json"
    path.write_text(json.dumps({"format": 1, "seed": seed, "meta": {}, "sizes": sizes}), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recording_llm(seen):
    def script(messages, sample):
        seen.append(messages[-1]["content"])
        return {"content": json.dumps({"summary": f"after {len(seen)}"})}
    return ScriptedLLM(script)


async def test_build_folds_every_instruction_in_order(tmp_path):
    _world_file(tmp_path)
    seen = []
    out = await summaries.build(1, llm=_recording_llm(seen), model="m", directory=tmp_path)
    assert out["sizes"] == {"lived-10": {"trent": "after 2"}} and out["empty"] == 0
    assert "<current_summary>\nafter 1\n</current_summary>" in seen[1] and "bathroom remodel" in seen[1]
    assert len(out["world_sha256"]) == 64 and out["source"] == "history" and out["logs_sha256"] is None
    assert out["results_folded"] == {"lived-10": 0} and out["agents_with_results"] == {"lived-10": 0}


def test_runs_pair_each_instruction_with_the_results_before_the_next():
    entries = [[0, "result", "early report"], [1, "instruction", "plan"], [2, "result", "r1"], [3, "result", "r2"],
               [4, "instruction", "book"], [5, "instruction", "pay"], [6, "result", "paid"]]
    assert summaries.runs_from_log(entries) == [("", "early report"), ("plan", "r1\n\nr2"), ("book", ""),
                                                ("pay", "paid")]


async def test_build_folds_the_logged_results_up_to_each_sizes_cutoff(tmp_path):
    other = {**_agent("bg", [["Water the plants.", ""]]), "name": "Plants"}
    sizes = [{"label": "lived-10", "size": 10, "day": 9.0, "agents": [_agent()], "thread_agents": {}, "cases": []},
             {"label": "lived-1000", "size": 1000, "day": 9.0, "agents": [_agent(), other], "thread_agents": {},
              "cases": []}]
    sha = _world_file(tmp_path, sizes=sizes)
    logs = {"format": 1, "seed": 1, "world_sha256": sha, "cutoffs": {"lived-10": 5, "lived-1000": 5},
            "logs": {"trent": [[1, "instruction", "Ask Trent about permits."], [2, "result", "Trent Hollis has them."],
                               [5, "instruction", "How is the bathroom remodel going with Trent?"],
                               [7, "result", "Too late for this size."]]}}
    (tmp_path / "agent-logs-s1.json").write_text(json.dumps(logs), encoding="utf-8")
    seen = []
    out = await summaries.build(1, llm=_recording_llm(seen), model="m", directory=tmp_path, concurrency=1)
    assert "Trent Hollis has them." in seen[0] and not any("Too late" in prompt for prompt in seen)
    assert any("Water the plants." in prompt for prompt in seen)  # background agents fold their history
    assert out["source"] == "logs" and len(out["logs_sha256"]) == 64
    assert out["results_folded"] == {"lived-10": 1, "lived-1000": 1}
    assert out["agents_with_results"] == {"lived-10": 1, "lived-1000": 1}


async def test_build_refuses_logs_from_a_different_world(tmp_path):
    _world_file(tmp_path)
    (tmp_path / "agent-logs-s1.json").write_text(json.dumps({"world_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="different world"):
        await summaries.build(1, llm=_recording_llm([]), model="m", directory=tmp_path)


async def test_a_failed_update_keeps_the_previous_summary(tmp_path):
    _world_file(tmp_path)
    replies = iter([json.dumps({"summary": "first"}), "not json"])
    out = await summaries.build(1, llm=ScriptedLLM(lambda m, s: {"content": next(replies)}), model="m",
                                directory=tmp_path)
    assert out["sizes"]["lived-10"]["trent"] == "first"


def test_cards_use_the_summary_only_when_switched_on():
    agent = AgentSpec(id="t", name="Trent - Permit Stuff", purpose="p", entities=[], last_active_days=1.0,
                      created_days=2.0, summary="Trent Hollis: bathroom remodel.")
    world = World(seed=1, agents=[agent])
    card = card_for(agent, world)
    # As in production, the summary is not copied into the card's entities; retrieval reads its people itself.
    assert card.summary == "Trent Hollis: bathroom remodel." and "hollis" not in " ".join(card.entities).lower()
    assert set(content_tokens("Hollis")) <= _entity_terms(card)
    assert card_for(agent, world, summaries=False).summary == ""
    assert RetrievedSystem(summaries=False).cards(world)[0].summary == ""
    assert RetrievedSystem().cards(world)[0].summary == "Trent Hollis: bathroom remodel."
    assert FullCardsSystem().cards(world)[0].summary == ""
