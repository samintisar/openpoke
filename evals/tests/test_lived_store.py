import json

import pytest

from evals.cases import NEW, Case
from evals.layers.routing import render_context
from evals.llm import fake_llm
from evals.run import Config, load_config, run_experiment
from evals.worlds.lived.model import Snapshot
from evals.worlds.lived.store import load_lived, save
from evals.worlds.model import AgentSpec

STAMP = "2026-06-01T09:30:00+00:00"


def _snapshot(size=2):
    agents = [
        AgentSpec(id="email-alice", name="Email Alice", purpose="draft to Alice", entities=[], last_active_days=1.0,
                  created_days=5.0, history=[("draft to Alice", "")], threads=["t1"]),
        AgentSpec(id="lisbon-flight", name="Lisbon flight", purpose="find flights", entities=[], last_active_days=2.0,
                  created_days=3.0, history=[("find flights", "")], threads=["t2"]),
    ]
    transcript = [("user_message", "email alice about the lease", STAMP), ("poke_reply", "on it", STAMP)]
    return Snapshot(size=size, day=60.0, agents=agents, transcript=transcript,
                    thread_agents={"t1": ["email-alice"], "t2": ["lisbon-flight"]}, thread_last_day={"t1": 59.0})


def _cases():
    context = [("user_message", "email alice about the lease", STAMP)]
    return [
        Case(id="lived-s1-2-follow_up-00", kind="follow_up", world_seed=1, message="tell alice friday works",
             gold_targets=["email-alice"], context=context),
        Case(id="lived-s1-2-new_task-00", kind="new_task", world_seed=1, message="find a plumber",
             gold_targets=[NEW], context=context),
    ]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "w.json"
    save(path, seed=1, units=[(_snapshot(), _cases())], meta={"model": "m"})
    [(label, world, cases)] = load_lived(path)
    assert label == "lived-2"
    assert [a.name for a in world.agents] == ["Email Alice", "Lisbon flight"]
    assert world.agents[0].history == [("draft to Alice", "")] and world.agents[0].threads == ["t1"]
    assert cases == _cases()
    assert isinstance(cases[0].context[0], tuple)
    assert world.now.isoformat().startswith("2026-05-31")  # LIFE_START + 60 days


def test_load_rejects_unknown_format(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"format": 99, "seed": 1, "sizes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        load_lived(path)


def test_render_context_uses_real_timestamps_for_three_tuples():
    case = Case(id="c", kind="follow_up", world_seed=1, message="m", gold_targets=[],
                context=[("user_message", "hi <there>", STAMP), ("poke_reply", "hello")])
    text = render_context(case)
    assert '<user_message timestamp="2026-06-01 09:30:00">hi &lt;there&gt;</user_message>' in text
    assert "<poke_reply timestamp=" in text  # 2-tuples keep the synthetic stamp


@pytest.mark.parametrize("body", ["world: imaginary", "lived_path: 3"])
def test_config_validates_world_keys(tmp_path, body):
    config = tmp_path / "c.yaml"
    config.write_text(f"name: t\n{body}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config)


async def test_runner_routes_a_lived_world(tmp_path):
    path = tmp_path / "w.json"
    save(path, seed=1, units=[(_snapshot(), _cases())], meta={})
    cfg = Config(name="t", world="lived", lived_path=str(path), n_agents=[2], k=1, include_handwritten=False)
    out = await run_experiment(cfg, llm=fake_llm(), out_dir=tmp_path / "res", split="dev", final=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    rows = data["routing_rows"]
    dev_ids = {c.id for c in _cases() if c.split == "dev"}
    assert {r["case_id"] for r in rows} == dev_ids
    assert all(r["n_agents"] == "lived-2" for r in rows)
    assert set(data["summary"]["retrieval"]) <= {"lived-2"}
