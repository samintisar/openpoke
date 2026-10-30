import copy
import hashlib
import json

import pytest

from evals.worlds.lived import replay
from evals.worlds.lived.grow import GrowthResult
from evals.worlds.lived.model import Snapshot

STAMP = "2026-06-01T09:30:00+00:00"


def _agent(agent_id, *requests):
    return {"id": agent_id, "name": agent_id.title(), "purpose": requests[0], "entities": [], "last_active_days": 1.0,
            "created_days": 2.0, "person": None, "topic": None, "core": False,
            "history": [[r, ""] for r in requests], "threads": ["t1"]}


def _entry(size, agents, *, state=True):
    entry = {"label": f"lived-{size}", "size": size, "day": 5.0, "agents": agents, "thread_agents": {"t1": ["a"]}}
    if state:
        entry.update(transcript=[["user_message", "hi", STAMP]], thread_last_day={"t1": 5.0}, last_user_threads=["t1"])
    entry["cases"] = []
    return entry


def _world(**meta):
    a2 = [_agent("a", "plan the trip", "book it"), _agent("b", "find a plumber")]
    return {"format": 1, "seed": 1, "meta": {"model": "m/x", "generated_at": "then", "events_played": 9, **meta},
            "sizes": [_entry(1, [_agent("a", "plan the trip")]), _entry(2, a2), _entry(3, a2 + [_agent("bg", "x")])]}


LOGS = {"a": [(1, "instruction", "plan the trip"), (3, "result", "Flights found."), (4, "instruction", "book it"),
              (6, "result", "Booked.")],
        "b": [(4, "instruction", "find a plumber")]}


def _fake_generate(regenerated, seen, logs=LOGS):
    async def generate(*, seed, model, llm, per_kind, out, log=print, on_grown=None):
        seen.update(seed=seed, model=model, per_kind=per_kind)
        snapshots = [Snapshot(size=n, day=5.0, agents=[], transcript=[], thread_agents={}, thread_last_day={},
                              event_index=i) for n, i in ((1, 2), (2, 5), (3, 5))]
        on_grown(GrowthResult(snapshots=snapshots[:2], agents=[], transcript=[], logs=logs), snapshots)
        out.write_text(json.dumps(regenerated, indent=1), encoding="utf-8")
        return out
    return generate


def _write(tmp_path, world):
    path = tmp_path / "world-s1.json"
    path.write_text(json.dumps(world, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


async def test_replay_returns_the_logs_when_the_world_is_reproduced(tmp_path, monkeypatch):
    world = _world()
    path = _write(tmp_path, world)
    seen = {}
    monkeypatch.setattr(replay, "generate", _fake_generate({**world, "meta": {**world["meta"], "generated_at": "now"}}, seen))
    body = await replay.replay(1, llm=object(), directory=tmp_path, log=lambda _: None)
    assert seen == {"seed": 1, "model": "m/x", "per_kind": 0}  # the world's own model; no probes drafted
    assert body["world_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert body["cutoffs"] == {"lived-1": 2, "lived-2": 5, "lived-3": 5}
    assert body["logs"]["a"][1] == [3, "result", "Flights found."] and body["seed"] == 1
    assert replay.counts(world, body)["lived-2"] == {"agents": 2, "with_log": 2, "results": 1, "with_result": 1}


async def test_replay_refuses_a_world_that_differs(tmp_path, monkeypatch):
    world = _world()
    _write(tmp_path, world)
    changed = copy.deepcopy(world)
    changed["sizes"][1]["agents"][1]["purpose"] = "something else"
    monkeypatch.setattr(replay, "generate", _fake_generate(changed, {}))
    with pytest.raises(replay.ReplayMismatch, match=r"lived-2\.agents"):
        await replay.replay(1, llm=object(), directory=tmp_path, log=lambda _: None)


async def test_replay_refuses_logs_that_disagree_with_the_histories(tmp_path, monkeypatch):
    world = _world()
    _write(tmp_path, world)
    logs = {**LOGS, "b": [(4, "instruction", "find an electrician")]}
    monkeypatch.setattr(replay, "generate", _fake_generate(world, {}, logs))
    with pytest.raises(replay.ReplayMismatch, match="lived-2: b"):
        await replay.replay(1, llm=object(), directory=tmp_path, log=lambda _: None)


def test_entries_saved_before_the_probe_state_are_checked_against_the_rebuilt_snapshot(monkeypatch):
    committed = _world()
    for entry in committed["sizes"]:
        for key in replay.PROBE_STATE:
            del entry[key]
        entry["cases"] = [{"id": "c", "context": [["user_message", "hi", STAMP]]}]
    regenerated = _world()
    rebuilt = Snapshot(size=0, day=5.0, agents=[], transcript=[("user_message", "hi", STAMP)], thread_agents={},
                       thread_last_day={"t1": 5.0}, last_user_threads=("t1",))
    monkeypatch.setattr(replay, "rebuild_snapshot", lambda entry, plan, skipped: rebuilt)
    assert replay.differences(committed, regenerated, plan=None) == []
    regenerated["sizes"][0]["thread_last_day"] = {"t1": 4.0}
    assert replay.differences(committed, regenerated, plan=None) == ["lived-1.thread_last_day"]


def test_main_replays_with_a_zero_budget_and_writes_the_logs(tmp_path, monkeypatch):
    seen = {}

    async def fake_replay(seed, *, llm, directory, log=print):
        seen["cap"] = llm.budget.cap_usd
        return {"format": 1, "seed": seed, "world_sha256": "0" * 64, "cutoffs": {}, "logs": {}}

    monkeypatch.setattr(replay, "replay", fake_replay)
    monkeypatch.setattr(replay, "_world", lambda seed, directory: {"sizes": []})
    assert replay.main(["--seed", "3", "--dir", str(tmp_path)]) == 0
    assert seen["cap"] == 0
    assert json.loads((tmp_path / "agent-logs-s3.json").read_text(encoding="utf-8"))["seed"] == 3
