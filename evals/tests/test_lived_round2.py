import asyncio
import inspect
import json
import re
import shutil

import pytest

from evals.cases import NEW, Case, filter_cases
from evals.worlds.lived import round2, store
from evals.worlds.lived.plan import build_plan
from evals.worlds.lived.round2 import (
    TestCase, add_reviewed, apply_decisions, fake_writer, generate_round2, rebuild_snapshot, reviewed_path,
)
from evals.worlds.lived.store import DEFAULT_PATH, load_lived

WORLD = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
ENTRIES = {entry["size"]: entry for entry in WORLD["sizes"]}
ID = re.compile(r"lived-s1-(50|200)-r2-([a-z_]+)-\d{2}")


@pytest.fixture(scope="module")
def body():
    out = asyncio.run(generate_round2(world_path=DEFAULT_PATH, llm=fake_writer(), model="m", per_kind=2,
                                      log=lambda *_: None))
    out["meta"]["fake_llm"] = False  # stands in for a real round-2 file
    return out


@pytest.fixture
def world(tmp_path):
    path = tmp_path / "world-s1.json"
    shutil.copyfile(DEFAULT_PATH, path)
    return path


def _reviewed(world, body, decisions):
    data = apply_decisions(body, decisions, source_sha256="x")
    reviewed_path(world, 1).write_text(json.dumps(data), encoding="utf-8")
    return data


@pytest.mark.parametrize("size", [50, 200])
def test_rebuilt_snapshot_matches_the_saved_roster(size):
    entry = ENTRIES[size]
    snapshot = rebuild_snapshot(entry, build_plan(1), skipped=WORLD["meta"]["skipped"])
    [(_, saved_world, saved_cases)] = [u for u in load_lived() if u[0] == f"lived-{size}"]
    assert snapshot.agents == saved_world.agents
    assert snapshot.day == entry["day"] and snapshot.size == size
    assert snapshot.thread_agents == entry["thread_agents"]
    assert snapshot.transcript == saved_cases[0].context
    assert max(snapshot.thread_last_day.values()) == snapshot.day
    assert set(snapshot.thread_agents) <= set(snapshot.thread_last_day)
    assert snapshot.last_user_threads and set(snapshot.last_user_threads) <= set(snapshot.thread_last_day)


def test_rebuild_refuses_a_world_with_skipped_events():
    with pytest.raises(ValueError, match="skipped"):
        rebuild_snapshot(ENTRIES[50], build_plan(1), skipped=3)


def test_round2_ids_are_namespaced_unique_and_new(body):
    ids = [p["id"] for p in body["probes"]]
    assert ids and len(ids) == len(set(ids))
    existing = {c["id"] for entry in WORLD["sizes"] for c in entry["cases"]}
    assert not set(ids) & existing
    for probe in body["probes"]:
        match = ID.fullmatch(probe["id"])
        assert match and int(match.group(1)) == probe["size"] and match.group(2) == probe["kind"]
        assert probe["status"] == "unreviewed"
        assert probe["labels"] == (["lived-50"] if probe["size"] == 50 else ["lived-200", "lived-1000"])
        roster = {a["id"]: a for a in ENTRIES[probe["size"]]["agents"]}
        assert [a["id"] for a in probe["gold_agents"]] == (
            [g for g in probe["gold_targets"] if g != NEW] + probe["acceptable_ids"])
        assert all(a["name"] == roster[a["id"]]["name"] for a in probe["gold_agents"])
    assert body["meta"]["rng_seeds"] == {"50": 2001050, "200": 2001200}


def test_round2_messages_never_repeat_an_existing_case(body):
    existing = {" ".join(c["message"].lower().split()) for entry in WORLD["sizes"] for c in entry["cases"]}
    messages = [" ".join(p["message"].lower().split()) for p in body["probes"]]
    assert len(messages) == len(set(messages)) and not set(messages) & existing


def test_duplicate_drafts_are_skipped_and_ids_stay_contiguous(monkeypatch):
    old = ENTRIES[50]["cases"][0]["message"]

    async def fake_build(snapshot, plan, **kwargs):
        def case(i, message):
            return Case(id=f"lived-s1-{snapshot.size}-new_task-{i:02d}", kind="new_task", world_seed=1,
                        message=message, gold_targets=[NEW], context=[])
        return [case(0, old.upper()), case(1, f"fresh {snapshot.size}"), case(2, f"FRESH  {snapshot.size}")]

    monkeypatch.setattr(round2, "build_probes", fake_build)
    out = asyncio.run(generate_round2(world_path=DEFAULT_PATH, llm=fake_writer(), model="m", per_kind=1,
                                      log=lambda *_: None))
    assert [p["id"] for p in out["probes"]] == ["lived-s1-50-r2-new_task-00", "lived-s1-200-r2-new_task-00"]
    assert len(out["meta"]["skipped_duplicates"]) == 4


def test_no_reviewed_file_leaves_units_untouched(world):
    units = load_lived(world)
    assert add_reviewed(world, json.loads(world.read_text(encoding="utf-8")), units) is units
    saved = {c["id"] for entry in WORLD["sizes"] for c in entry["cases"]}
    assert all(c.id in saved for _, _, cases in load_lived(world) for c in cases)
    assert all(type(c) is Case for _, _, cases in units for c in cases)


def _decisions(body):
    p50 = [p for p in body["probes"] if p["size"] == 50]
    p200 = next(p for p in body["probes"] if p["size"] == 200 and p["gold_targets"] != [NEW] and p["gold_targets"])
    other = next(a["id"] for a in body["rosters"]["lived-200"] if a["id"] not in p200["gold_targets"])
    return p50, p200, other, [
        {"id": p50[0]["id"], "decision": "accept"},
        {"id": p50[1]["id"], "decision": "reject", "note": "ambiguous"},
        {"id": p200["id"], "decision": "fixed", "gold_targets": [other], "acceptable_ids": []},
    ]


def test_reviewed_probes_join_their_sizes_as_test_only(world, body):
    p50, p200, other, decisions = _decisions(body)
    base = {label: cases for label, _, cases in load_lived(world)}  # before the reviewed file exists
    _reviewed(world, body, decisions)
    units = {label: cases for label, _, cases in load_lived(world)}  # load_lived adds the reviewed probes
    added = {label: units[label][len(base[label]):] for label in units}
    assert units["lived-10"] == base["lived-10"] and not added["lived-10"]
    assert [c.id for c in added["lived-50"]] == [p50[0]["id"]]  # the rejected one is left out
    assert [c.id for c in added["lived-200"]] == [c.id for c in added["lived-1000"]] == [p200["id"]]
    fixed = added["lived-200"][0]
    assert fixed.gold_targets == [other] and fixed.acceptable_ids == []
    assert isinstance(fixed, TestCase) and isinstance(fixed.context[0], tuple)
    for cases in added.values():
        assert all(c.split == "test" for c in cases)
        assert filter_cases(cases, split="dev") == [] and filter_cases(cases, split="test") == cases
    undecided = {p["id"] for p in body["probes"]} - {d["id"] for d in decisions}
    assert not undecided & {c.id for cases in units.values() for c in cases}


def test_reviewed_file_for_another_world_is_refused(world, body):
    _, _, _, decisions = _decisions(body)
    _reviewed(world, body, decisions)
    data = json.loads(world.read_text(encoding="utf-8"))
    world.write_text(world.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="different"):
        add_reviewed(world, data, load_lived(world))


@pytest.mark.parametrize("bad, match", [
    ({"id": "nope", "decision": "accept"}, "unknown"),
    ({"decision": "maybe"}, "decision must be"),
    ({"decision": "fixed"}, "needs gold"),
    ({"decision": "accept", "gold_targets": [NEW]}, "only a fixed"),
    ({"decision": "fixed", "gold_targets": ["no-such-agent"]}, "not in its roster"),
])
def test_apply_rejects_bad_decisions(body, bad, match):
    item = {"id": body["probes"][0]["id"], **bad}
    with pytest.raises(ValueError, match=match):
        apply_decisions(body, [item], source_sha256="x")


def test_apply_rejects_repeats_and_fake_files(body):
    first = {"id": body["probes"][0]["id"], "decision": "accept"}
    with pytest.raises(ValueError, match="twice"):
        apply_decisions(body, [first, first], source_sha256="x")
    with pytest.raises(ValueError, match="fake"):
        apply_decisions({**body, "meta": {**body["meta"], "fake_llm": True}}, [first], source_sha256="x")


def test_apply_cli_writes_the_reviewed_file(world, body, tmp_path):
    round2.round2_path(world, 1).write_text(json.dumps(body), encoding="utf-8")
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(_decisions(body)[3]), encoding="utf-8")
    assert round2.main(["--world", str(world), "--apply", str(decisions)]) == 0
    written = json.loads(reviewed_path(world, 1).read_text(encoding="utf-8"))
    assert [p["decision"] for p in written["probes"]] == ["accept", "reject", "fixed"]


@pytest.mark.skipif("add_reviewed" not in inspect.getsource(store), reason="store.load_lived hook not applied yet")
def test_load_lived_includes_reviewed_probes(world, body):
    p50, p200, _, decisions = _decisions(body)
    _reviewed(world, body, decisions)
    ids = {label: [c.id for c in cases] for label, _, cases in load_lived(world)}
    assert p50[0]["id"] in ids["lived-50"] and p50[1]["id"] not in ids["lived-50"]
    assert p200["id"] in ids["lived-200"] and p200["id"] in ids["lived-1000"]


def test_saved_probe_state_rebuilds_without_replay(tmp_path):
    from evals.worlds.lived.store import save

    entry = ENTRIES[200]
    original = rebuild_snapshot(entry, build_plan(1), skipped=WORLD["meta"]["skipped"])
    path = tmp_path / "w.json"
    save(path, seed=1, units=[(original, [])], meta={"skipped": 5})
    saved = json.loads(path.read_text(encoding="utf-8"))["sizes"][0]
    again = rebuild_snapshot(saved, build_plan(1), skipped=5)  # skips no longer matter
    assert again.agents == original.agents and again.thread_agents == original.thread_agents
    assert again.transcript == original.transcript[-20:]
    assert again.thread_last_day == original.thread_last_day
    assert again.last_user_threads == original.last_user_threads
