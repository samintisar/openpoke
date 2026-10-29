import json

from evals.worlds.lived.bench_v2 import QUOTA, balance, mark_ambiguous


def _probe(pid, size, kind):
    return {"id": pid, "size": size, "kind": kind}


def test_balance_trims_to_the_smaller_count_per_size_and_kind():
    a = [_probe(f"a{i}", 200, "typo") for i in range(5)] + [_probe("a9", 50, "typo")]
    b = [_probe(f"b{i}", 200, "typo") for i in range(3)]
    a2, b2 = balance(a, b)
    assert [p["id"] for p in a2] == ["v2-sx-200-typo-00", "v2-sx-200-typo-01", "v2-sx-200-typo-02"]
    assert len(b2) == 3 and not [p for p in a2 if p["size"] == 50]  # life b has no 50-agent typo probes


def test_balance_keeps_order_and_numbers_by_world_seed():
    a = [{**_probe(f"a{i}", 10, "typo"), "world_seed": 1} for i in range(2)]
    b = [{**_probe(f"b{i}", 10, "typo"), "world_seed": 2} for i in range(4)]
    a2, b2 = balance(a, b)
    assert [p["id"] for p in a2] == ["v2-s1-10-typo-00", "v2-s1-10-typo-01"]
    assert [p["id"] for p in b2] == ["v2-s2-10-typo-00", "v2-s2-10-typo-01"]
    assert a[0]["id"] == "a0"  # inputs untouched


def test_quota():
    assert QUOTA == {10: 3, 50: 5, 200: 10}


def test_mark_ambiguous_flags_same_person_overlapping_topics():
    from evals.worlds.lived.model import Contact, LifePlan, Thread
    plan = LifePlan(seed=9, persona="p", contacts=[Contact("Ann", "Lee", "friend")], events=[], threads=[
        Thread("t1", "Ann Lee", "ski trip", "friend"), Thread("t2", "Ann Lee", "ski trip photos", "friend"),
        Thread("t3", "Ann Lee", "book club", "friend"),
    ])
    agents = {"t1": ["a1"], "t2": ["a2"], "t3": ["a3"]}
    assert mark_ambiguous(plan, agents, "t1") is True
    assert mark_ambiguous(plan, agents, "t3") is False
    assert mark_ambiguous(plan, {"t1": ["a1"], "t2": []}, "t1") is False  # other thread has no agent


async def test_build_life_offline_on_life1():
    from evals.worlds.lived import bench_v2
    from evals.worlds.lived.round2 import fake_writer
    body = await bench_v2.build_life(1, llm=fake_writer(), model="m")
    kinds = {p["kind"] for p in body["probes"]}
    assert {"status_question", "paraphrase", "follow_up"} <= kinds
    counts = {}
    for p in body["probes"]:
        counts[(p["size"], p["kind"])] = counts.get((p["size"], p["kind"]), 0) + 1
    assert all(n <= QUOTA[size] for (size, _), n in counts.items())
    assert all("lived-1000" in p["labels"] for p in body["probes"] if p["size"] == 200)
    assert body["meta"]["seed"] == 1 and len(body["meta"]["world_sha256"]) == 64
    for p in body["probes"]:
        roster = {a["id"] for a in body["rosters"][p["labels"][0]]}
        assert {a["id"] for a in p["gold_agents"]} <= roster
        assert isinstance(p["clarify_ok"], bool)


def test_cli_build_single_seed_writes_unbalanced_file(tmp_path, capsys):
    from evals.worlds.lived.bench_v2 import main
    assert main(["build", "--seed", "1", "--fake-llm", "--out-dir", str(tmp_path)]) == 0
    body = json.loads((tmp_path / "bench-v2-s1.json").read_text(encoding="utf-8"))
    assert body["meta"]["seed"] == 1 and body["probes"]
    assert all(p["id"].startswith("v2-s1-") for p in body["probes"])
    assert "unbalanced" in capsys.readouterr().out
