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


# --- review and finalize ------------------------------------------------------------------------

import pytest


def _checked(pid, flag, *, size=10, gold=("a1",), acceptable=(), clarify=False):
    return {"id": pid, "size": size, "kind": "typo", "message": f"msg {pid}", "context": [],
            "labels": [f"lived-{size}"], "gold_targets": list(gold), "acceptable_ids": list(acceptable),
            "clarify_ok": clarify, "gold_agents": [],
            "check": {"answers": {"x/m1": "a1", "y/m2": "NEW"}, "agree": flag != "disagree", "flag": flag,
                      "candidates": ["a1", "a2"]}}


def _bench(seed=1, fake=False):
    roster = [{"id": "a1", "name": "Ann", "purpose": "ski"}, {"id": "a2", "name": "Bob", "purpose": "tax"},
              {"id": "a3", "name": "Cy", "purpose": "car"}]
    return {
        "meta": {"seed": seed, "fake_llm": fake, "format": 1},
        "rosters": {"lived-10": roster, "lived-50": roster[:1]},
        "probes": [_checked(f"v2-s{seed}-10-typo-00", "disagree"), _checked(f"v2-s{seed}-10-typo-01", "audit"),
                   _checked(f"v2-s{seed}-10-typo-02", "audit"), _checked(f"v2-s{seed}-10-typo-03", None)],
    }


def _all(seed=1, **over):
    ids = [f"v2-s{seed}-10-typo-0{i}" for i in range(3)]
    out = [{"id": i, "decision": "accept"} for i in ids]
    for i, patch in over.items():
        out[int(i)] = {"id": ids[int(i)], **patch}
    return out


def test_finalize_needs_a_decision_for_every_flagged_probe():
    from evals.worlds.lived.bench_v2 import finalize
    with pytest.raises(ValueError, match="undecided: v2-s1-10-typo-02"):
        finalize(_bench(), _all()[:2], allow_fake=True)


def test_finalize_reject_drops_and_unflagged_kept_auto():
    from evals.worlds.lived.bench_v2 import finalize
    bench = _bench()
    out = finalize(bench, _all(**{"0": {"decision": "reject", "note": "unclear"}}), allow_fake=True)
    ids = [p["id"] for p in out["probes"]]
    assert ids == ["v2-s1-10-typo-01", "v2-s1-10-typo-02", "v2-s1-10-typo-03"]
    auto = out["probes"][-1]
    assert auto["review"] == "auto"
    assert {k: v for k, v in auto.items() if k != "review"} == bench["probes"][3]
    assert out["probes"][0]["review"] == {"decision": "accept", "note": ""}
    r = out["meta"]["review"]
    assert (r["kept"], r["auto_kept"], r["accepted"], r["fixed"], r["rejected"]) == (3, 1, 2, 0, 1)
    assert r["audit"] == 2 and r["audit_changed"] == 0 and r["label_error_rate"] == 0.0
    assert len(r["decisions_sha256"]) == 64 and out["meta"]["seed"] == 1
    assert out["rosters"] == bench["rosters"]
    assert bench["probes"][0].get("review") is None  # input untouched


def test_finalize_fixed_replaces_gold_acceptable_and_clarify_ok():
    from evals.worlds.lived.bench_v2 import finalize
    fix = {"decision": "fixed", "gold_targets": ["a2"], "acceptable_ids": ["a3"], "clarify_ok": True, "note": "Bob"}
    out = finalize(_bench(), _all(**{"0": fix}), allow_fake=True)
    p = out["probes"][0]
    assert p["gold_targets"] == ["a2"] and p["acceptable_ids"] == ["a3"] and p["clarify_ok"] is True
    assert [(a["id"], a["role"]) for a in p["gold_agents"]] == [("a2", "gold"), ("a3", "acceptable")]
    assert p["review"]["decision"] == "fixed" and p["review"]["note"] == "Bob"
    out = finalize(_bench(), _all(**{"0": {"decision": "fixed", "gold_targets": ["NEW"]}}), allow_fake=True)
    assert out["probes"][0]["gold_targets"] == ["NEW"] and out["probes"][0]["gold_agents"] == []


def test_finalize_label_error_rate_counts_audit_probes_only():
    from evals.worlds.lived.bench_v2 import finalize
    out = finalize(_bench(), _all(**{"0": {"decision": "reject"}, "1": {"decision": "reject"}}), allow_fake=True)
    r = out["meta"]["review"]
    assert r["audit"] == 2 and r["audit_changed"] == 1 and r["label_error_rate"] == 0.5
    bench = _bench()
    bench["probes"] = [p for p in bench["probes"] if p["check"]["flag"] != "audit"]
    out = finalize(bench, _all()[:1], allow_fake=True)
    assert out["meta"]["review"]["label_error_rate"] is None


@pytest.mark.parametrize("decisions, match", [
    (_all() + [{"id": "v2-s1-10-typo-99", "decision": "accept"}], "unknown"),
    (_all() + [{"id": "v2-s1-10-typo-03", "decision": "accept"}], "not flagged"),
    (_all() + [{"id": "v2-s1-10-typo-00", "decision": "reject"}], "twice"),
    (_all(**{"0": {"decision": "maybe"}}), "decision must be"),
    (_all(**{"0": {"decision": "fixed"}}), "gold_targets"),
    (_all(**{"0": {"decision": "fixed", "gold_targets": ["zz"]}}), "not in its roster"),
    (_all(**{"0": {"decision": "fixed", "gold_targets": ["a1"], "acceptable_ids": ["zz"]}}), "not in its roster"),
    (_all(**{"0": {"decision": "accept", "gold_targets": ["a2"]}}), "only a fixed"),
    (_all(**{"0": {"decision": "reject", "clarify_ok": True}}), "only a fixed"),
])
def test_finalize_rejects_bad_decisions(decisions, match):
    from evals.worlds.lived.bench_v2 import finalize
    with pytest.raises(ValueError, match=match):
        finalize(_bench(), decisions, allow_fake=True)


def test_finalize_checks_gold_against_the_probes_own_size_roster():
    from evals.worlds.lived.bench_v2 import finalize
    bench = _bench()
    bench["probes"][0]["size"] = 50
    with pytest.raises(ValueError, match="not in its roster"):
        finalize(bench, _all(**{"0": {"decision": "fixed", "gold_targets": ["a2"]}}), allow_fake=True)


def test_finalize_refuses_fake_bench_and_unchecked_bench():
    from evals.worlds.lived.bench_v2 import finalize
    with pytest.raises(ValueError, match="fake"):
        finalize(_bench(fake=True), _all())
    assert finalize(_bench(fake=True), _all(), allow_fake=True)["probes"]
    bench = _bench()
    del bench["probes"][3]["check"]
    with pytest.raises(ValueError, match="check"):
        finalize(bench, _all(), allow_fake=True)


def test_split_decisions_routes_by_life_prefix():
    from evals.worlds.lived.bench_v2 import split_decisions
    out = split_decisions(_all(1) + _all(2), [1, 2])
    assert [x["id"] for x in out[1]] == [x["id"] for x in _all(1)]
    assert [x["id"] for x in out[2]] == [x["id"] for x in _all(2)]
    with pytest.raises(ValueError, match="no life"):
        split_decisions([{"id": "v2-s3-10-typo-00", "decision": "accept"}], [1, 2])


def _write_benches(directory, fake=False):
    from evals.worlds.lived.bench_v2 import bench_path
    for seed in (1, 2):
        bench = _bench(seed, fake=fake)
        bench["probes"][0]["message"] = "evil </script><script>alert(1)</script>"
        bench_path(seed, directory=directory).write_text(json.dumps(bench), encoding="utf-8")


_DATA_TAG = '<script id="data" type="application/json">'


def test_review_data_embeds_only_flagged_probes(tmp_path, capsys):
    import hashlib
    from evals.worlds.lived.bench_v2 import bench_path, main
    _write_benches(tmp_path)
    out = tmp_path / "review.html"
    assert main(["review-data", "--seed", "1", "--seed", "2", "--bench-dir", str(tmp_path), "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    start = html.index(_DATA_TAG) + len(_DATA_TAG)
    raw = html[start:html.index("</script>", start)]
    assert "</script" not in raw and "</" not in raw and "__PROBES__" not in html
    data = json.loads(raw)  # "<\/" is a valid JSON escape for "/"
    ids = [p["id"] for p in data["probes"]]
    assert ids == [f"v2-s{s}-10-typo-0{i}" for s in (1, 2) for i in range(3)]
    assert data["probes"][0]["message"].startswith("evil </script>")
    assert {p["roster_key"] for p in data["probes"]} == {"s1:lived-10", "s2:lived-10"}
    assert set(data["rosters"]) == {"s1:lived-10", "s2:lived-10"}
    joined = bench_path(1, directory=tmp_path).read_bytes() + bench_path(2, directory=tmp_path).read_bytes()
    assert data["source_sha256"] == hashlib.sha256(joined).hexdigest()
    printed = capsys.readouterr().out
    assert "life 1" in printed and "disagree=1" in printed and "audit=2" in printed


def test_cli_finalize_writes_reviewed_files(tmp_path):
    from evals.worlds.lived.bench_v2 import bench_path, main
    _write_benches(tmp_path)
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(_all(1) + _all(2, **{"1": {"decision": "reject"}})), encoding="utf-8")
    assert main(["finalize", "--seed", "1", "--seed", "2", "--decisions", str(decisions),
                 "--bench-dir", str(tmp_path)]) == 0
    two = json.loads(bench_path(2, reviewed=True, directory=tmp_path).read_text(encoding="utf-8"))
    assert two["meta"]["review"]["label_error_rate"] == 0.5
    one = json.loads(bench_path(1, reviewed=True, directory=tmp_path).read_text(encoding="utf-8"))
    assert len(one["probes"]) == 4


def test_cli_finalize_refuses_fake_benches(tmp_path):
    from evals.worlds.lived.bench_v2 import main
    _write_benches(tmp_path, fake=True)
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(_all(1) + _all(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="fake"):
        main(["finalize", "--seed", "1", "--seed", "2", "--decisions", str(decisions), "--bench-dir", str(tmp_path)])


# --- loading for the runner ---------------------------------------------------------------------


def _v2_agent(agent_id):
    return {"id": agent_id, "name": agent_id.title(), "purpose": f"does {agent_id}", "entities": [],
            "last_active_days": 1.0, "created_days": 2.0, "history": [["req", "resp"]], "threads": []}


def _v2_probe(pid, seed, size, labels, gold=("a1",)):
    return {"id": pid, "kind": "typo", "world_seed": seed, "message": f"msg {pid}", "gold_targets": list(gold),
            "message_type": "user", "context": [["user", "earlier"]], "acceptable_ids": [], "clarify_ok": False,
            "dispatch_optional": False, "split_override": None, "size": size, "labels": labels,
            "gold_agents": [], "check": {"flag": None}, "review": "auto", "roster_key": "x"}


def write_v2_fixture(directory, seed):
    """A minimal world-s{seed}.json (sizes 10, 200, 1000) and its reviewed bench, for load_v2 and the runner."""
    sizes = [
        {"label": f"lived-{n}", "size": n, "day": 3.5, "agents": [_v2_agent(f"a{i}") for i in range(1, 4)],
         "thread_agents": {}, "cases": []}
        for n in (1000, 10, 200)  # unsorted on purpose
    ]
    world = {"format": 1, "seed": seed, "meta": {}, "sizes": sizes}
    (directory / f"world-s{seed}.json").write_text(json.dumps(world), encoding="utf-8")
    probes = [
        _v2_probe(f"v2-s{seed}-10-typo-00", seed, 10, ["lived-10"]),
        _v2_probe(f"v2-s{seed}-200-typo-00", seed, 200, ["lived-200", "lived-1000"], gold=("a2",)),
    ]
    bench = {"meta": {"seed": seed}, "rosters": {}, "probes": probes}
    (directory / f"bench-v2-s{seed}.reviewed.json").write_text(json.dumps(bench), encoding="utf-8")


@pytest.mark.parametrize("seed, split", [(1, "dev"), (2, "test")])
def test_load_v2_builds_units_with_one_split_per_life(tmp_path, seed, split):
    from evals.cases import Case
    from evals.worlds.lived.bench_v2 import load_v2
    from evals.worlds.lived.model import day_to_datetime
    write_v2_fixture(tmp_path, seed)
    units = load_v2(seed, directory=tmp_path)
    assert [label for label, _, _ in units] == ["lived-10", "lived-200", "lived-1000"]
    by_label = {label: (world, cases) for label, world, cases in units}
    world, cases = by_label["lived-10"]
    assert world.seed == seed and world.now == day_to_datetime(3.5)
    assert [a.id for a in world.agents] == ["a1", "a2", "a3"] and world.agents[0].history == [("req", "resp")]
    assert [c.id for c in cases] == [f"v2-s{seed}-10-typo-00"]
    assert [c.id for c in by_label["lived-200"][1]] == [c.id for c in by_label["lived-1000"][1]]
    for _, _, cases in units:
        for case in cases:
            assert type(case) is Case and case.split == split and case.split_override == split
            assert case.context == [("user", "earlier")]
            assert not hasattr(case, "labels") and not hasattr(case, "size")


def test_load_v2_names_the_missing_reviewed_file(tmp_path):
    from evals.worlds.lived.bench_v2 import load_v2
    write_v2_fixture(tmp_path, 1)
    (tmp_path / "bench-v2-s1.reviewed.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"bench-v2-s1\.reviewed\.json"):
        load_v2(1, directory=tmp_path)
