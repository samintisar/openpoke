"""Benchmark v2 probe sets: per-size quotas, both lives balanced per kind, ambiguity marked.

    python -m evals.worlds.lived.bench_v2 build --seed 1 --seed 2 --budget 2   # bench-v2-s1/s2.json
    python -m evals.worlds.lived.bench_v2 build --seed 1 --fake-llm            # one life, offline, unbalanced
    python -m evals.worlds.lived.bench_v2 majority --seed 1 --seed 2            # re-flag by 2-of-3 checkers
    python -m evals.worlds.lived.bench_v2 review-data --seed 1 --seed 2         # page of the flagged probes
    python -m evals.worlds.lived.bench_v2 finalize --seed 1 --seed 2 --decisions decisions.json

For each life (seed) the 10-, 50- and 200-agent snapshots are rebuilt from world-s{seed}.json
(round2.rebuild_snapshot) and probed with build_probes at QUOTA[size] per kind. The 200 probes also
carry the lived-1000 label when the world has a 1000 entry, as round 2 does.

Ambiguity: a probe is clarify_ok when the person behind its best gold agent's thread has another
agent-backed thread whose topic shares a content word (plan.topic_words) -- asking which one is a
fair answer there. Cases carry no thread id, so the probe's thread is recovered from the gold agent:
every thread whose first-served agent is the best gold agent is considered (one agent can own several
threads), and the probe is ambiguous if any of them is.

Balancing: with two lives, each (size, kind) keeps min(count_a, count_b) probes in each life, the
first ones in build order, renumbered v2-s{seed}-{size}-{kind}-{i:02d}.

Review: after checkers.py, only flagged probes ("disagree" or "audit") go to a human, on the round-2
review page (review-data, which also shows each checker's answer). finalize applies the decisions per
life: rejected probes are dropped, fixed ones take the reviewer's gold, accepted ones are kept as they
are, and unflagged probes are kept untouched with review "auto". The label error rate is the share of
audit probes (a random sample of those every checker agreed with) that the reviewer fixed or rejected.

Majority rule (majority, run between checkers.py and review-data): to keep the review small, a probe
two of the three checkers agree with skips review too, and only the rest stay "disagree". The audit is
then redrawn as MAJORITY_AUDIT probes per life from everything kept, so the label error rate measures
the probes no human looked at.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import random
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from ...cases import NEW, Case
from ...llm import LLMClient
from ..model import World
from .model import LifePlan, Snapshot, day_to_datetime
from .plan import build_plan, topic_words
from .probes import build_probes
from .round2 import DECISIONS, _sha256, fake_writer, rebuild_snapshot
from .store import FORMAT as WORLD_FORMAT
from .store import _agent

HERE = Path(__file__).resolve().parent
PAGE = HERE.parents[1] / "labelling" / "probes_page.html"
REVIEW_PAGE = HERE.parents[1] / "labelling" / "bench_v2_review.html"
QUOTA = {10: 3, 50: 5, 200: 10}
FULL_SIZE = 1000  # reuses the 200 snapshot's probes
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
FORMAT = 1
MAJORITY_MIN_AGREE = 2  # of the three checkers
MAJORITY_AUDIT = 10  # audit probes per life, drawn from everything majority rule keeps


DEV_LIFE, TEST_LIFE = 1, 2  # life 2 is sealed: the runner only loads it for --split test --final


def world_path(seed: int, directory: Path | None = HERE) -> Path:
    return Path(directory or HERE) / f"world-s{seed}.json"


def bench_path(seed: int, *, reviewed: bool = False, directory: Path | None = HERE) -> Path:
    return Path(directory or HERE) / f"bench-v2-s{seed}{'.reviewed' if reviewed else ''}.json"


def rng_seed(seed: int, size: int) -> int:
    return seed * 7000 + size


# --- ambiguity ----------------------------------------------------------------------------------


def mark_ambiguous(plan: LifePlan, thread_agents: dict[str, list[str]], thread_id: str) -> bool:
    """Whether the thread's person has another agent-backed thread sharing a topic word."""
    thread = plan.thread(thread_id)
    if not thread.contact:
        return False
    words = topic_words(thread.topic)
    return any(
        other.id != thread_id and other.contact == thread.contact and thread_agents.get(other.id)
        and bool(words & topic_words(other.topic))
        for other in plan.threads
    )


def _threads_of(case: Case, snapshot: Snapshot) -> list[str]:
    """Every thread whose first-served agent is the case's best gold agent (see the module doc)."""
    best = next((g for g in case.gold_targets if g != NEW), None)
    if best is None:
        return []
    return [t for t, ids in snapshot.thread_agents.items() if ids and ids[0] == best]


def _is_ambiguous(case: Case, snapshot: Snapshot, plan: LifePlan) -> bool:
    return any(mark_ambiguous(plan, snapshot.thread_agents, t) for t in _threads_of(case, snapshot))


# --- balancing ----------------------------------------------------------------------------------


def _renumber(probes: list[dict[str, Any]], keep: dict[tuple[int, str], int] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: dict[tuple[int, str], int] = {}
    for probe in probes:
        key = (probe["size"], probe["kind"])
        index = seen.get(key, 0)
        if keep is not None and index >= keep.get(key, 0):
            continue
        seen[key] = index + 1
        seed = probe.get("world_seed", "x")
        out.append({**probe, "id": f"v2-s{seed}-{probe['size']}-{probe['kind']}-{index:02d}"})
    return out


def _counts(probes: list[dict[str, Any]]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    for probe in probes:
        key = (probe["size"], probe["kind"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def balance(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Both lives trimmed to the smaller count per (size, kind), first probes kept, ids renumbered."""
    count_a, count_b = _counts(a), _counts(b)
    keep = {key: min(n, count_b.get(key, 0)) for key, n in count_a.items()}
    return _renumber(a, keep), _renumber(b, keep)


# --- building -----------------------------------------------------------------------------------


def _labels(size: int, sizes_saved: set[int]) -> list[str]:
    labels = [f"lived-{size}"]
    if size == max(QUOTA) and FULL_SIZE in sizes_saved:
        labels.append(f"lived-{FULL_SIZE}")
    return labels


def _probe_record(case: Case, size: int, labels: list[str], roster: dict[str, Any]) -> dict[str, Any]:
    def agent(agent_id: str, role: str) -> dict[str, Any]:
        spec = roster[agent_id]
        return {"id": agent_id, "name": spec["name"], "purpose": spec["purpose"], "role": role}

    gold_agents = [agent(a, "gold") for a in case.gold_targets if a != NEW]
    gold_agents += [agent(a, "acceptable") for a in case.acceptable_ids if a != NEW]
    record = asdict(case)
    record["context"] = [list(c) for c in case.context]
    return {**record, "size": size, "labels": labels, "gold_agents": gold_agents}


async def build_life(seed: int, *, llm: LLMClient, model: str = DEFAULT_MODEL, directory: Path = HERE,
                     fake: bool = False, log=print) -> dict[str, Any]:
    """One life's unbalanced bench body; no file is written."""
    path = world_path(seed, directory)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["seed"] != seed:
        raise ValueError(f"{path}: holds seed {data['seed']}, not {seed}")
    plan = build_plan(seed)
    entries = {entry["size"]: entry for entry in data["sizes"]}
    rosters: dict[str, list[dict[str, Any]]] = {}
    probes: list[dict[str, Any]] = []
    for size, per_kind in QUOTA.items():
        entry = entries[size]
        snapshot = rebuild_snapshot(entry, plan, skipped=data["meta"].get("skipped", 0))
        roster = {a["id"]: a for a in entry["agents"]}
        rosters[entry["label"]] = [{"id": a["id"], "name": a["name"], "purpose": a["purpose"]} for a in entry["agents"]]
        cases = await build_probes(snapshot, plan, llm=llm, model=model, per_kind=per_kind,
                                   rng_seed=rng_seed(seed, size))
        labels = _labels(size, set(entries))
        flagged = 0
        for case in cases:
            if not case.clarify_ok and _is_ambiguous(case, snapshot, plan):
                case = replace(case, clarify_ok=True)
                flagged += 1
            probes.append(_probe_record(case, size, labels, roster))
        log(f"life {seed} size {size}: {len(cases)} probes, {flagged} newly marked clarify_ok")
    return {
        "meta": {"seed": seed, "world_sha256": _sha256(path), "model": model, "quota": dict(QUOTA),
                 "fake_llm": fake, "format": FORMAT},
        "rosters": rosters,
        "probes": probes,
    }


# --- review -------------------------------------------------------------------------------------


def _flagged(probe: dict[str, Any]) -> bool:
    return probe["check"]["flag"] is not None


def majority(bench: dict[str, Any], *, seed: int, audit: int = MAJORITY_AUDIT,
             min_agree: int = MAJORITY_MIN_AGREE) -> dict[str, Any]:
    """The bench re-flagged by majority rule: a probe at least `min_agree` checkers agree with is kept
    without review; the rest stay "disagree". The audit is redrawn as `audit` probes sampled from
    everything kept (unanimous and majority alike), so the label error rate covers what skipped review.
    The checkers' own flag is kept as check.unanimous_flag."""
    from .checkers import agrees  # lazy: checkers imports bench-side helpers

    if bench.get("meta", {}).get("review_rule"):
        raise ValueError("majority rule already applied to this bench")
    out = copy.deepcopy(bench)
    kept = []
    for probe in out["probes"]:
        check = probe["check"]
        count = sum(agrees(answer, probe) for answer in check["answers"].values())
        check.update(unanimous_flag=check["flag"], agree_count=count,
                     flag="disagree" if count < min_agree else None)
        if check["flag"] is None:
            kept.append(probe)
    for probe in random.Random(f"majority-audit:{seed}").sample(sorted(kept, key=lambda p: p["id"]),
                                                                min(audit, len(kept))):
        probe["check"]["flag"] = "audit"
    out.setdefault("meta", {})["review_rule"] = {"rule": "majority", "min_agree": min_agree, "audit": audit,
                                                 "seed": seed}
    return out


def review_data(benches: dict[int, dict[str, Any]], source_sha256: str) -> dict[str, Any]:
    """The page data: the flagged probes of every life, each pointing at its life's roster."""
    probes, rosters = [], {}
    for seed in sorted(benches):
        bench = benches[seed]
        for probe in bench["probes"]:
            if "check" not in probe:
                raise ValueError(f"life {seed}: probe {probe['id']} has no check; run checkers.py first")
            if not _flagged(probe):
                continue
            label = f"lived-{probe['size']}"
            key = f"s{seed}:{label}"
            rosters[key] = bench["rosters"][label]
            probes.append({**probe, "roster_key": key, "world_seed": seed})
    probes.sort(key=lambda p: (p["world_seed"], p["size"], p["kind"], p["id"]))
    return {"source_sha256": source_sha256, "rosters": rosters, "probes": probes}


def render_page(data: dict[str, Any], template: Path = PAGE) -> str:
    """The review page with `data` embedded; "</" is escaped so no string can close the script tag."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.read_text(encoding="utf-8").replace("__PROBES__", payload)


def split_decisions(decisions: list[dict[str, Any]], seeds: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Decisions per life, by the v2-s{seed}- id prefix."""
    out: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for item in decisions:
        probe_id = str(item.get("id"))
        seed = next((s for s in seeds if probe_id.startswith(f"v2-s{s}-")), None)
        if seed is None:
            raise ValueError(f"decision for {probe_id!r} belongs to no life in {seeds}")
        out[seed].append(item)
    return out


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _check_decision(item: dict[str, Any], probe: dict[str, Any], roster: dict[str, Any]) -> None:
    probe_id, decision = probe["id"], item.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"{probe_id}: decision must be one of {DECISIONS}, not {decision!r}")
    fixes = {k: item[k] for k in ("gold_targets", "acceptable_ids", "clarify_ok") if k in item}
    if decision != "fixed":
        if fixes:
            raise ValueError(f"{probe_id}: only a fixed decision may change gold")
        return
    gold, acceptable = fixes.get("gold_targets"), fixes.get("acceptable_ids", [])
    if not isinstance(gold, list) or not gold:
        raise ValueError(f"{probe_id}: a fixed decision needs gold_targets")
    if not isinstance(acceptable, list):
        raise ValueError(f"{probe_id}: acceptable_ids must be a list")
    bad = [a for a in [*gold, *acceptable] if a != NEW and a not in roster]
    if bad:
        raise ValueError(f"{probe_id}: gold names agents not in its roster: {bad}")
    if not isinstance(fixes.get("clarify_ok", False), bool):
        raise ValueError(f"{probe_id}: clarify_ok must be true or false")


def _apply_fix(probe: dict[str, Any], item: dict[str, Any], roster: dict[str, Any]) -> None:
    probe["gold_targets"] = list(item["gold_targets"])
    probe["acceptable_ids"] = list(item.get("acceptable_ids", []))
    probe["clarify_ok"] = item.get("clarify_ok", probe["clarify_ok"])
    probe["gold_agents"] = [
        {"id": a, "name": roster[a]["name"], "purpose": roster[a]["purpose"], "role": role}
        for role, ids in (("gold", probe["gold_targets"]), ("acceptable", probe["acceptable_ids"]))
        for a in ids if a != NEW
    ]


def finalize(bench: dict[str, Any], decisions: list[dict[str, Any]], *, allow_fake: bool = False,
             decisions_sha256: str | None = None) -> dict[str, Any]:
    """One life's reviewed bench: flagged probes decided by a human, unflagged ones kept as "auto".

    A decision is {id, decision: accept|fixed|reject, note?, gold_targets?, acceptable_ids?, clarify_ok?};
    fixed needs gold_targets (NEW allowed) and only fixed may carry the gold fields; a fix without
    acceptable_ids clears them, one without clarify_ok keeps the probe's. Every flagged probe needs
    exactly one decision. decisions_sha256 defaults to a hash of `decisions` as given."""
    if bench.get("meta", {}).get("fake_llm") and not allow_fake:
        raise ValueError("refusing to finalize a bench built with the fake writer (--fake-llm)")
    missing = [p["id"] for p in bench["probes"] if "check" not in p]
    if missing:
        raise ValueError(f"probes without a check (run checkers.py first): {missing}")
    by_id = {p["id"]: p for p in bench["probes"]}
    rosters = {label: {a["id"]: a for a in agents} for label, agents in bench["rosters"].items()}
    roster_of = lambda probe: rosters.get(f"lived-{probe['size']}", {})  # noqa: E731
    chosen: dict[str, dict[str, Any]] = {}
    for item in decisions:
        probe_id = item.get("id")
        if probe_id not in by_id:
            raise ValueError(f"unknown probe id {probe_id!r}")
        if not _flagged(by_id[probe_id]):
            raise ValueError(f"probe {probe_id} was not flagged for review")
        if probe_id in chosen:
            raise ValueError(f"probe {probe_id} decided twice")
        _check_decision(item, by_id[probe_id], roster_of(by_id[probe_id]))
        chosen[probe_id] = item
    undecided = [p["id"] for p in bench["probes"] if _flagged(p) and p["id"] not in chosen]
    if undecided:
        raise ValueError(f"undecided: {', '.join(undecided)}")

    kept: list[dict[str, Any]] = []
    tally = {"accept": 0, "fixed": 0, "reject": 0}
    audit = audit_changed = auto = 0
    for source in bench["probes"]:
        probe = copy.deepcopy(source)
        if not _flagged(probe):
            auto += 1
            kept.append({**probe, "review": "auto"})
            continue
        item = chosen[probe["id"]]
        decision = item["decision"]
        tally[decision] += 1
        if probe["check"]["flag"] == "audit":
            audit += 1
            audit_changed += decision != "accept"
        if decision == "reject":
            continue
        review: dict[str, Any] = {"decision": decision, "note": item.get("note", "")}
        if decision == "fixed":
            review["original"] = {k: probe[k] for k in ("gold_targets", "acceptable_ids", "clarify_ok")}
            _apply_fix(probe, item, roster_of(probe))
        kept.append({**probe, "review": review})
    summary = {
        "kept": len(kept), "auto_kept": auto, "accepted": tally["accept"], "fixed": tally["fixed"],
        "rejected": tally["reject"], "audit": audit, "audit_changed": audit_changed,
        "label_error_rate": audit_changed / audit if audit else None,
        "decisions_sha256": decisions_sha256 or _canonical_sha256(decisions),
    }
    return {"meta": {**copy.deepcopy(bench.get("meta", {})), "review": summary},
            "rosters": copy.deepcopy(bench["rosters"]), "probes": kept}


# --- loading for the runner ---------------------------------------------------------------------

CASE_FIELDS = frozenset(f.name for f in fields(Case))


def load_v2(seed: int, *, directory: Path | None = None) -> list[tuple[str, World, list[Case]]]:
    """(label, world, cases) per size of life `seed`, smallest first, for evals.run.

    Rosters and now come from world-s{seed}.json, as store.load_lived builds them; the cases are the
    reviewed bench's probes (Case fields only), each under every label it carries. A whole life is one
    split: life 1 dev, any other test."""
    reviewed = bench_path(seed, reviewed=True, directory=directory)
    if not reviewed.exists():
        raise FileNotFoundError(f"benchmark v2 life {seed}: no reviewed probe file {reviewed} (run finalize first)")
    path = world_path(seed, directory)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != WORLD_FORMAT or data.get("seed") != seed:
        raise ValueError(f"{path}: wrong format or seed for life {seed}")
    split = "dev" if seed == DEV_LIFE else "test"
    by_label: dict[str, list[Case]] = {}
    for probe in json.loads(reviewed.read_text(encoding="utf-8"))["probes"]:
        case = Case(**{
            **{k: v for k, v in probe.items() if k in CASE_FIELDS},
            "context": [tuple(c) for c in probe.get("context", [])], "split_override": split,
        })
        for label in probe["labels"]:
            by_label.setdefault(label, []).append(case)
    units = []
    for entry in sorted(data["sizes"], key=lambda e: e["size"]):
        world = World(seed=seed, agents=[_agent(a) for a in entry["agents"]], now=day_to_datetime(entry["day"]))
        units.append((entry["label"], world, by_label.pop(entry["label"], [])))
    if by_label:
        raise ValueError(f"{reviewed}: probes for sizes {path.name} does not have: {sorted(by_label)}")
    return units


# --- CLI ----------------------------------------------------------------------------------------


def _review_data_cli(seeds: list[int], bench_dir: Path, out: Path) -> int:
    seeds = sorted(seeds)
    raw = {seed: bench_path(seed, directory=bench_dir).read_bytes() for seed in seeds}
    benches = {seed: json.loads(raw[seed].decode("utf-8")) for seed in seeds}
    data = review_data(benches, hashlib.sha256(b"".join(raw[s] for s in seeds)).hexdigest())
    for seed in seeds:
        flags = [p["check"]["flag"] for p in data["probes"] if p["world_seed"] == seed]
        print(f"life {seed}: {len(flags)} flagged  disagree={flags.count('disagree')} audit={flags.count('audit')}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(data), encoding="utf-8")
    print(f"{len(data['probes'])} probes for review  wrote {out}")
    return 0


def _finalize_cli(seeds: list[int], decisions_file: Path, bench_dir: Path) -> int:
    raw = decisions_file.read_bytes()
    per_life = split_decisions(json.loads(raw.decode("utf-8")), seeds)
    digest = hashlib.sha256(raw).hexdigest()
    bodies = {seed: finalize(json.loads(bench_path(seed, directory=bench_dir).read_text(encoding="utf-8")),
                             per_life[seed], decisions_sha256=digest) for seed in seeds}
    for seed, body in bodies.items():  # all lives validated before anything is written
        out = bench_path(seed, reviewed=True, directory=bench_dir)
        out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        r = body["meta"]["review"]
        rate = "n/a" if r["label_error_rate"] is None else f"{r['label_error_rate']:.3f}"
        print(f"life {seed}: kept {r['kept']} ({r['auto_kept']} auto, {r['accepted']} accepted, {r['fixed']} fixed), "
              f"rejected {r['rejected']}; audit {r['audit_changed']}/{r['audit']} changed, label error rate {rate}  "
              f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the bench-v2 probe files")
    build.add_argument("--seed", type=int, action="append", required=True, help="once or twice")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--budget", type=float, default=2.0)
    build.add_argument("--fake-llm", action="store_true", help="offline scripted writer; no API calls")
    build.add_argument("--world-dir", type=Path, default=HERE, help="where world-s{seed}.json live")
    build.add_argument("--out-dir", type=Path, help="default: next to the world files")
    rule = sub.add_parser("majority", help="re-flag checked benches by 2-of-3 checker agreement, in place")
    rule.add_argument("--seed", type=int, action="append", required=True)
    rule.add_argument("--bench-dir", type=Path, default=HERE, help="where bench-v2-s{seed}.json live")
    page = sub.add_parser("review-data", help="write the review page of the flagged probes")
    page.add_argument("--seed", type=int, action="append", required=True)
    page.add_argument("--bench-dir", type=Path, default=HERE, help="where bench-v2-s{seed}.json live")
    page.add_argument("--out", type=Path, default=REVIEW_PAGE)
    fin = sub.add_parser("finalize", help="apply review decisions; writes bench-v2-s{seed}.reviewed.json")
    fin.add_argument("--seed", type=int, action="append", required=True)
    fin.add_argument("--decisions", type=Path, required=True, help="JSON list of decisions, both lives")
    fin.add_argument("--bench-dir", type=Path, default=HERE, help="where bench-v2-s{seed}.json live")
    args = parser.parse_args(argv)
    if args.command == "majority":
        for seed in args.seed:
            path = bench_path(seed, directory=args.bench_dir)
            body = majority(json.loads(path.read_text(encoding="utf-8")), seed=seed)
            path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            flags = [p["check"]["flag"] for p in body["probes"]]
            print(f"life {seed}: disagree={flags.count('disagree')} audit={flags.count('audit')}  wrote {path}")
        return 0
    if args.command == "review-data":
        return _review_data_cli(args.seed, args.bench_dir, args.out)
    if args.command == "finalize":
        return _finalize_cli(args.seed, args.decisions, args.bench_dir)
    if len(args.seed) not in (1, 2) or len(set(args.seed)) != len(args.seed):
        parser.error("give one --seed, or two different ones")
    out_dir = args.out_dir or args.world_dir

    if args.fake_llm:
        llm: Any = fake_writer()
    else:
        from ...llm import Budget, OpenRouterLLM
        from ...run import CACHE_DIR, _api_key  # lazy, as in round2

        key = _api_key()
        if not key:
            print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
            return 2
        llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(args.budget), concurrency=8)

    async def go() -> list[dict[str, Any]]:
        try:
            return [await build_life(seed, llm=llm, model=args.model, directory=args.world_dir, fake=args.fake_llm)
                    for seed in args.seed]
        finally:
            await llm.aclose()

    bodies = asyncio.run(go())
    if len(bodies) == 2:
        bodies[0]["probes"], bodies[1]["probes"] = balance(bodies[0]["probes"], bodies[1]["probes"])
        for body in bodies:
            body["meta"]["balanced_with"] = [s for s in args.seed if s != body["meta"]["seed"]][0]
    else:
        bodies[0]["probes"] = _renumber(bodies[0]["probes"])
        print(f"note: only life {args.seed[0]} built; this file is unbalanced (rebuild with both seeds)")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for body in bodies:
        out = bench_path(body["meta"]["seed"], directory=out_dir)
        out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"life {body['meta']['seed']}: {len(body['probes'])} probes  wrote {out}")
    spent = getattr(getattr(llm, "budget", None), "spent_usd", 0.0)
    print(f"spent ${spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
