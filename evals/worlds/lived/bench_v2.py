"""Benchmark v2 probe sets: per-size quotas, both lives balanced per kind, ambiguity marked.

    python -m evals.worlds.lived.bench_v2 build --seed 1 --seed 2 --budget 2   # bench-v2-s1/s2.json
    python -m evals.worlds.lived.bench_v2 build --seed 1 --fake-llm            # one life, offline, unbalanced

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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ...cases import NEW, Case
from ...llm import LLMClient
from .model import LifePlan, Snapshot
from .plan import build_plan, topic_words
from .probes import build_probes
from .round2 import _sha256, fake_writer, rebuild_snapshot

HERE = Path(__file__).resolve().parent
QUOTA = {10: 3, 50: 5, 200: 10}
FULL_SIZE = 1000  # reuses the 200 snapshot's probes
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
FORMAT = 1


def world_path(seed: int, directory: Path = HERE) -> Path:
    return Path(directory) / f"world-s{seed}.json"


def bench_path(seed: int, *, reviewed: bool = False, directory: Path = HERE) -> Path:
    return Path(directory) / f"bench-v2-s{seed}{'.reviewed' if reviewed else ''}.json"


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


# --- CLI ----------------------------------------------------------------------------------------


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
    args = parser.parse_args(argv)
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
