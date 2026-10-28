"""Run an eval experiment.

    python -m evals.run --config evals/experiments/baseline.yaml            # dev split
    python -m evals.run --config evals/experiments/smoke.yaml --fake-llm    # offline
    python -m evals.run --config ... --split test --final                   # held-out, once per finalist

If the budget runs out, scheduling stops: the case that hit the cap and every case not yet
started are left out of routing_rows (they count as incomplete, not as errors), and
meta.budget_exhausted is true. A case that fails for any other reason is an ERROR row that
keeps the cost it ran up before failing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .cases import Case, filter_cases
from .grading import ERROR, Decision, duplicate_possible, grade
from .integrity import REPO_ROOT, compute_checksum, expected_checksum
from .layers.retrieval import run_retrieval_case
from .layers.routing import RoutingCaseError, run_routing_case
from .leaderboard import write_leaderboard
from .llm import Budget, BudgetExceeded, LLMClient, OpenRouterLLM, fake_llm
from .metrics import summarize_retrieval, summarize_routing
from .systems import get_system
from .worlds.generator import build_world, generate_cases
from .worlds.handwritten import load_handwritten
from .worlds.model import World

DEFAULT_OUT = REPO_ROOT / "evals" / "results"
CACHE_DIR = REPO_ROOT / "evals" / ".cache" / "llm"


@dataclass
class Config:
    name: str
    system: str = "baseline"
    model: str = "anthropic/claude-sonnet-4"
    layers: list[str] = field(default_factory=lambda: ["retrieval", "routing"])
    n_agents: list[int] = field(default_factory=lambda: [10, 50, 200, 1000])
    world_seeds: list[int] = field(default_factory=lambda: [1])
    k: int = 3
    cases_per_kind: int = 3
    kinds: list[str] | None = None
    suite: str = "all"
    budget_usd: float = 15.0
    concurrency: int = 8
    include_handwritten: bool = True


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)


Unit = tuple[Any, World, list[Case]]  # (roster-size label, world, cases)


def _build_units(cfg: Config, split: str) -> list[Unit]:
    units: list[Unit] = []
    for seed in cfg.world_seeds:
        full = build_world(seed)
        cases = filter_cases(generate_cases(full, cfg.cases_per_kind), split=split, suite=cfg.suite, kinds=cfg.kinds)
        for n in cfg.n_agents:
            units.append((n, full.truncated(n), cases))
    if cfg.include_handwritten:
        world, cases = load_handwritten()
        units.append(("handwritten", world, filter_cases(cases, split=split, suite=cfg.suite, kinds=cfg.kinds)))
    return units


def _routing_row(
    label: Any, world: World, case: Case, sample: int, d: Decision, outcome: str, system: str, error: str | None = None
) -> dict[str, Any]:
    return {
        "system": system, "n_agents": label, "world_seed": world.seed, "case_id": case.id, "kind": case.kind,
        "sample": sample, "message": case.message, "gold": case.gold_targets, "acceptable": case.acceptable_ids,
        "expects_reuse": case.expects_reuse, "duplicate_possible": duplicate_possible(case), "outcome": outcome,
        "error": error, "dispatched_ids": d.dispatched_ids, "created_names": d.created_names, "clarified": d.clarified,
        "user_messages": d.user_messages, "tool_calls": d.tool_calls, "iterations": d.iterations,
        "hit_iteration_cap": d.hit_iteration_cap, "cost": d.cost, "fresh_cost": d.fresh_cost,
        "cached_calls": d.cached_calls, "latency_s": d.latency_s, "prompt_tokens": d.prompt_tokens,
        "completion_tokens": d.completion_tokens, "cache_keys": d.cache_keys,
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def _run_routing(
    cfg: Config, units: list[Unit], system: Any, llm: LLMClient
) -> tuple[list[dict[str, Any]], bool]:
    """Run every (case, sample) with cfg.concurrency workers; stop scheduling once the budget is hit."""
    jobs = [(label, world, case, s) for label, world, cases in units for case in cases for s in range(cfg.k)]
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    pending = iter(enumerate(jobs))
    exhausted = False

    async def worker() -> None:
        nonlocal exhausted
        for index, (label, world, case, sample) in pending:  # shared iterator: each job runs once
            if exhausted:
                return
            try:
                decision = await run_routing_case(case, world, system, llm, model=cfg.model, sample=sample)
            except BudgetExceeded:
                exhausted = True  # this in-flight case is dropped, not recorded as an error
                return
            except RoutingCaseError as exc:
                results[index] = _routing_row(label, world, case, sample, exc.decision, ERROR, system.name, str(exc))
                continue
            try:
                outcome, error = grade(case, decision), None
            except Exception as exc:  # recorded as a failing sample, never silently dropped
                outcome, error = ERROR, f"grading failed: {type(exc).__name__}: {exc}"
            results[index] = _routing_row(label, world, case, sample, decision, outcome, system.name, error)

    await asyncio.gather(*(worker() for _ in range(max(1, cfg.concurrency))))
    return [row for row in results if row is not None], exhausted


async def run_experiment(cfg: Config, *, llm: LLMClient, out_dir: Path, split: str, final: bool) -> Path:
    started = datetime.now(timezone.utc)
    checksum, expected = compute_checksum(), expected_checksum()
    system = get_system(cfg.system)
    units = _build_units(cfg, split)

    retrieval_rows: list[dict[str, Any]] = []
    if "retrieval" in cfg.layers:
        for label, world, cases in units:
            for case in cases:
                retrieval_rows.append({
                    "n_agents": label, "world_seed": world.seed, "case_id": case.id, "kind": case.kind,
                    **run_retrieval_case(case, world, system),
                })

    routing_rows: list[dict[str, Any]] = []
    budget_exhausted = False
    planned: dict[str, set[tuple]] = {}
    if "routing" in cfg.layers:
        for label, world, cases in units:
            planned.setdefault(str(label), set()).update((world.seed, case.id) for case in cases)
        routing_rows, budget_exhausted = await _run_routing(cfg, units, system, llm)

    budget = getattr(llm, "budget", None)
    spent = budget.spent_usd if budget is not None else None
    fresh = sum(row["fresh_cost"] for row in routing_rows)
    result = {
        "meta": {
            **asdict(cfg),
            "split": split,
            "final": final,
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "expected_checksum": expected,
            "valid": expected is not None and checksum == expected,
            # spent: everything the client charged; fresh: what finished rows account for;
            # unattributed: the rest (cases dropped when the budget ran out).
            "budget_spent_usd": None if spent is None else round(spent, 4),
            "fresh_cost_usd": round(fresh, 4),
            "unattributed_cost_usd": None if spent is None else round(spent - fresh, 4),
            "budget_exhausted": budget_exhausted,
        },
        "summary": {
            "retrieval": summarize_retrieval(retrieval_rows),
            "routing": summarize_routing(routing_rows, planned=planned, k=cfg.k),
        },
        "retrieval_rows": retrieval_rows,
        "routing_rows": routing_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{started:%Y%m%d-%H%M%S}-{cfg.name}-{split}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_leaderboard(out_dir)
    return path


def _api_key() -> str | None:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    from server.config import get_settings  # loads the repo-root .env

    return get_settings().openrouter_api_key


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required to run the held-out test split")
    parser.add_argument("--fake-llm", action="store_true", help="offline scripted model; no API calls")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.split == "test" and not args.final:
        print("Refusing to run the test split without --final. Tune on dev; run test once per finalist.", file=sys.stderr)
        return 2
    cfg = load_config(args.config)

    llm: LLMClient
    if args.fake_llm:
        llm = fake_llm()
    else:
        key = _api_key()
        if not key:
            print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
            return 2
        llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(cfg.budget_usd), concurrency=cfg.concurrency)

    async def go() -> Path:
        try:
            return await run_experiment(cfg, llm=llm, out_dir=args.out_dir, split=args.split, final=args.final)
        finally:
            await llm.aclose()

    path = asyncio.run(go())
    data = json.loads(path.read_text(encoding="utf-8"))
    for size, g in data["summary"]["routing"].items():
        print(
            f"n={size:>11}  pass@1={_fmt(g['pass_at_1'])}  macro={_fmt(g['macro_pass_at_1'])}  "
            f"pass^k={_fmt(g['pass_hat_k'])}  dup={_fmt(g['duplicate_rate'])}  errors={g['error_count']}  "
            f"hit_cap={_fmt(g['hit_cap_rate'])}  incomplete={g['incomplete_cases']}"
        )
    meta = data["meta"]
    print(
        f"spent {_usd(meta['budget_spent_usd'])}  fresh {_usd(meta['fresh_cost_usd'])}  "
        f"unattributed {_usd(meta['unattributed_cost_usd'])}  budget_exhausted={meta['budget_exhausted']}  "
        f"valid={meta['valid']}  wrote {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
