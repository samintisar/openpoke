"""Run the memory eval: fact recall from the full history vs the compacted one.

    python -m evals.memory_run --config evals/experiments/memory.yaml              # live, cached
    python -m evals.memory_run --config evals/experiments/memory.yaml --fake-llm   # offline

Writes evals/results/<ts>-<name>-dev.json with the same meta, validity and integrity fields as `evals.run`. If the
budget runs out or OpenRouter refuses the account, scheduling stops: agents not finished are left out of the rows
(counted as incomplete), meta.aborted names the reason and the CLI exits 1; a refused account also makes the run
invalid. An agent that fails for any other reason is an error row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .integrity import compute_checksum, expected_checksum, expected_file_committed
from .layers.memory import CONDITIONS, fake_memory_llm, run_memory_agent, summarize_memory
from .llm import (
    Budget,
    BudgetExceeded,
    DailyQuotaExhausted,
    FatalAPIError,
    LLMClient,
    OpenRouterLLM,
    RateLimiter,
    RunAborted,
)
from .run import CACHE_DIR, DEFAULT_OUT, _api_key, _fmt, _git, _is_int, _usd
from .worlds.memory_gen import MIN_RUNS, AgentHistory, build_histories


@dataclass
class MemoryConfig:
    name: str = "memory"
    model: str = "anthropic/claude-haiku-5.5"
    seed: int = 1
    n_agents: int = 12
    runs: int = 40
    budget_tokens: int = 4000
    budget_usd: float = 1.0
    concurrency: int = 4
    requests_per_minute: int | None = None
    max_retries: int = 3


def load_config(path: Path) -> MemoryConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(MemoryConfig)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = MemoryConfig(**raw)
    for key in ("seed", "n_agents", "runs", "budget_tokens", "concurrency"):
        value = getattr(cfg, key)
        if not _is_int(value) or (value <= 0 and key != "seed"):
            raise ValueError(f"config key {key!r}: must be a positive int")
    if cfg.runs < MIN_RUNS:
        raise ValueError(f"config key 'runs': must be at least {MIN_RUNS}")
    if not _is_int(cfg.max_retries) or cfg.max_retries < 0:
        raise ValueError("config key 'max_retries': must be a non-negative int")
    rpm = cfg.requests_per_minute
    if rpm is not None and (not _is_int(rpm) or rpm <= 0):
        raise ValueError("config key 'requests_per_minute': must be null or a positive int")
    b = cfg.budget_usd
    if isinstance(b, bool) or not isinstance(b, (int, float)) or b <= 0:
        raise ValueError("config key 'budget_usd': must be a positive number")
    return cfg


async def _run_agents(
    cfg: MemoryConfig, agents: list[AgentHistory], llm: LLMClient
) -> tuple[list[dict[str, Any]], RunAborted | None]:
    """Run every agent with cfg.concurrency workers; stop scheduling once the run is aborted."""
    results: list[dict[str, Any] | None] = [None] * len(agents)
    pending = iter(enumerate(agents))
    aborted: RunAborted | None = None

    async def worker() -> None:
        nonlocal aborted
        for index, agent in pending:  # shared iterator: each agent runs once
            if aborted is not None:
                return
            try:
                results[index] = await run_memory_agent(agent, llm, model=cfg.model, budget_tokens=cfg.budget_tokens)
            except RunAborted as exc:
                aborted = aborted or exc  # this in-flight agent is dropped, not recorded as an error
                return
            except Exception as exc:  # never escape gather
                results[index] = {"agent_id": agent.agent_id, "name": agent.name, "error": f"{type(exc).__name__}: {exc}"}

    await asyncio.gather(*(worker() for _ in range(max(1, cfg.concurrency))))
    return [row for row in results if row is not None], aborted


async def run_memory_experiment(cfg: MemoryConfig, *, llm: LLMClient, out_dir: Path) -> Path:
    started = datetime.now(timezone.utc)
    checksum, expected = compute_checksum(), expected_checksum()
    agents = build_histories(cfg.seed, cfg.n_agents, cfg.runs)
    rows, aborted = await _run_agents(cfg, agents, llm)

    budget = getattr(llm, "budget", None)
    spent = budget.spent_usd if budget is not None else None
    summary = summarize_memory(rows, planned=len(agents))
    fresh = sum(
        row["compaction"]["fresh_cost"]
        + sum(
            a["cost"] for condition in CONDITIONS if not row["conditions"][condition].get("fallback")
            for a in row["conditions"][condition]["facts"] if not a["cached"]
        )
        for row in rows if row.get("error") is None
    )
    result = {
        "meta": {
            **asdict(cfg),
            "system": "memory",
            "split": "dev",
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "expected_checksum": expected,
            "valid": (
                expected is not None and checksum == expected and expected_file_committed()
                and not isinstance(aborted, FatalAPIError)
            ),
            "budget_spent_usd": None if spent is None else round(spent, 4),
            "fresh_cost_usd": round(fresh, 4),
            "unattributed_cost_usd": None if spent is None else round(spent - fresh, 4),
            "budget_exhausted": isinstance(aborted, BudgetExceeded),
            "aborted": None if aborted is None else f"{type(aborted).__name__}: {aborted}",
        },
        "summary": {"memory": summary},
        "agent_rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{started:%Y%m%d-%H%M%S}-{cfg.name}-dev.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fake-llm", action="store_true", help="offline scripted model; no API calls")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    llm: LLMClient
    if args.fake_llm:
        llm = fake_memory_llm()
    else:
        key = _api_key()
        if not key:
            print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
            return 2
        limiter = RateLimiter(cfg.requests_per_minute) if cfg.requests_per_minute else None
        llm = OpenRouterLLM(
            api_key=key, cache_dir=CACHE_DIR, budget=Budget(cfg.budget_usd), concurrency=cfg.concurrency,
            max_retries=cfg.max_retries, rate_limiter=limiter,
        )

    async def go() -> Path:
        try:
            return await run_memory_experiment(cfg, llm=llm, out_dir=args.out_dir)
        finally:
            await llm.aclose()

    path = asyncio.run(go())
    data = json.loads(path.read_text(encoding="utf-8"))
    summary, meta = data["summary"]["memory"], data["meta"]
    for condition in CONDITIONS:
        group = summary[condition]
        kinds = "  ".join(f"{kind}={_fmt(value)}" for kind, value in group["by_kind"].items())
        print(
            f"{condition:>9}  recall={_fmt(group['recall'])}  stale={_fmt(group['stale_rate'])}  "
            f"history_tokens={_fmt(group['mean_history_tokens'])}  prompt_tokens={_fmt(group['mean_prompt_tokens'])}  "
            f"{kinds}"
        )
    compaction = summary["compaction"]
    print(
        f"compaction calls={compaction['calls']}  failures={compaction['failures']}  skipped={compaction['skipped']}  "
        f"cost {_usd(compaction['cost_usd'])}  agents={summary['agents']}/{summary['planned_agents']}  "
        f"errors={summary['errors']}"
    )
    for row in data["agent_rows"]:
        if row.get("error") is None:
            print(
                f"  {row['agent_id']:<20} full={row['full_history_tokens']:>6}  "
                f"compacted={row['compacted_history_tokens']:>6}  prefix={row['prefix_tokens']:>6}  "
                f"compaction={row['compaction']['status']}"
            )
        else:
            print(f"  {row['agent_id']:<20} ERROR {row['error']}")
    print(
        f"spent {_usd(meta['budget_spent_usd'])}  fresh {_usd(meta['fresh_cost_usd'])}  "
        f"budget_exhausted={meta['budget_exhausted']}  valid={meta['valid']}  wrote {path}"
    )
    if meta.get("aborted"):
        print(f"RUN ABORTED: {meta['aborted']}", file=sys.stderr)
        if meta["aborted"].startswith(DailyQuotaExhausted.__name__):
            print("Finished calls are cached: rerun the same command after the quota resets to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
