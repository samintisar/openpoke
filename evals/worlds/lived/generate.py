"""Generate the frozen lived-in world (run once; every LLM call is cached).

    python -m evals.worlds.lived.generate --seed 1 --model anthropic/claude-haiku-5.5 --budget 6
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from ...llm import Budget, OpenRouterLLM
from ...run import CACHE_DIR, _api_key
from .grow import grow, merge_background, name_background
from .plan import build_plan
from .probes import build_probes
from .store import DEFAULT_PATH, save

SIZES = (10, 50, 200)
FULL_SIZE = 1000


async def generate(*, seed: int, model: str, llm, per_kind: int, out: Path, log=print) -> Path:
    plan = build_plan(seed)
    log(f"plan: {len(plan.contacts)} contacts, {len(plan.threads)} threads, {len(plan.events)} events")
    growth = await grow(plan, llm=llm, model=model, sizes=SIZES, max_agents=max(SIZES), log=log)
    log(f"grown: {len(growth.agents)} agents, {len(growth.skipped)} events skipped")
    background = await name_background(plan, llm=llm, model=model)
    log(f"background: {len(background)} agents named")
    snapshots = list(growth.snapshots)
    snapshots.append(merge_background(snapshots[-1], background, target=FULL_SIZE))

    units = []
    probes_200 = None
    for snapshot in snapshots:
        if snapshot.size == FULL_SIZE:
            cases = probes_200  # the full roster reuses the 200 snapshot's moment and probes
        else:
            cases = await build_probes(snapshot, plan, llm=llm, model=model, per_kind=per_kind, rng_seed=seed * 1000 + snapshot.size)
            if snapshot.size == max(SIZES):
                probes_200 = cases
        units.append((snapshot, cases))
        log(f"size {snapshot.size}: {len(snapshot.agents)} agents, {len(cases)} probes")

    meta = {
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events_played": len(plan.events),
        "skipped": len(growth.skipped),
        "background_named": len(background),
    }
    save(out, seed=seed, units=units, meta=meta)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="anthropic/claude-haiku-5.5")
    parser.add_argument("--budget", type=float, default=6.0)
    parser.add_argument("--per-kind", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    key = _api_key()
    if not key:
        print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
        return 2
    llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(args.budget), concurrency=8)

    async def go() -> Path:
        try:
            return await generate(seed=args.seed, model=args.model, llm=llm, per_kind=args.per_kind, out=args.out)
        finally:
            await llm.aclose()

    out = asyncio.run(go())
    print(f"spent ${llm.budget.spent_usd:.4f}  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
