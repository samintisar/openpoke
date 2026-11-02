"""Living summaries for the lived worlds' agents, made by the production updater.

    python -m evals.worlds.lived.summaries --seed 1 --seed 2 --budget 1.5    # writes summaries-s{seed}.json

Each summary folds one agent's runs in order through production's `next_summary`, as production updates the card
after every run (a fold, or a rebuild from all runs so far every REBUILD_EVERY runs). A run is an instruction
and the helper reports that followed it before the next instruction (joined with a blank line; empty if none). The
runs come from agent-logs-s{seed}.json (replay.py), cut at each size's event index, when that file exists for the
same world; a report before any instruction is folded with empty instructions. Agents without a log (background
agents) and worlds without a logs file fold their saved history: instructions only, each with an empty result.

Not every instruction got a report during growth and background agents have none, while production sees a result
after every run, so eval summaries are a lower bound on production's; `agents_with_results` per size says how many
saw at least one report. A failed update keeps the previous summary, as in production.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from server.services.agents.summaries import SummaryError, next_summary

from ...llm import LLMClient, RunAborted
from .bench_v2 import DEFAULT_MODEL, HERE, world_path
from .replay import logs_path

FORMAT = 1


def summaries_path(seed: int, directory: Path | None = None) -> Path:
    return Path(directory or HERE) / f"summaries-s{seed}.json"


def llm_complete(llm: LLMClient, model: str):
    async def complete(system: str, user: str) -> str:
        response = await llm.complete(model=model, system=system, messages=[{"role": "user", "content": user}],
                                      tools=[], sample=0)
        return ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return complete


def runs_from_log(entries: list[list[Any]]) -> list[tuple[str, str]]:
    """(instructions, result) per run: each instruction with the results logged after it, before the next one."""
    runs: list[tuple[str, list[str]]] = []
    for _, kind, text in entries:
        if kind == "instruction":
            runs.append((text, []))
        else:
            if not runs:
                runs.append(("", []))
            runs[-1][1].append(text)
    return [(instructions, "\n\n".join(results)) for instructions, results in runs]


async def summarize_runs(agent: dict[str, Any], runs: list[tuple[str, str]], complete) -> str:
    summary = ""
    for i in range(len(runs)):
        try:
            summary = await next_summary(name=agent["name"], purpose=agent["purpose"], current=summary,
                                         runs=runs[: i + 1], complete=complete)
        except SummaryError as exc:
            if isinstance(exc.__cause__, RunAborted):
                raise exc.__cause__
    return summary


def _read_logs(seed: int, directory: Path | None, world_sha: str, world_name: str) -> tuple[dict[str, Any] | None, str | None]:
    path = logs_path(seed, directory)
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    body = json.loads(raw.decode("utf-8"))
    if body.get("world_sha256") != world_sha:
        raise ValueError(f"{path.name} was made for a different world than {world_name}; rerun the replay")
    return body, hashlib.sha256(raw).hexdigest()


async def build(seed: int, *, llm: LLMClient, model: str, directory: Path | None = None,
                concurrency: int = 8) -> dict[str, Any]:
    path = world_path(seed, directory)
    raw = path.read_bytes()
    world = json.loads(raw.decode("utf-8"))
    world_sha = hashlib.sha256(raw).hexdigest()
    logs, logs_sha = _read_logs(seed, directory, world_sha, path.name)
    complete = llm_complete(llm, model)
    gate = asyncio.Semaphore(concurrency)

    async def one(agent: dict[str, Any], runs: list[tuple[str, str]]) -> tuple[str, str]:
        async with gate:
            return agent["id"], await summarize_runs(agent, runs, complete)

    sizes: dict[str, dict[str, str]] = {}
    results_folded: dict[str, int] = {}
    agents_with_results: dict[str, int] = {}
    for entry in world["sizes"]:
        label = entry["label"]
        if logs is not None and label not in logs["cutoffs"]:
            raise ValueError(f"agent logs for life {seed} have no cutoff for {label}")
        jobs = []
        folded = with_results = 0
        for agent in entry["agents"]:
            entries = (logs or {}).get("logs", {}).get(agent["id"])
            if entries is None:
                runs = [(instructions, result) for instructions, result in agent.get("history", [])]
            else:
                kept = [e for e in entries if e[0] <= logs["cutoffs"][label]]
                runs = runs_from_log(kept)
                results = sum(1 for e in kept if e[1] == "result")
                folded += results
                with_results += bool(results)
            jobs.append(one(agent, runs))
        sizes[label] = dict(await asyncio.gather(*jobs))
        results_folded[label], agents_with_results[label] = folded, with_results
    empty = sum(1 for by_id in sizes.values() for text in by_id.values() if not text)
    return {"format": FORMAT, "seed": seed, "model": model, "world_sha256": world_sha,
            "source": "logs" if logs is not None else "history", "logs_sha256": logs_sha, "empty": empty,
            "results_folded": results_folded, "agents_with_results": agents_with_results, "sizes": sizes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget", type=float, default=1.5)
    parser.add_argument("--dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    from ...llm import Budget, OpenRouterLLM
    from ...run import CACHE_DIR, _api_key

    key = _api_key()
    if not key:
        print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
        return 2
    llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(args.budget), concurrency=8)

    async def go() -> None:
        try:
            for seed in args.seed:
                body = await build(seed, llm=llm, model=args.model, directory=args.dir)
                out = summaries_path(seed, args.dir)
                out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                count = sum(len(v) for v in body["sizes"].values())
                print(f"life {seed}: {count} agent summaries, {body['empty']} empty, source {body['source']}, "
                      f"agents with results {body['agents_with_results']}  wrote {out}")
        finally:
            await llm.aclose()
            print(f"spent ${llm.budget.spent_usd:.4f}")

    asyncio.run(go())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
