"""Recover each grown agent's log (its instructions and its helper reports) by replaying growth from the cache, at $0.

    python -m evals.worlds.lived.replay --seed 1     # writes agent-logs-s1.json

Growth's helper reports ("[SUCCESS] {agent}: {report}") went only into the transcript, so the world file keeps each
agent's instructions with empty results. Every growth call is cached, so generate() is rerun on the world's own model
(meta.model) with a zero budget: a cache miss raises BudgetExceeded, and no request can be sent. The regenerated world
must equal the committed one apart from meta.generated_at, or nothing is written.

Probes are not growth and are not replayed (per_kind=0): life 1's were drafted by an older probes.py whose prompts are
no longer cached, so each size's `cases` are left out of the comparison. Entries saved before the world file kept its
probe state (life 1) have that state checked instead against the probes' shared context and the plan
(round2.rebuild_snapshot). Each agent's logged instructions must also equal its saved history at every size.

Output: {format, seed, world_sha256, cutoffs: {label: event index}, logs: {agent id: [[event index, kind, text], ...]}},
kind "instruction" or "result". A size's agents saw their log entries up to its cutoff (the 1,000 size shares the 200
snapshot's). Background agents have no log. Life 2 is sealed: this prints counts only.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import httpx

from .bench_v2 import HERE, world_path
from .generate import generate
from .grow import GrowthResult
from .model import LifePlan, Snapshot
from .plan import build_plan
from .round2 import rebuild_snapshot

FORMAT = 1
PROBE_STATE = ("transcript", "thread_last_day", "last_user_threads")


class ReplayMismatch(RuntimeError):
    """The replay did not reproduce the committed world, or its logs disagree with the saved histories."""


def logs_path(seed: int, directory: Path | None = None) -> Path:
    return Path(directory or HERE) / f"agent-logs-s{seed}.json"


def _world(seed: int, directory: Path | None) -> dict[str, Any]:
    return json.loads(world_path(seed, directory).read_text(encoding="utf-8"))


def _state_differences(committed: dict[str, Any], new: dict[str, Any], plan: LifePlan, skipped: int) -> list[str]:
    """The probe-state keys of `new` that disagree with the snapshot rebuilt from a legacy `committed` entry."""
    label = committed["label"]
    try:
        rebuilt = rebuild_snapshot(committed, plan, skipped=skipped)
    except ValueError as exc:
        return [f"{label}: cannot rebuild the snapshot ({exc})"]
    expected = {
        "transcript": [list(entry) for entry in rebuilt.transcript],
        "thread_last_day": dict(rebuilt.thread_last_day),
        "last_user_threads": list(rebuilt.last_user_threads),
    }
    return [f"{label}.{key}" for key in PROBE_STATE if new.get(key) != expected[key]]


def differences(committed: dict[str, Any], regenerated: dict[str, Any], plan: LifePlan | None) -> list[str]:
    """Where the regenerated world differs from the committed one (paths only, no content); empty when equal."""
    old, new = copy.deepcopy(committed), copy.deepcopy(regenerated)
    for body in (old, new):
        body.get("meta", {}).pop("generated_at", None)
    out = [key for key in sorted(set(old) | set(new)) if key != "sizes" and old.get(key) != new.get(key)]
    if len(old["sizes"]) != len(new["sizes"]):
        return out + ["sizes"]
    for old_entry, new_entry in zip(old["sizes"], new["sizes"]):
        label = old_entry.get("label")
        if "transcript" not in old_entry and "transcript" in new_entry:  # saved before the probe state was
            out += _state_differences(old_entry, new_entry, plan, committed["meta"].get("skipped", 0))
            for key in PROBE_STATE:
                new_entry.pop(key, None)
        old_entry.pop("cases", None)
        new_entry.pop("cases", None)
        out += [f"{label}.{key}" for key in sorted(set(old_entry) | set(new_entry))
                if old_entry.get(key) != new_entry.get(key)]
    return out


def _upto(entries: list[list[Any]], cutoff: int) -> list[list[Any]]:
    return [entry for entry in entries if entry[0] <= cutoff]


def history_mismatches(world: dict[str, Any], body: dict[str, Any]) -> list[str]:
    """`label: agent id` for every logged agent whose instructions up to the size's cutoff are not its history."""
    out = []
    for entry in world["sizes"]:
        cutoff = body["cutoffs"][entry["label"]]
        for agent in entry["agents"]:
            entries = body["logs"].get(agent["id"])
            if entries is None:
                continue
            logged = [text for _, kind, text in _upto(entries, cutoff) if kind == "instruction"]
            if logged != [request for request, _ in agent["history"]]:
                out.append(f"{entry['label']}: {agent['id']}")
    return out


def counts(world: dict[str, Any], body: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per size: agents, agents with a log, result entries they saw, and agents that saw at least one result."""
    out = {}
    for entry in world["sizes"]:
        cutoff = body["cutoffs"][entry["label"]]
        logs = [_upto(body["logs"][a["id"]], cutoff) for a in entry["agents"] if a["id"] in body["logs"]]
        results = [sum(1 for e in entries if e[1] == "result") for entries in logs]
        out[entry["label"]] = {"agents": len(entry["agents"]), "with_log": len(logs), "results": sum(results),
                               "with_result": sum(1 for n in results if n)}
    return out


async def replay(seed: int, *, llm, directory: Path | None = None, log=print) -> dict[str, Any]:
    """The agent-logs body for life `seed`. Raises ReplayMismatch; RunAborted (a cache miss) propagates."""
    path = world_path(seed, directory)
    raw = path.read_bytes()
    committed = json.loads(raw.decode("utf-8"))
    grown: dict[str, Any] = {}

    def keep(growth: GrowthResult, snapshots: list[Snapshot]) -> None:
        grown.update(logs=growth.logs, cutoffs={f"lived-{s.size}": s.event_index for s in snapshots})

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / path.name
        await generate(seed=seed, model=committed["meta"]["model"], llm=llm, per_kind=0, out=out, log=log,
                       on_grown=keep)
        regenerated = json.loads(out.read_text(encoding="utf-8"))
    diffs = differences(committed, regenerated, build_plan(seed) if any(
        "transcript" not in entry for entry in committed["sizes"]) else None)
    if diffs:
        raise ReplayMismatch(f"life {seed}: the replay differs from {path.name} at {', '.join(diffs[:20])}")
    body = {
        "format": FORMAT, "seed": seed, "world_sha256": hashlib.sha256(raw).hexdigest(), "cutoffs": grown["cutoffs"],
        "logs": {agent_id: [list(entry) for entry in entries] for agent_id, entries in grown["logs"].items()},
    }
    bad = history_mismatches(committed, body)
    if bad:
        raise ReplayMismatch(f"life {seed}: logged instructions differ from the saved history for {', '.join(bad[:20])}")
    return body


def _refuse(request: httpx.Request) -> httpx.Response:
    raise RuntimeError("the replay must not send requests")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    from ...llm import Budget, OpenRouterLLM
    from ...run import CACHE_DIR

    # No key and a transport that refuses: with Budget(0) a cache miss raises before any request anyway.
    llm = OpenRouterLLM(api_key="", cache_dir=CACHE_DIR, budget=Budget(0), transport=httpx.MockTransport(_refuse))

    async def go() -> dict[str, Any]:
        try:
            return await replay(args.seed, llm=llm, directory=args.dir, log=lambda line: None)
        finally:
            await llm.aclose()

    body = asyncio.run(go())
    out = logs_path(args.seed, args.dir)
    out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"life {args.seed}: world reproduced (apart from generated_at); {len(body['logs'])} agents with logs, "
          f"{sum(1 for e in body['logs'].values() for x in e if x[1] == 'result')} result entries; "
          f"spent ${llm.budget.spent_usd:.4f}  wrote {out}")
    for label, row in counts(_world(args.seed, args.dir), body).items():
        print(f"  {label}: {row['with_log']}/{row['agents']} agents with a log, {row['results']} results seen, "
              f"{row['with_result']} agents with at least one result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
