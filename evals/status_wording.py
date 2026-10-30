"""Status-rule wording test (Plan 6d): does routing of progress questions survive rewording?

    python -m evals.status_wording                 # original, v1 and final on life 1 (dev) at 1,000 agents
    python -m evals.status_wording --fake-llm      # offline

The status rule's wording was revised once after reading dev failures (Plan 6c step 2b), so its dev score on
the benchmark's own phrasings may flatter it. Here every dev status question at 1,000 agents is asked ten more
ways, from TEMPLATES, written before any of these runs and never changed after. Each reworded case keeps the
original's answer key and conversation context, so a correct answer is the same as for the original: dispatch to
the matter's agent, or, when no agent fits (three of the ten), ask which matter is meant without creating one.

The slots are filled by hand from each original message (SLOTS): `what` names the matter as the user did, `who`
is the person waiting on it, or None. One sample per (case, template, system), graded by the benchmark's grader.

Found after the first run, and stated as such: in cases 00-02 the original names the person ("did Leo get back to
me about the renewal yet?") and the roster has other agents on the same matter (8 renewals, 2 bachelor parties, 2
conference talks), with no name in the context. The nine templates that drop the person make those messages
ambiguous, so the original answer key no longer fits them. The report scores them apart (AMBIGUOUS): asking which
matter is right there, and dispatching to any one agent is a guess. The templates and slots were not changed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import Case
from .llm import Budget, LLMClient, OpenRouterLLM
from .run import CACHE_DIR, DEFAULT_OUT, _api_key, _run_routing, fake_llm, load_config
from .systems import get_system
from .worlds.lived import bench_v2

SIZE = "lived-1000"
EXPERIMENTS = Path(__file__).resolve().parent / "experiments"
SYSTEMS = {"original": "v2-baseline.yaml", "v1": "v2-v1.yaml", "final": "v2-full.yaml"}

TEMPLATES = (
    "any news on {what}?",
    "where are we with {what}?",
    "{what} - what's the latest?",
    "did we ever hear back on {what}",
    "is {what} sorted yet?",
    "remind me where {what} stands",
    "quick check: has anything moved on {what}?",
    "status on {what} pls",
    "what happened with {what} in the end?",
    "has {who} replied about {what}?",  # "anyone" when no one is named
)

AMBIGUOUS = {"00", "01", "02"}  # cases whose matter is shared by several agents unless the person is named
PERSON_TEMPLATE = len(TEMPLATES) - 1

SLOTS = {  # case id suffix -> (what, who), from the original message
    "00": ("the renewal", "Leo"),
    "01": ("the bachelor party", "Sara"),
    "02": ("the conference talk proposal", "Zoe"),
    "03": ("the parking", None),
    "04": ("the driving test booking", None),
    "05": ("the Austin flight", None),
    "06": ("the procurement paperwork", "Quinn"),
    "07": ("the Boston flight search", None),
    "08": ("the Athens flight", None),
    "09": ("the uniform order for Mia", None),
}


def reword(case: Case) -> list[Case]:
    what, who = SLOTS[case.id.rsplit("-", 1)[1]]
    return [replace(case, id=f"{case.id}-w{i}", message=template.format(what=what, who=who or "anyone"))
            for i, template in enumerate(TEMPLATES)]


def units() -> list[tuple[str, Any, list[Case]]]:
    for label, world, cases in bench_v2.load_v2(bench_v2.DEV_LIFE):
        if label == SIZE:
            status = [c for c in cases if c.kind == "status_question"]
            if len(status) != len(SLOTS):
                raise ValueError(f"expected {len(SLOTS)} status questions at {SIZE}, found {len(status)}")
            return [(label, world, [r for c in status for r in reword(c)])]
    raise ValueError(f"no {SIZE} roster in life {bench_v2.DEV_LIFE}")


async def run(llm: LLMClient) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, config in SYSTEMS.items():
        cfg = replace(load_config(EXPERIMENTS / config), k=1, n_agents=[1000])
        rows, aborted = await _run_routing(cfg, units(), get_system(cfg.system, cfg.system_options), llm)
        if aborted is not None:
            raise aborted
        out[name] = rows
    return out


def ambiguous(row: dict[str, Any]) -> bool:
    case, template = row["case_id"].rsplit("-w", 1)
    return case.rsplit("-", 1)[1] in AMBIGUOUS and int(template) != PERSON_TEMPLATE


def report(rows_by_system: dict[str, list[dict[str, Any]]]) -> str:
    def rate(rows: list[dict[str, Any]]) -> str:
        return f"{sum(r['outcome'] == 'correct' for r in rows) / len(rows):.2f}" if rows else "-"

    names = list(rows_by_system)
    lines = [f"{'':46} " + " ".join(f"{n:>8}" for n in names)]
    lines.append(f"{'all (' + str(len(rows_by_system[names[0]])) + ' messages)':46} "
                 + " ".join(f"{rate(rows_by_system[n]):>8}" for n in names))
    for i, template in enumerate(TEMPLATES):
        pick = lambda rows: [r for r in rows if r["case_id"].endswith(f"-w{i}")]
        lines.append(f"  {template:44} " + " ".join(f"{rate(pick(rows_by_system[n])):>8}" for n in names))
    for label, test in (("matter has an agent", bool), ("no agent fits", lambda g: not g)):
        pick = lambda rows: [r for r in rows if test([g for g in r["gold"] if g != "NEW"])]
        lines.append(f"  {label:44} " + " ".join(f"{rate(pick(rows_by_system[n])):>8}" for n in names))
    lines.append("scored apart (see AMBIGUOUS):")
    for n in names:
        clear = [r for r in rows_by_system[n] if not ambiguous(r)]
        amb = [r for r in rows_by_system[n] if ambiguous(r)]
        asked = sum(1 for r in amb if r["clarified"] and not r["dispatched_ids"] and not r["created_names"])
        guessed = sum(1 for r in amb if r["dispatched_ids"])
        created = sum(1 for r in amb if r["created_names"])
        lines.append(f"  {n:8} unambiguous {rate(clear)} of {len(clear)}; ambiguous {len(amb)}: asked {asked}, "
                     f"dispatched to one agent {guessed}, created {created}")
    lines.append("outcomes:")
    for n in names:
        counts = Counter(r["outcome"] for r in rows_by_system[n])
        created = sum(1 for r in rows_by_system[n] if r.get("created_names"))
        lines.append(f"  {n:8} {dict(counts.most_common())}  turns that created an agent: {created}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--fake-llm", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.fake_llm:
        llm: LLMClient = fake_llm()
    else:
        key = _api_key()
        if not key:
            print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
            return 2
        llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(args.budget), concurrency=4)

    async def go() -> dict[str, Any]:
        try:
            return await run(llm)
        finally:
            await llm.aclose()

    rows = asyncio.run(go())
    print(report(rows))
    if not args.fake_llm:
        spent = llm.budget.spent_usd
        out = args.out_dir / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-status-wording-dev.json"
        out.write_text(json.dumps({"size": SIZE, "templates": TEMPLATES, "slots": SLOTS, "spent_usd": round(spent, 4),
                                   "rows": rows}, indent=1) + "\n", encoding="utf-8")
        print(f"spent ${spent:.4f}  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
