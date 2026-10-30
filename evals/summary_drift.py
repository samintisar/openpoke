"""Summary drift check (Plan 6d): do living summaries stay true to an agent's runs as runs are folded in?

    python -m evals.summary_drift                    # life 1 (dev) only; life 2's summaries stay unread
    python -m evals.summary_drift --view full        # judge the stored summary, not the clipped card view
    python -m evals.summary_drift --fake-llm         # offline, scripted judge and summarizer

Two parts, both on life 1 at 200 agents (the 1,000 size has the same logs):

- Real: each agent's committed summary (summaries-s1.json) is judged against its own runs, cut where the
  summaries were cut. Rates are reported by how many runs were folded (1, 2-3, 4+), so drift shows as a fall
  with more folds.
- Stress: the agents with the most runs get STRESS_FOLDS more runs of a common, low-content kind (a status
  check that finds no news) folded on top, through the production updater, then the same judge. This asks
  whether a long tail of empty updates pushes the matter itself out of the summary.

The judge (DeepSeek, one call per summary) answers four yes/no questions about the summary against the runs:
supported (no claim contradicts or goes beyond the runs), current (it states where things stand after the
last run), origin (it still names the matter and person of the first run) and actionable (a router could tell
a new message about this matter belongs here). The judge sees the summary as a card shows it, clipped to
SUMMARY_CHARS, since that is what routing reads. It is not calibrated against human labels, so a sample of its
"no" answers is printed for reading. Also counted for free: empty summaries and summaries longer than the card shows (SUMMARY_CHARS).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from server.services.agents.render import SUMMARY_CHARS, truncate  # the card shows summaries clipped here
from server.services.agents.summaries import SummaryError, update_summary

from .llm import Budget, LLMClient, OpenRouterLLM, RunAborted, ScriptedLLM
from .run import CACHE_DIR, DEFAULT_OUT, _api_key
from .worlds.lived.bench_v2 import DEFAULT_MODEL, DEV_LIFE, world_path
from .worlds.lived.replay import logs_path
from .worlds.lived.summaries import llm_complete, runs_from_log, summaries_path

SIZE = "lived-200"
STRESS_AGENTS = 8
STRESS_FOLDS = 20
QUESTIONS = ("supported", "current", "origin", "actionable")
STATUS_CHECK = (
    "The user is asking again whether there is any news on this. Check for anything new since your last report "
    "and report back briefly. Do not send anything."
)
NO_NEWS = "Nothing new since the last report: no reply and no new messages on this."

JUDGE_PROMPT = """You check a running summary that an assistant keeps for one of its helper agents. You get the
helper's runs in order (each run is the instruction it was given and what it reported back, which can be empty),
then the summary written after the last run. Answer four questions about the summary, each "yes" or "no":

supported: every claim in the summary is backed by the runs; nothing contradicts them or is made up.
current: the summary says where things stand after the LAST run (if the last run reported nothing, saying the
  last step is pending counts).
origin: the summary still names the matter and the person or thing the FIRST run was about.
actionable: someone routing a new user message could tell from the summary alone that a message about this
  matter belongs to this helper.

Reply with JSON only: {"supported": "yes|no", "current": "yes|no", "origin": "yes|no", "actionable": "yes|no",
"reason": "one sentence on any no"}"""


def _content(response: dict[str, Any]) -> str:
    return ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def parse_answers(text: str) -> dict[str, bool]:
    """The four answers from the judge's reply; raises ValueError rather than guess."""
    for match in re.finditer(r"\{.*?\}", text, re.S):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and all(q in data for q in QUESTIONS):
            answers = {q: str(data[q]).strip().lower() for q in QUESTIONS}
            if not set(answers.values()) <= {"yes", "no"}:
                raise ValueError(f"answers must be yes or no: {answers}")
            return {q: a == "yes" for q, a in answers.items()} | {"reason": str(data.get("reason") or "")}
    raise ValueError(f"no answers in reply: {text[:200]!r}")


def judge_input(name: str, runs: list[tuple[str, str]], summary: str) -> str:
    lines = [f"Helper: {name}", ""]
    for i, (instructions, result) in enumerate(runs, 1):
        lines += [f"Run {i} instruction: {instructions or '(none)'}", f"Run {i} report: {result or '(none)'}", ""]
    return "\n".join(lines + ["Summary after the last run:", summary or "(empty)"])


async def judge(llm: LLMClient, model: str, name: str, runs: list[tuple[str, str]], summary: str) -> dict[str, Any]:
    response = await llm.complete(model=model, system=JUDGE_PROMPT, tools=[], sample=0,
                                  messages=[{"role": "user", "content": judge_input(name, runs, summary)}])
    return parse_answers(_content(response))


def agent_runs(seed: int = DEV_LIFE, size: str = SIZE) -> list[tuple[dict[str, Any], list[tuple[str, str]]]]:
    """(agent, runs) for each agent at `size`, cut as summaries.py cuts them."""
    world = json.loads(world_path(seed).read_text(encoding="utf-8"))
    logs = json.loads(logs_path(seed).read_text(encoding="utf-8"))
    entry = next(e for e in world["sizes"] if e["label"] == size)
    out = []
    for agent in entry["agents"]:
        entries = logs["logs"].get(agent["id"])
        if entries is None:
            runs = [(i, r) for i, r in agent.get("history", [])]
        else:
            runs = runs_from_log([e for e in entries if e[0] <= logs["cutoffs"][size]])
        out.append((agent, runs))
    return out


def bucket(n: int) -> str:
    return "1 run" if n <= 1 else "2-3 runs" if n <= 3 else "4+ runs"


def rates(items: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [i for i in items if "answers" in i]
    return {"agents": len(items), "judged": len(judged),
            **{q: round(sum(i["answers"][q] for i in judged) / len(judged), 3) if judged else None for q in QUESTIONS}}


async def run(llm: LLMClient, model: str, seed: int = DEV_LIFE, view: str = "shown") -> dict[str, Any]:
    if seed != DEV_LIFE:
        raise ValueError("the drift check reads summaries, so it runs on the dev life only")
    summaries = json.loads(summaries_path(seed).read_text(encoding="utf-8"))["sizes"][SIZE]
    pairs = agent_runs(seed)
    gate = asyncio.Semaphore(8)

    async def judged(agent: dict[str, Any], runs: list[tuple[str, str]], summary: str) -> dict[str, Any]:
        item = {"id": agent["id"], "runs": len(runs), "chars": len(summary), "summary": summary}
        async with gate:
            try:
                item["answers"] = await judge(llm, model, agent["name"], runs, truncate(summary, SUMMARY_CHARS) if view == "shown" else summary)
            except ValueError as exc:
                item["error"] = str(exc)
        return item

    real = await asyncio.gather(*(judged(a, r, summaries[a["id"]]) for a, r in pairs))

    complete = llm_complete(llm, model)
    longest = sorted(pairs, key=lambda p: (-len(p[1]), p[0]["id"]))[:STRESS_AGENTS]

    async def stressed(agent: dict[str, Any], runs: list[tuple[str, str]]) -> dict[str, Any]:
        summary = summaries[agent["id"]]
        tail = [(STATUS_CHECK, NO_NEWS)] * STRESS_FOLDS
        failed = 0
        async with gate:
            for instructions, result in tail:
                try:
                    summary = await update_summary(name=agent["name"], purpose=agent["purpose"], current=summary,
                                                   instructions=instructions, result=result, complete=complete)
                except SummaryError as exc:
                    if isinstance(exc.__cause__, RunAborted):
                        raise exc.__cause__
                    failed += 1
        item = await judged(agent, runs + tail, summary)
        return item | {"base_runs": len(runs), "failed_updates": failed, "before": summaries[agent["id"]]}

    stress = await asyncio.gather(*(stressed(a, r) for a, r in longest))
    by_bucket = {b: rates([i for i in real if bucket(i["runs"]) == b]) for b in ("1 run", "2-3 runs", "4+ runs")}
    return {
        "size": SIZE, "model": model, "seed": seed, "view": view,
        "real": {"all": rates(real), "by_runs": by_bucket,
                 "empty": sum(1 for i in real if not i["summary"]),
                 "over_cap": sum(1 for i in real if i["chars"] > SUMMARY_CHARS),
                 "judge_errors": sum(1 for i in real if "error" in i)},
        "stress": {"agents": len(stress), "folds_added": STRESS_FOLDS, **{k: v for k, v in rates(stress).items() if k != "agents"},
                   "before": rates([next(i for i in real if i["id"] == s["id"]) for s in stress]),
                   "over_cap": sum(1 for i in stress if i["chars"] > SUMMARY_CHARS),
                   "failed_updates": sum(i["failed_updates"] for i in stress)},
        "items": {"real": real, "stress": stress},
    }


def report(body: dict[str, Any]) -> str:
    real, stress = body["real"], body["stress"]
    row = lambda label, r: f"  {label:22} n={r['judged']:>3}  " + "  ".join(f"{q} {r[q]:.2f}" for q in QUESTIONS if r[q] is not None)
    lines = [f"Real summaries ({body['view']}), life 1 at {body['size']}: empty {real['empty']}, over the {SUMMARY_CHARS}-char cap "
             f"{real['over_cap']}, judge errors {real['judge_errors']}", row("all", real["all"])]
    lines += [row(b, r) for b, r in real["by_runs"].items() if r["judged"]]
    lines += [f"Stress: {stress['agents']} longest agents, +{stress['folds_added']} no-news status checks each "
              f"(failed updates {stress['failed_updates']}, over cap {stress['over_cap']})",
              row("before the extra folds", stress["before"]), row("after", stress)]
    noes = [i for part in ("real", "stress") for i in body["items"][part] if "answers" in i
            and not all(i["answers"][q] for q in QUESTIONS)]
    lines.append(f"Judge 'no' answers to read ({len(noes)}; first 8):")
    for item in noes[:8]:
        failed = [q for q in QUESTIONS if not item["answers"][q]]
        lines.append(f"  {item['id']} ({item['runs']} runs) {failed}: {item['answers']['reason']}")
    return "\n".join(lines)


def fake_llm() -> ScriptedLLM:
    def script(messages: list[dict[str, Any]], sample: int) -> dict[str, Any]:
        text = messages[-1]["content"]
        if text.startswith("Helper:"):
            return {"content": json.dumps({q: "yes" for q in QUESTIONS} | {"reason": ""})}
        return {"content": json.dumps({"summary": "Pending: waiting on a reply."})}
    return ScriptedLLM(script)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--view", choices=["shown", "full"], default="shown", help="what the judge reads")
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
        llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(args.budget), concurrency=8)

    async def go() -> dict[str, Any]:
        try:
            return await run(llm, args.model, view=args.view)
        finally:
            await llm.aclose()

    body = asyncio.run(go())
    print(report(body))
    if not args.fake_llm:
        spent = llm.budget.spent_usd
        body["spent_usd"] = round(spent, 4)
        out = args.out_dir / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-summary-drift-{args.view}-dev.json"
        out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"spent ${spent:.4f}  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
