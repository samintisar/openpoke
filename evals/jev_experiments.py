"""Offline Jev experiments over a recorded life-1 dev run: could a "System One" model check or replace routing?

    python -m evals.jev_experiments {j1,j2,j3,j4} [--result FILE] [--limit N] [--dry-run] [--json PATH] [--budget USD]

Each row of the run is replayed as `evals.second_look_calibration` does (production `dispatch` over a fresh
directory of the world's cards, LIFE 1 ONLY), so every `send_message_to_agent` call is seen with the directory it
met. Jev gets the turn as production would render it: the recent conversation, the new message and the cards
through `render_cards` (summaries included).

- j1 reviewer: on rows where the calibrated second-look trigger fires (`DEFAULT_SECOND_LOOK`), one choice over the
  gate's cards (top 5, the target forced in) + "new" + "ask". Jev's verdict is graded against gold as a routing
  decision (`evals.grading.grade`), and fed through the second look's `effect()` to see which rows it would fix or
  break.
- j2 trigger: the same question for every proposed decision on user turns at 200 and 1,000 agents. A row is
  flagged when, for any call, Jev's choice differs from the IA's (reuse X -> X, create -> "new") or Jev's
  confidence is under t. Rate and failure precision/recall per t, next to the calibrated trigger on the same rows.
- j3 router: Jev's choice over the cards the IA was shown (+ "new" + "ask") as the whole routing decision, graded
  against gold per kind and size, against the recorded IA on the same rows. One choice cannot pass multi_task.
- j4 duplicate: for every create that went through, "which of the top 3 cards already covers this matter, or
  none"; score 1 - P(none). Precision/recall at catching duplicate_create rows across thresholds.

The question does not show Jev the IA's proposal in j1-j3 (j2 must not be anchored on it; j1 shares j2's cache).
`--dry-run` prints one request body and the token/cost estimate, and calls nothing. Answers are cached by request
body (`evals/.cache/llm/jev-*.json`), so reruns are free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from server.services.agents.cards import AgentCard
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.dispatch import dispatch
from server.services.agents.jev import JEV_USD_PER_M_INPUT, JevError, choice_question, parse_choice
from server.services.agents.render import SUMMARY_CHARS, render_cards, truncate
# The route question (J1/J2) is the production Jev reviewer's question (Plan 6c J5).
from server.services.agents.jev_review import (  # noqa: F401
    ASK_CRITERION,
    NEW_CRITERION,
    ROUTE_INSTRUCTIONS,
    card_criterion,
    jev_state,
    route_criteria,
)
from server.services.agents.retrieval import RetrievalConfig
from server.services.agents.second_look import (
    CONTEXT_TURNS,
    DEFAULT_SECOND_LOOK,
    TOP_CARDS,
    Verdict,
    _safe,
    effect,
    top_cards,
)

from .cases import Case
from .charts import RESULTS
from .grading import CORRECT, DUPLICATE_CREATE, ERROR, Decision, grade
from .jev_llm import body_key, estimate_tokens
from .second_look_calibration import SIZES, Proposal, _proposal, _replay, sweep, trigger_reason

LIFE = 1
NEW, ASK, NONE = "new", "ask", "none"
DUP_TOP = 3
REVIEW_QID, ROUTE_QID, DUP_QID = "route", "route", "covered_by"
CONFIDENCES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]  # 0.0: disagreement only; 1.01: every call
DUP_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OUT_DIR = Path(__file__).resolve().parent / ".cache" / "jev"

DUP_INSTRUCTIONS = (
    "The assistant is about to create a new agent for the new message. Each agent owns one matter: a person plus a "
    "subject, including every later step of it. Which listed agent already owns this matter? Choose \"none\" if no "
    "listed agent does."
)
NONE_CRITERION = "No listed agent already covers this matter."


# --- what Jev sees --------------------------------------------------------------------------------------------------

def dup_criteria(cards: Sequence[AgentCard]) -> dict[str, str]:
    criteria = {card.id: card_criterion(card) for card in cards}
    if NONE in criteria:
        raise ValueError("a card id collides with the 'none' option")
    return {**criteria, NONE: NONE_CRITERION}


# --- the replay -----------------------------------------------------------------------------------------------------

@dataclass
class Call:
    """One valid `send_message_to_agent` call of a row, with the Jev inputs built from the directory it met."""

    index: int
    args: dict[str, Any]
    proposal: Proposal
    review_state: str  # the gate's cards: top 5 at the call, the reuse target forced in
    review_criteria: dict[str, str]
    dup_state: Optional[str] = None  # creates only: the top 3 cards and the proposed create
    dup_criteria: Optional[dict[str, str]] = None

    @property
    def ia_choice(self) -> Optional[str]:
        if self.proposal.kind == "reuse":
            return self.proposal.target
        return NEW if self.proposal.kind == "create" else None


@dataclass
class Turn:
    row: dict[str, Any]
    case: Case
    cards: list[AgentCard]
    now: datetime
    guard: bool
    rcfg: Optional[RetrievalConfig]
    router_state: str  # the cards the IA was shown
    router_criteria: dict[str, str]
    calls: list[Call] = field(default_factory=list)

    @property
    def user(self) -> bool:
        return self.case.message_type == "user"

    @property
    def size(self) -> str:
        return self.row["n_agents"]


def build_turn(row: dict[str, Any], case: Case, cards: Sequence[AgentCard], shown: Sequence[AgentCard],
               now: datetime, *, guard: bool = True, rcfg: Optional[RetrievalConfig] = None) -> Turn:
    context = [entry[1] for entry in case.context[-CONTEXT_TURNS:]]
    turn = Turn(row=row, case=case, cards=list(cards), now=now, guard=guard, rcfg=rcfg,
                router_state=jev_state(case.message, context, shown, now, len(cards)),
                router_criteria=route_criteria(shown))
    for index, (args, directory) in enumerate(_replay(row, cards, now, guard, rcfg)):
        prop = _proposal(args, case.message, directory, now, guard, rcfg)
        cards_now = directory.list_cards()
        ranked = [card for card, _ in top_cards(case.message, cards_now, now, rcfg)]
        review = ranked
        if prop.kind == "reuse" and all(card.id != prop.target for card in ranked):
            review = [directory.get(prop.target), *ranked[: TOP_CARDS - 1]]  # as the gate does
        call = Call(index, dict(args), prop, jev_state(case.message, context, review, now, len(cards_now)),
                    route_criteria(review))
        if prop.kind == "create" and not prop.guarded and ranked:
            described = f"Create a new agent named {args.get('name')!r} (purpose: {args.get('purpose')})."
            call.dup_state = jev_state(case.message, context, ranked[:DUP_TOP], now, len(cards_now), described)
            call.dup_criteria = dup_criteria(ranked[:DUP_TOP])
        turn.calls.append(call)
    return turn


def resolve_result(result: Optional[str]) -> Path:
    if result:
        path = Path(result)
        return path if path.exists() else RESULTS / result
    return latest_result()


def latest_result() -> Path:
    """The newest valid life-1 dev step run with the status rule and the second look off (the IA's own proposals)."""
    def ok(path: Path) -> bool:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        except (OSError, ValueError, KeyError):
            return False
        options = meta.get("system_options") or {}
        return (bool(meta.get("valid")) and meta.get("life") == LIFE and meta.get("split") == "dev"
                and not options.get("second_look") and options.get("status_policy") is False)
    found = [p for p in sorted(RESULTS.glob("*-v2-step*-dev.json")) if ok(p)]
    if not found:
        raise FileNotFoundError("no valid life-1 dev v2-step result in evals/results")
    return found[-1]


def read_result(path: Path) -> dict[str, Any]:
    """The result body; anything but a life-1 dev run is refused before any world is read."""
    body = json.loads(path.read_text(encoding="utf-8"))
    meta = body["meta"]
    if meta.get("life") != LIFE or meta.get("split") != "dev":
        raise ValueError(f"{path.name}: Jev experiments read life 1 dev runs only (life={meta.get('life')}, "
                         f"split={meta.get('split')})")
    return body


def load_turns(path: Path, sizes: Optional[Sequence[str]] = None,
               limit: Optional[int] = None) -> tuple[dict[str, Any], list[Turn]]:
    from .systems.retrieved import RetrievedSystem
    from .worlds.lived.bench_v2 import load_v2

    body = read_result(path)
    meta = body["meta"]
    system = RetrievedSystem(**{k: v for k, v in (meta.get("system_options") or {}).items() if k != "second_look"})
    rows = [row for row in body["routing_rows"] if row["outcome"] != ERROR
            and (sizes is None or row["n_agents"] in sizes)]
    if limit is not None:
        rows = rows[:limit]
    worlds = {}
    for label, world, cases in load_v2(LIFE):  # life 1 only: life 2 is never read
        worlds[label] = (world, system.cards(world), {case.id: case for case in cases})
    turns = []
    for row in rows:
        world, cards, cases = worlds[row["n_agents"]]
        case = cases[row["case_id"]]
        turns.append(build_turn(row, case, cards, system.selected(world, case), world.now,
                                guard=system.guard, rcfg=system.cfg))
    return meta, turns


# --- grading --------------------------------------------------------------------------------------------------------

def grade_choice(case: Case, choice: str) -> str:
    """A single choice as the whole routing decision, graded like the IA's."""
    if choice == ASK:
        return grade(case, Decision(clarified=True))
    if choice == NEW:
        return grade(case, Decision(created_names=["new matter"]))
    return grade(case, Decision(dispatched_ids=[choice]))


def redecide(turn: Turn, overrides: Optional[dict[int, Optional[dict[str, Any]]]] = None) -> str:
    """Re-dispatch the row's calls as the eval did, with call i's arguments replaced (None: held, the IA asks)."""
    overrides = overrides or {}
    directory = InMemoryDirectory(turn.cards)
    decision = Decision(clarified=bool(turn.row.get("clarified")))
    created: set[str] = set()
    index = -1
    for call in turn.row.get("tool_calls") or []:
        if call.get("name") != "send_message_to_agent" or "error" in call:
            continue
        index += 1
        args = call.get("arguments") or {}
        if index in overrides:
            args = overrides[index]
            if args is None:
                decision.clarified = True
                continue
        result = dispatch(directory, instructions=str(args.get("instructions", "")), agent_id=args.get("agent_id"),
                          name=args.get("name"), purpose=args.get("purpose"),
                          force_new=bool(args.get("force_new", False)) or not turn.guard, now=turn.now, cfg=turn.rcfg)
        if result.action == "created":
            created.add(result.card.id)
            decision.created_names.append(result.card.name)
        elif result.action == "reused" and result.card.id not in created \
                and result.card.id not in decision.dispatched_ids:
            decision.dispatched_ids.append(result.card.id)
    return grade(turn.case, decision)


def verdict_for(choice: str) -> Verdict:
    if choice in (NEW, ASK):
        return Verdict(choice, None, "jev")
    return Verdict("reuse", choice, "jev")


def apply_verdict(turn: Turn, call: Call, choice: str) -> tuple[str, str]:
    """(effect action, the row's outcome after it) when the second look on `call` returns Jev's `choice`."""
    proposal = call.proposal.kind
    outcome = effect(verdict_for(choice), proposal=proposal, target_id=call.proposal.target)
    instructions = call.args.get("instructions", "")
    if outcome.action == "proceed":
        return outcome.action, redecide(turn)
    if outcome.action == "redirect":
        return outcome.action, redecide(turn, {call.index: {"agent_id": outcome.agent_id,
                                                            "instructions": instructions}})
    if choice == NEW:  # held reuse: the IA is told to create with force_new=true
        return "hold_new", redecide(turn, {call.index: {"name": "New matter", "purpose": turn.case.message,
                                                        "instructions": instructions, "force_new": True}})
    return "hold_ask", redecide(turn, {call.index: None})


# --- requests -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    key: tuple[int, int]  # (turn, call); call -1: the turn's router question
    state: str
    qid: str
    criteria: tuple[tuple[str, str], ...]
    instructions: str

    @property
    def questions(self) -> dict[str, dict[str, Any]]:
        return {self.qid: choice_question(self.instructions, dict(self.criteria))}

    def body(self, model: str = "jev-latest") -> dict[str, Any]:
        return {"model": model, "state": self.state, "questions": self.questions}


def _route_request(t: int, call: Call) -> Request:
    return Request((t, call.index), call.review_state, REVIEW_QID, tuple(call.review_criteria.items()),
                   ROUTE_INSTRUCTIONS)


def j1_rows(turns: Sequence[Turn]) -> list[tuple[int, Call]]:
    """The call the calibrated trigger looks at, per user turn at 200/1,000 (at most one per turn)."""
    out = []
    for t, turn in enumerate(turns):
        if not turn.user or turn.size not in SIZES:
            continue
        for call in turn.calls:
            if trigger_reason(call.proposal, DEFAULT_SECOND_LOOK) is not None:
                out.append((t, call))
                break
    return out


def plan(experiment: str, turns: Sequence[Turn]) -> list[Request]:
    if experiment == "j1":
        return [_route_request(t, call) for t, call in j1_rows(turns)]
    if experiment == "j2":
        return [_route_request(t, call) for t, turn in enumerate(turns)
                if turn.user and turn.size in SIZES for call in turn.calls if call.ia_choice is not None]
    if experiment == "j3":
        return [Request((t, -1), turn.router_state, ROUTE_QID, tuple(turn.router_criteria.items()),
                        ROUTE_INSTRUCTIONS) for t, turn in enumerate(turns)]
    if experiment == "j4":
        return [Request((t, call.index), call.dup_state, DUP_QID, tuple(call.dup_criteria.items()), DUP_INSTRUCTIONS)
                for t, turn in enumerate(turns) for call in turn.calls if call.dup_state is not None]
    raise ValueError(f"unknown experiment {experiment!r}")


def estimate(requests: Sequence[Request]) -> dict[str, Any]:
    """Unique bodies (the cache answers repeats), their estimated input tokens and cost."""
    unique = {body_key(r.body()): estimate_tokens(r.body()) for r in requests}
    tokens = sum(unique.values())
    return {"requests": len(requests), "unique_requests": len(unique), "est_input_tokens": tokens,
            "est_max_request_tokens": max(unique.values(), default=0),
            "est_cost_usd": round(tokens * JEV_USD_PER_M_INPUT / 1e6, 4)}


Answer = tuple[str, dict[str, float], float]


async def ask_all(jev: Any, requests: Sequence[Request]) -> tuple[dict[tuple[int, int], Answer], list[str]]:
    """Every request's (choice, probabilities, confidence); a failed one is left out and its error listed."""
    async def one(request: Request):
        try:
            response = await jev.ask(request.state, request.questions)
            return request.key, parse_choice(response, request.qid, {k for k, _ in request.criteria}), None
        except JevError as exc:
            return request.key, None, f"{request.key}: {exc}"
    answers, errors = {}, []
    for key, answer, error in await asyncio.gather(*(one(r) for r in requests)):
        if error:
            errors.append(error)
        else:
            answers[key] = answer
    return answers, errors


# --- scoring --------------------------------------------------------------------------------------------------------

def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def score_j1(turns: Sequence[Turn], answers: dict[tuple[int, int], Answer]) -> dict[str, Any]:
    rows, verdicts, effects, changes = [], Counter(), Counter(), Counter()
    mismatch = 0
    for t, call in j1_rows(turns):
        if (t, call.index) not in answers:
            continue
        turn = turns[t]
        mismatch += redecide(turn) != turn.row["outcome"]  # the re-dispatch reproduces the recorded grade
        choice, _, confidence = answers[(t, call.index)]
        verdict_outcome = grade_choice(turn.case, choice)
        action, after = apply_verdict(turn, call, choice)
        before = turn.row["outcome"]
        change = ("fixed" if before != CORRECT and after == CORRECT else
                  "broke" if before == CORRECT and after != CORRECT else "same")
        verdicts[verdict_outcome == CORRECT] += 1
        effects[action] += 1
        changes[change] += 1
        rows.append({"case_id": turn.case.id, "sample": turn.row.get("sample"), "size": turn.size,
                     "kind": turn.case.kind, "proposal": call.proposal.kind, "ia_choice": call.ia_choice,
                     "reason": trigger_reason(call.proposal, DEFAULT_SECOND_LOOK), "jev_choice": choice,
                     "confidence": round(confidence, 4), "verdict_outcome": verdict_outcome, "effect": action,
                     "before": before, "after": after, "change": change})
    n = len(rows)
    return {"looked": n, "verdict_accuracy": _rate(verdicts[True], n), "effects": dict(effects),
            "fixed": changes["fixed"], "broke": changes["broke"], "same": changes["same"],
            "net": changes["fixed"] - changes["broke"], "regrade_mismatch": mismatch,
            "ia_correct_before": sum(r["before"] == CORRECT for r in rows),
            "correct_after": sum(r["after"] == CORRECT for r in rows), "rows": rows}


def calibration_items(turns: Sequence[Turn], sizes: Sequence[str] = SIZES) -> list[dict[str, Any]]:
    return [{"outcome": t.row["outcome"], "n_agents": t.size, "user": t.user,
             "proposals": [c.proposal for c in t.calls]} for t in turns if t.size in sizes]


def score_j2(turns: Sequence[Turn], answers: dict[tuple[int, int], Answer],
             confidences: Sequence[float] = CONFIDENCES) -> dict[str, Any]:
    items = [(t, turn) for t, turn in enumerate(turns) if turn.size in SIZES]
    failures = Counter(turn.row["outcome"] for _, turn in items if turn.row["outcome"] != CORRECT)
    size_rows = Counter(turn.size for _, turn in items)
    table = []
    for threshold in confidences:
        flagged, covered, by_size = 0, Counter(), Counter()
        for t, turn in items:
            if not turn.user:
                continue
            hit = False
            for call in turn.calls:
                answer = answers.get((t, call.index))
                if answer is None or call.ia_choice is None:
                    continue
                if answer[0] != call.ia_choice or answer[2] < threshold:
                    hit = True
                    break
            if not hit:
                continue
            flagged += 1
            by_size[turn.size] += 1
            if turn.row["outcome"] != CORRECT:
                covered[turn.row["outcome"]] += 1
        hit_total = sum(covered.values())
        table.append({"confidence": threshold, "rows": len(items), "flagged": flagged,
                      "rate": _rate(flagged, len(items)),
                      "by_size": {s: {"rows": n, "flagged": by_size[s], "rate": _rate(by_size[s], n)}
                                  for s, n in sorted(size_rows.items())},
                      "failures": sum(failures.values()), "failures_covered": hit_total,
                      "precision": _rate(hit_total, flagged), "recall": _rate(hit_total, sum(failures.values())),
                      "by_outcome": {o: {"failures": n, "covered": covered[o]} for o, n in sorted(failures.items())}})
    agree = sum(answers[(t, c.index)][0] == c.ia_choice for t, turn in items if turn.user for c in turn.calls
                if (t, c.index) in answers and c.ia_choice is not None)
    asked = sum((t, c.index) in answers for t, turn in items if turn.user for c in turn.calls
                if c.ia_choice is not None)
    [calibrated] = sweep(calibration_items([turn for _, turn in items]), [DEFAULT_SECOND_LOOK], SIZES)
    return {"calls": asked, "agreement": _rate(agree, asked), "table": table, "calibrated_trigger": calibrated}


def score_j3(turns: Sequence[Turn], answers: dict[tuple[int, int], Answer]) -> dict[str, Any]:
    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    outcomes: Counter = Counter()
    for t, turn in enumerate(turns):
        if (t, -1) not in answers:
            continue
        choice = answers[(t, -1)][0]
        jev_outcome = grade_choice(turn.case, choice)
        outcomes[jev_outcome] += 1
        for key in ((turn.case.kind, turn.size), (turn.case.kind, "all"), ("all", turn.size), ("all", "all")):
            cells[key]["rows"] += 1
            cells[key]["jev"] += jev_outcome == CORRECT
            cells[key]["ia"] += turn.row["outcome"] == CORRECT
    table = [{"kind": kind, "size": size, "rows": c["rows"], "jev_pass": _rate(c["jev"], c["rows"]),
              "ia_pass": _rate(c["ia"], c["rows"])} for (kind, size), c in sorted(cells.items())]
    return {"table": table, "jev_outcomes": dict(outcomes)}


def score_j4(turns: Sequence[Turn], answers: dict[tuple[int, int], Answer],
             thresholds: Sequence[float] = DUP_THRESHOLDS) -> dict[str, Any]:
    scored = []  # (row's max covered-probability, chose a card, is duplicate_create)
    for t, turn in enumerate(turns):
        probs = [answers[(t, c.index)] for c in turn.calls if (t, c.index) in answers]
        if not probs:
            continue
        covered = max(1.0 - p.get(NONE, 0.0) for _, p, _ in probs)
        chose = any(choice != NONE for choice, _, _ in probs)
        scored.append((covered, chose, turn.row["outcome"] == DUPLICATE_CREATE, turn.row["outcome"]))
    positives = sum(dup for _, _, dup, _ in scored)

    def line(name: str, flags: list[bool]) -> dict[str, Any]:
        tp = sum(f and s[2] for f, s in zip(flags, scored))
        flagged = sum(flags)
        false_on = Counter(s[3] for f, s in zip(flags, scored) if f and not s[2])
        return {"rule": name, "flagged": flagged, "true_positives": tp, "precision": _rate(tp, flagged),
                "recall": _rate(tp, positives), "false_flags_by_outcome": dict(false_on)}

    table = [line(f"P(covered) >= {t}", [s[0] >= t for s in scored]) for t in thresholds]
    table.append(line("choice != none", [s[1] for s in scored]))
    return {"rows_with_creates": len(scored), "duplicate_create_rows": positives, "table": table}


SCORERS = {"j1": score_j1, "j2": score_j2, "j3": score_j3, "j4": score_j4}
SIZES_FOR = {"j1": SIZES, "j2": SIZES, "j3": None, "j4": None}


# --- output ---------------------------------------------------------------------------------------------------------

def print_report(experiment: str, report: dict[str, Any]) -> None:
    if experiment == "j1":
        print(f"looked at {report['looked']} rows; Jev verdict correct vs gold {report['verdict_accuracy']:.3f}; "
              f"effects {report['effects']}")
        print(f"IA correct before {report['ia_correct_before']} -> after {report['correct_after']}: "
              f"fixed {report['fixed']}, broke {report['broke']}, net {report['net']:+d}")
        for row in report["rows"]:
            if row["change"] != "same":
                print(f"  {row['change']:5} {row['case_id']} s{row['sample']} {row['proposal']} {row['ia_choice']} "
                      f"-> jev {row['jev_choice']} ({row['confidence']:.2f}) {row['before']} -> {row['after']}")
    elif experiment == "j2":
        print(f"{report['calls']} calls asked; Jev agrees with the IA on {report['agreement']:.3f}")
        cal = report["calibrated_trigger"]
        print(f"calibrated trigger, same rows: rate={cal['trigger_rate']:.3f} covered={cal['failures_covered']}/"
              f"{cal['failures']} precision={cal['precision']:.2f} recall={cal['recall']:.2f}")
        print(f"{'conf<':>6} {'rate':>6} {'200':>6} {'1000':>6} {'covered':>9} {'prec':>5} {'recall':>6}  by outcome")
        for line in report["table"]:
            sizes = [line["by_size"].get(s, {}).get("rate", 0.0) for s in SIZES]
            outcomes = " ".join(f"{o}={v['covered']}/{v['failures']}" for o, v in line["by_outcome"].items())
            print(f"{line['confidence']:>6.2f} {line['rate']:>6.3f} {sizes[0]:>6.3f} {sizes[1]:>6.3f} "
                  f"{line['failures_covered']:>4}/{line['failures']:<4} {line['precision']:>5.2f} "
                  f"{line['recall']:>6.2f}  {outcomes}")
    elif experiment == "j3":
        print(f"Jev outcomes: {report['jev_outcomes']}")
        print(f"{'kind':24} {'size':11} {'rows':>5} {'jev':>6} {'ia':>6}")
        for line in report["table"]:
            print(f"{line['kind']:24} {line['size']:11} {line['rows']:>5} {line['jev_pass']:>6.3f} "
                  f"{line['ia_pass']:>6.3f}")
    else:
        print(f"{report['rows_with_creates']} rows with creates, {report['duplicate_create_rows']} duplicate_create")
        for line in report["table"]:
            print(f"  {line['rule']:20} flagged={line['flagged']:4} tp={line['true_positives']:3} "
                  f"precision={line['precision']:.2f} recall={line['recall']:.2f}  false={line['false_flags_by_outcome']}")


def _make_jev(budget_usd: float):
    from .jev_llm import JEV_REQUESTS_PER_MINUTE, CachedJev
    from .llm import Budget, RateLimiter
    from .run import CACHE_DIR
    from server.services.agents.jev import JevClient, api_key_from_env

    key = api_key_from_env()
    if not key:
        return None
    return CachedJev(JevClient(api_key=key, timeout_s=30.0), cache_dir=CACHE_DIR, budget=Budget(budget_usd),
                     concurrency=8, rate_limiter=RateLimiter(JEV_REQUESTS_PER_MINUTE))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(SCORERS))
    parser.add_argument("--result", help="a life-1 dev result (default: the latest step run without the status rule)")
    parser.add_argument("--limit", type=int, help="only the first N rows (after the experiment's size filter)")
    parser.add_argument("--dry-run", action="store_true", help="print one request body and the estimate; no calls")
    parser.add_argument("--json", type=Path, help=f"where to write the report (default {OUT_DIR}/<experiment>.json)")
    parser.add_argument("--budget", type=float, default=1.0, help="USD cap on uncached Jev calls")
    args = parser.parse_args(argv)

    path = resolve_result(args.result)
    read_result(path)  # refuses anything but life 1 dev before the worlds are loaded
    meta, turns = load_turns(path, SIZES_FOR[args.experiment], args.limit)
    requests = plan(args.experiment, turns)
    est = estimate(requests)
    print(f"{path.name}: {len(turns)} rows; {args.experiment}: {est}")
    if args.dry_run:
        if requests:
            print(json.dumps(requests[0].body(), indent=1))
        return 0
    jev = _make_jev(args.budget)
    if jev is None:
        print("TYPESAFE_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
        return 2

    async def go():
        try:
            return await ask_all(jev, requests)
        finally:
            await jev.client.aclose()

    answers, errors = asyncio.run(go())
    report = SCORERS[args.experiment](turns, answers)
    print_report(args.experiment, report)
    print(f"jev: {jev.meter()}  errors={len(errors)}")
    out = args.json or OUT_DIR / f"{args.experiment}-{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"result": path.name, "experiment": args.experiment, "estimate": est,
                               "jev": jev.meter(), "errors": errors, "report": report}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
