"""Offline trigger calibration for the second look: replay the triggers over a run's decisions.

    python -m evals.second_look_calibration [RESULT.json]   # a life-1 dev run without the second look

Each routing row's `send_message_to_agent` calls are replayed through production `dispatch` over a fresh
directory of the world's cards (life 1 only), so every call sees what the gate would see, in-turn creates
included. Per call, the triggers' config-independent features are computed once (best score and evidence for a
create, the duplicate guard, a shared person, the coverages for a reuse); each config then only applies its
thresholds, with at most one look per user turn, like `RetrievedTurn`.

For each (min_score, reuse_coverage, name_coverage): the trigger rate (over valid rows, as `second_look_rate`),
the rate by trigger reason, and how many failed rows a look would have seen, by failure outcome. The replay is
checked against the production trigger functions at the default config. Free: no model calls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import product
from typing import Any, Iterable, Iterator, Optional, Sequence

from server.agents.interaction_agent.second_look_gate import _force_new
from server.services.agents.cards import AgentCard
from server.services.agents.dedup import find_duplicate
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.dispatch import dispatch
from server.services.agents.retrieval import RetrievalConfig, _entity_terms, content_tokens, tokenize
from server.services.agents.second_look import (
    DEFAULT_SECOND_LOOK,
    MIN_SHARED_TERMS,
    SecondLookConfig,
    _card_people,
    _card_text,
    _message_people,
    _person_terms,
    _weighted_coverage,
    create_trigger,
    reuse_trigger,
    top_cards,
    topic_coverage,
)

from .charts import RESULTS
from .grading import CORRECT, ERROR

REASONS = ("create", "shared_person", "unmentioned_name", "low_coverage")
# The last min_score (> 1.0) sweeps the create trigger off (`create=False`); the 0.0 coverages switch a signal off.
MIN_SCORES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
COVERAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
NAME_COVERAGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
CAP = 0.15  # leaves headroom under the plan's 0.20 trigger-rate limit
SIZES = ("lived-200", "lived-1000")


@dataclass(frozen=True)
class Proposal:
    """One `send_message_to_agent` call as the gate sees it. `kind` "none": the gate never looks at it."""

    kind: str
    target: Optional[str] = None
    guarded: bool = False  # create: the duplicate guard answers first
    best_score: Optional[float] = None  # create: the best card's retrieval score
    evidence: bool = False  # create: that card shares a person or MIN_SHARED_TERMS content words
    shared_person: bool = False  # reuse: a top card that outranks the target shares a person with the message
    named_coverage: Optional[float] = None  # reuse: coverage with the unmentioned names, when there are any
    coverage: Optional[float] = None  # reuse: topic coverage on the target


def trigger_reason(prop: Proposal, cfg: SecondLookConfig) -> Optional[str]:
    """The trigger that fires on `prop` under `cfg` (the order of `reuse_trigger`'s checks), or None."""
    if prop.kind == "create":
        fires = cfg.create and not prop.guarded and prop.evidence and prop.best_score is not None and prop.best_score >= cfg.min_score
        return "create" if fires else None
    if prop.kind == "reuse":
        if prop.shared_person:
            return "shared_person"
        if prop.named_coverage is not None and prop.named_coverage < cfg.name_coverage:
            return "unmentioned_name"
        if prop.coverage is not None and prop.coverage < cfg.reuse_coverage:
            return "low_coverage"
    return None


def first_trigger(proposals: Iterable[Proposal], cfg: SecondLookConfig) -> Optional[str]:
    """At most one look per turn: the first call that triggers."""
    for prop in proposals:
        reason = trigger_reason(prop, cfg)
        if reason is not None:
            return reason
    return None


def _replay(row: dict[str, Any], cards: Sequence[AgentCard], now: datetime, guard: bool,
            rcfg: Optional[RetrievalConfig]) -> Iterator[tuple[dict[str, Any], InMemoryDirectory]]:
    """Each valid `send_message_to_agent` call with the directory as it stood, then dispatched as the eval did."""
    directory = InMemoryDirectory(cards)
    for call in row.get("tool_calls") or []:
        if call.get("name") != "send_message_to_agent" or "error" in call:
            continue
        args = call.get("arguments") or {}
        yield args, directory
        dispatch(directory, instructions=str(args.get("instructions", "")), agent_id=args.get("agent_id"),
                 name=args.get("name"), purpose=args.get("purpose"),
                 force_new=bool(args.get("force_new", False)) or not guard, now=now, cfg=rcfg)


def _proposal(args: dict[str, Any], message: str, directory: InMemoryDirectory, now: datetime, guard: bool,
              rcfg: Optional[RetrievalConfig]) -> Proposal:
    cards = directory.list_cards()
    agent_id = str(args.get("agent_id") or "").strip()
    target = directory.get(agent_id) if agent_id else None
    if target is not None:  # mirrors reuse_trigger, keeping the values instead of comparing them
        known = set().union(*(_card_people(other) for other in cards), _card_people(target))
        people = _message_people(message, known)
        shared = False
        for card, _ in top_cards(message, cards, now, rcfg):  # only cards that outrank the target
            if card.id == target.id:
                break
            if people & _card_people(card):
                shared = True
                break
        target_people = _card_people(target)
        missing = people - target_people - set(tokenize(_card_text(target)))
        named = None
        if missing and not people & target_people:
            names = _person_terms(message)
            words = {t for t in content_tokens(message) if t not in names} | missing
            named = _weighted_coverage(words, target, cards)
        return Proposal("reuse", target=target.id, shared_person=shared, named_coverage=named,
                        coverage=topic_coverage(message, target, cards))
    if not agent_id and args.get("name") and args.get("purpose"):  # mirrors the gate, then create_trigger
        name, purpose = str(args["name"]), str(args["purpose"])
        guarded = guard and not _force_new(args.get("force_new")) and bool(
            find_duplicate(name, purpose, cards, now, rcfg))
        best = top_cards(message, cards, now, rcfg)
        if not best:
            return Proposal("create", guarded=guarded)
        card, score = best[0]
        known = set().union(*(_card_people(other) for other in cards))
        people = _message_people(message, known) & _entity_terms(card)
        words = set(content_tokens(message)) & set(tokenize(_card_text(card)))
        return Proposal("create", guarded=guarded, best_score=score,
                        evidence=bool(people) or len(words) >= MIN_SHARED_TERMS)
    return Proposal("none")


def proposals_for(row: dict[str, Any], message: str, cards: Sequence[AgentCard], now: datetime, *,
                  guard: bool = True, rcfg: Optional[RetrievalConfig] = None) -> list[Proposal]:
    """The gate's view of each of the row's dispatches, in order."""
    return [_proposal(args, message, directory, now, guard, rcfg)
            for args, directory in _replay(row, cards, now, guard, rcfg)]


def production_triggers(row: dict[str, Any], message: str, cards: Sequence[AgentCard], now: datetime,
                        cfg: SecondLookConfig, *, guard: bool = True,
                        rcfg: Optional[RetrievalConfig] = None) -> list[bool]:
    """Whether the production trigger fires on each dispatch: the replay's check."""
    fired = []
    for args, directory in _replay(row, cards, now, guard, rcfg):
        cards_now = directory.list_cards()
        agent_id = str(args.get("agent_id") or "").strip()
        target = directory.get(agent_id) if agent_id else None
        if target is not None:
            fired.append(reuse_trigger(message, target, cards_now, now, cfg, rcfg) is not None)
        elif not agent_id and args.get("name") and args.get("purpose"):
            guarded = guard and not _force_new(args.get("force_new")) and bool(
                find_duplicate(str(args["name"]), str(args["purpose"]), cards_now, now, rcfg))
            fired.append(not guarded and create_trigger(message, cards_now, now, cfg, rcfg) is not None)
        else:
            fired.append(False)
    return fired


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def sweep(items: Sequence[dict[str, Any]], configs: Iterable[SecondLookConfig],
          sizes: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
    """One line per config. `items`: {"outcome", "n_agents", "user", "proposals"} per valid row."""
    if sizes is not None:
        items = [item for item in items if item["n_agents"] in sizes]
    failures = Counter(item["outcome"] for item in items if item["outcome"] != CORRECT)
    size_rows = Counter(item["n_agents"] for item in items)
    table = []
    for cfg in configs:
        reasons: Counter[str] = Counter()
        reason_failures: Counter[str] = Counter()
        covered: Counter[str] = Counter()
        by_size: Counter[str] = Counter()
        for item in items:
            reason = first_trigger(item["proposals"], cfg) if item["user"] else None
            if reason is None:
                continue
            reasons[reason] += 1
            by_size[item["n_agents"]] += 1
            if item["outcome"] != CORRECT:
                reason_failures[reason] += 1
                covered[item["outcome"]] += 1
        triggered, hit = sum(reasons.values()), sum(covered.values())
        table.append({
            "create": cfg.create, "min_score": cfg.min_score, "reuse_coverage": cfg.reuse_coverage, "name_coverage": cfg.name_coverage,
            "rows": len(items), "triggered": triggered, "trigger_rate": _rate(triggered, len(items)),
            "by_size": {size: {"rows": n, "triggered": by_size[size], "trigger_rate": _rate(by_size[size], n)}
                        for size, n in size_rows.items()},
            "by_reason": dict(reasons), "by_reason_failures": dict(reason_failures),
            "failures": sum(failures.values()), "failures_covered": hit,
            "precision": _rate(hit, triggered), "recall": _rate(hit, sum(failures.values())),
            "by_outcome": {outcome: {"failures": n, "covered": covered[outcome]} for outcome, n in failures.items()},
        })
    return table


def choose(table: Sequence[dict[str, Any]], cap: float = CAP) -> dict[str, Any]:
    """Most failures covered with the rate ≤ cap overall and at every size (ties: lower rate); else the lowest rate."""
    under = [line for line in table if line["trigger_rate"] <= cap
             and all(size["trigger_rate"] <= cap for size in line["by_size"].values())]
    if under:
        return max(under, key=lambda line: (line["failures_covered"], -line["trigger_rate"]))
    return min(table, key=lambda line: line["trigger_rate"])


def grid(min_scores: Sequence[float] = MIN_SCORES, coverages: Sequence[float] = COVERAGES,
         name_coverages: Sequence[float] = NAME_COVERAGES) -> list[SecondLookConfig]:
    return [replace(DEFAULT_SECOND_LOOK, create=m <= 1.0, min_score=m, reuse_coverage=c, name_coverage=n)
            for m, c, n in product(min_scores, coverages, name_coverages)]


def is_default(line: dict[str, Any], cfg: SecondLookConfig = DEFAULT_SECOND_LOOK) -> bool:
    """The grid line that triggers like `cfg` (with the create trigger off, its min_score does not matter)."""
    same_create = line["create"] == cfg.create and (not cfg.create or line["min_score"] == cfg.min_score)
    return same_create and (line["reuse_coverage"], line["name_coverage"]) == (cfg.reuse_coverage, cfg.name_coverage)


# --- the run -------------------------------------------------------------------------------------------------------

def _latest_step1() -> str:
    """The newest complete step-1 dev result at the shipped SUMMARY_CHARS (480; the -300 run was reverted)."""
    def ok(path) -> bool:
        try:
            return bool(json.loads(path.read_text(encoding="utf-8"))["meta"].get("valid"))
        except (OSError, ValueError, KeyError):
            return False
    complete = [p for p in sorted(RESULTS.glob("*-v2-step1-summaries-dev.json")) if ok(p)]
    if not complete:
        raise FileNotFoundError("no complete v2-step1-summaries dev result in evals/results")
    return complete[-1].name


def load_items(result: str, *, check: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    from .systems.retrieved import RetrievedSystem
    from .worlds.lived.bench_v2 import load_v2

    body = json.loads((RESULTS / result).read_text(encoding="utf-8"))
    meta = body["meta"]
    if meta.get("life") != 1 or meta.get("split") != "dev":
        raise ValueError(f"{result}: calibration reads life 1 dev runs only")
    system = RetrievedSystem(**{k: v for k, v in (meta.get("system_options") or {}).items() if k != "second_look"})
    worlds = {}
    for label, world, cases in load_v2(1):  # life 1 only: life 2 is never read
        worlds[label] = (world, system.cards(world), {case.id: case for case in cases})
    items, stats = [], Counter()
    for row in body["routing_rows"]:
        if row["outcome"] == ERROR:
            continue
        world, cards, cases = worlds[row["n_agents"]]
        case = cases[row["case_id"]]
        props = proposals_for(row, case.message, cards, world.now, guard=system.guard, rcfg=system.cfg)
        # the replay reproduces the run's dispatches
        directory_ids = [p.target for p in props if p.kind == "reuse" and p.target in {c.id for c in cards}]
        stats["rows"] += 1
        stats["replay_mismatch"] += set(directory_ids) != set(row["dispatched_ids"])
        if check and case.message_type == "user":
            fired = production_triggers(row, case.message, cards, world.now, DEFAULT_SECOND_LOOK,
                                        guard=system.guard, rcfg=system.cfg)
            mine = [trigger_reason(p, DEFAULT_SECOND_LOOK) is not None for p in props]
            stats["checked_calls"] += len(fired)
            stats["check_mismatch"] += sum(a != b for a, b in zip(fired, mine))
        items.append({"outcome": row["outcome"], "n_agents": row["n_agents"], "case_id": row["case_id"],
                      "kind": row["kind"], "user": case.message_type == "user", "proposals": props})
    return meta, items, dict(stats)


def _detail(title: str, line: dict[str, Any]) -> None:
    print(f"{title}: {_fmt(line)}")
    for size, part in sorted(line["by_size"].items()):
        print(f"    {size}: {part['triggered']}/{part['rows']} rows look ({part['trigger_rate']:.3f})")
    for reason in REASONS:
        n, hit = line["by_reason"].get(reason, 0), line["by_reason_failures"].get(reason, 0)
        print(f"    {reason:16} looks={n:4} ({_rate(n, line['rows']):.3f})  on failures={hit:3}  "
              f"precision={_rate(hit, n):.2f}")
    for outcome, part in sorted(line["by_outcome"].items()):
        print(f"    {outcome:20} looked at {part['covered']:3}/{part['failures']:3}  "
              f"recall={_rate(part['covered'], part['failures']):.2f}")


def _fmt(line: dict[str, Any]) -> str:
    sizes = " ".join(f"{k}={v['trigger_rate']:.3f}" for k, v in sorted(line["by_size"].items()))
    create = f"min_score={line['min_score']:.1f}" if line.get("create", True) else "create=off   "
    return (f"{create} reuse_cov={line['reuse_coverage']:.1f} "
            f"name_cov={line['name_coverage']:.1f}  rate={line['trigger_rate']:.3f} ({sizes})  "
            f"covered={line['failures_covered']}/{line['failures']}  precision={line['precision']:.2f}  "
            f"reasons={line['by_reason']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", nargs="?", help="a result file in evals/results (default: the latest step 1)")
    parser.add_argument("--cap", type=float, default=CAP)
    parser.add_argument("--json", help="also write the full tables to this path")
    parser.add_argument("--no-check", action="store_true", help="skip the check against the production triggers")
    args = parser.parse_args(argv)
    result = args.result or _latest_step1()
    meta, items, stats = load_items(result, check=not args.no_check)
    print(f"{result}: {meta['name']} system_options={meta.get('system_options')}  {stats}")
    configs = grid()
    tables = {"200+1000": sweep(items, configs, SIZES), "all": sweep(items, configs)}
    for size in SIZES:
        tables[size] = sweep(items, configs, (size,))
    default = [line for line in tables["200+1000"] if is_default(line)]
    for line in default:
        _detail("default (200+1000)", line)
    best = choose(tables["200+1000"], args.cap)
    feasible = best["trigger_rate"] <= args.cap and all(
        size["trigger_rate"] <= args.cap for size in best["by_size"].values())
    _detail("recommended (200+1000)" if feasible else f"lowest rate (nothing <= {args.cap})", best)
    print(f"\nfrontier (200+1000, rate <= {args.cap} at every size, best recall per rate):")
    under = sorted((line for line in tables["200+1000"] if line["trigger_rate"] <= args.cap
                    and all(s["trigger_rate"] <= args.cap for s in line["by_size"].values())),
                   key=lambda line: (-line["failures_covered"], line["trigger_rate"]))
    for line in under[:15]:
        print("  ", _fmt(line))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"result": result, "stats": stats, "recommended": best, "tables": tables}, handle, indent=1)


if __name__ == "__main__":
    main()
