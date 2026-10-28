"""Aggregate per-sample rows into experiment metrics, grouped by roster size.

A routing row is one (case, sample) and carries: n_agents, world_seed, case_id, kind, sample,
outcome, expects_reuse, duplicate_possible, prompt_tokens, cost, fresh_cost, cached_calls,
iterations, hit_iteration_cap, latency_s.

`planned` (optional) maps each roster size to the (world_seed, case_id) keys that were meant to
run, and `k` (optional) is the configured sample count. With them, cases that never produced a
row (a run stopped by its budget) count as incomplete, and a planned size with no rows still
gets a group, with None for every rate.

Denominators, per roster size:
- Rows whose outcome is ERROR are dropped before every pass rate and duplicate rate; they are
  counted by error_count and error_rate (over all rows). A case whose samples are all errors
  drops out of `cases` and of every pass rate.
- pass_at_1: mean over cases of each case's share of correct non-error samples.
- macro_pass_at_1: mean over kinds of each kind's case-level pass@1; by_kind holds those values.
- pass_hat_k: share of cases that are correct on all k samples, over only the cases with exactly
  k non-error samples, where k is the configured k if given, else the largest non-error sample
  count of any case. Cases with fewer (planned cases with no rows included) are reported as
  incomplete_cases; planned_cases is the planned count (or the observed case count).
- sample_std: population std across sample indexes of the per-sample pass rate.
- duplicate_rate: duplicate_create share of non-error rows with expects_reuse.
- duplicate_rate_all: the same over non-error rows where duplicate_create is possible
  (duplicate_possible), which adds the optional-dispatch agent_result cases.
- outcome_rates: share of all rows, errors included.
- hit_cap_rate: share of non-error rows whose turn hit the iteration cap.
- total_cost_usd is attributed cost (cached calls count what they cost when first made);
  fresh_cost_usd is only what this run spent; cache_hit_rate is cached_calls / iterations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from .grading import CORRECT, DUPLICATE_CREATE, ERROR


def _by_size(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["n_agents"])].append(row)
    return groups


def summarize_routing(
    rows: list[dict[str, Any]], *, planned: dict[str, set[tuple]] | None = None, k: int | None = None
) -> dict[str, dict[str, Any]]:
    groups = _by_size(rows)
    sizes = list(groups) + [size for size in (planned or {}) if size not in groups]
    return {
        size: _routing_group(groups.get(size, []), None if planned is None else planned.get(size, set()), k)
        for size in sizes
    }


def _rate(values: list[Any], digits: int = 4) -> float | None:
    return round(mean(values), digits) if values else None


def _duplicate_rate(rows: list[dict[str, Any]]) -> float | None:
    return _rate([row["outcome"] == DUPLICATE_CREATE for row in rows])


def _routing_group(rows: list[dict[str, Any]], planned: set[tuple] | None, k_configured: int | None) -> dict[str, Any]:
    valid = [row for row in rows if row["outcome"] != ERROR]
    by_case: dict[tuple, list[bool]] = defaultdict(list)
    by_sample: dict[int, list[bool]] = defaultdict(list)
    case_kind: dict[tuple, str] = {}
    for row in valid:
        ok = row["outcome"] == CORRECT
        case = (row["world_seed"], row["case_id"])
        by_case[case].append(ok)
        by_sample[row["sample"]].append(ok)
        case_kind[case] = row["kind"]

    kind_scores: dict[str, list[float]] = defaultdict(list)
    for case, oks in by_case.items():
        kind_scores[case_kind[case]].append(mean(oks))
    by_kind = {kind: round(mean(scores), 4) for kind, scores in sorted(kind_scores.items())}

    k = k_configured if k_configured is not None else max((len(oks) for oks in by_case.values()), default=0)
    complete = [oks for oks in by_case.values() if len(oks) == k]
    planned_keys = set(by_case) if planned is None else planned | set(by_case)
    outcomes = Counter(row["outcome"] for row in rows)
    error_count = outcomes.get(ERROR, 0)
    total_calls = sum(row["iterations"] for row in rows)
    return {
        "cases": len(by_case),
        "planned_cases": len(planned_keys),
        "samples": len(rows),
        "k": k,
        "incomplete_cases": len(planned_keys) - len(complete),
        "pass_at_1": _rate([mean(oks) for oks in by_case.values()]),
        "macro_pass_at_1": _rate(list(by_kind.values())),
        "pass_hat_k": _rate([all(oks) for oks in complete]),
        "sample_std": round(pstdev([mean(v) for v in by_sample.values()]), 4) if len(by_sample) > 1 else 0.0,
        "error_count": error_count,
        "error_rate": round(error_count / len(rows), 4) if rows else None,
        "duplicate_rate": _duplicate_rate([row for row in valid if row["expects_reuse"]]),
        "duplicate_rate_all": _duplicate_rate([row for row in valid if row["duplicate_possible"]]),
        "outcome_rates": {name: round(count / len(rows), 4) for name, count in sorted(outcomes.items())},
        "hit_cap_rate": _rate([bool(row["hit_iteration_cap"]) for row in valid]),
        "by_kind": by_kind,
        "mean_prompt_tokens": _rate([row["prompt_tokens"] for row in rows], 1),
        "total_cost_usd": round(sum(row["cost"] for row in rows), 4),
        "fresh_cost_usd": round(sum(row["fresh_cost"] for row in rows), 4),
        "cache_hit_rate": round(sum(row["cached_calls"] for row in rows) / total_calls, 4) if total_calls else None,
        "mean_latency_s": _rate([row["latency_s"] for row in rows], 3),
    }


def summarize_retrieval(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for size, group in _by_size(rows).items():
        recalls = [row["recall"] for row in group if row["recall"] is not None]
        mrrs = [row["mrr"] for row in group if row["mrr"] is not None]
        summary[size] = {
            "cases": len(group),
            "recall": round(mean(recalls), 4) if recalls else None,
            "mrr": round(mean(mrrs), 4) if mrrs else None,
            "mean_shown": round(mean(row["shown_count"] for row in group), 1),
            "mean_roster_tokens": round(mean(row["roster_tokens"] for row in group), 1),
        }
    return summary
