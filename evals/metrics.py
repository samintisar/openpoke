"""Aggregate per-sample rows into experiment metrics, grouped by roster size."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from .grading import CORRECT, DUPLICATE_CREATE


def _by_size(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["n_agents"])].append(row)
    return groups


def summarize_routing(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {size: _routing_group(group) for size, group in _by_size(rows).items()}


def _routing_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[tuple, list[bool]] = defaultdict(list)
    by_sample: dict[int, list[bool]] = defaultdict(list)
    by_kind: dict[str, list[bool]] = defaultdict(list)
    case_kind: dict[tuple, str] = {}
    for row in rows:
        ok = row["outcome"] == CORRECT
        by_case[(row["world_seed"], row["case_id"])].append(ok)
        by_sample[row["sample"]].append(ok)
        by_kind[row["kind"]].append(ok)
        case_kind[(row["world_seed"], row["case_id"])] = row["kind"]
    # Case-level pass@1 per kind, then averaged over kinds: kinds can have unequal case counts.
    kind_case_scores: dict[str, list[float]] = defaultdict(list)
    for case, oks in by_case.items():
        kind_case_scores[case_kind[case]].append(mean(oks))
    reuse = [row for row in rows if row["expects_reuse"]]
    outcomes = Counter(row["outcome"] for row in rows)
    return {
        "cases": len(by_case),
        "samples": len(rows),
        "k": max(len(v) for v in by_case.values()),
        "pass_at_1": round(mean(mean(v) for v in by_case.values()), 4),
        "macro_pass_at_1": round(mean(mean(v) for v in kind_case_scores.values()), 4),
        "pass_hat_k": round(mean(all(v) for v in by_case.values()), 4),
        "sample_std": round(pstdev([mean(v) for v in by_sample.values()]), 4) if len(by_sample) > 1 else 0.0,
        "duplicate_rate": (
            round(sum(row["outcome"] == DUPLICATE_CREATE for row in reuse) / len(reuse), 4) if reuse else None
        ),
        "outcome_rates": {name: round(count / len(rows), 4) for name, count in sorted(outcomes.items())},
        "by_kind": {kind: round(mean(v), 4) for kind, v in sorted(by_kind.items())},
        "mean_prompt_tokens": round(mean(row["prompt_tokens"] for row in rows), 1),
        "total_cost_usd": round(sum(row["cost"] for row in rows), 4),
        "mean_latency_s": round(mean(row["latency_s"] for row in rows), 3),
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
