"""Paired comparison of two routing runs on the same cases: the v2 keep/revert rule.

    python -m evals.compare BASE.json NEW.json     # result file names under evals/results/

For each size: pass@1, harmful and duplicate rates as NEW minus BASE over the cases both runs have, with a
paired bootstrap interval (charts.bootstrap); per-kind pass@1 for both; prompt tokens per decision; and the
second-look trigger rate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Callable

from .charts import RESULTS, bootstrap
from .grading import SEVERITY

SIZES = ("lived-10", "lived-50", "lived-200", "lived-1000")
METRICS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "pass@1": lambda r: r["outcome"] == "correct",
    "harmful": lambda r: SEVERITY.get(r["outcome"]) == "harmful",
    "duplicate": lambda r: r["outcome"] == "duplicate_create",
}


def _per_case(rows: list[dict[str, Any]], size: str, metric: str) -> dict[str, float]:
    by_case: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["n_agents"] != size or row["outcome"] == "error":
            continue
        if metric == "duplicate" and not row["expects_reuse"]:  # metrics.py: duplicates among reuse cases only
            continue
        by_case[row["case_id"]].append(METRICS[metric](row))
    return {case: sum(v) / len(v) for case, v in by_case.items()}


def paired(base: list[dict[str, Any]], new: list[dict[str, Any]], size: str, metric: str) -> tuple[float, float, float, int]:
    """(mean, low, high, cases) of NEW − BASE over the shared cases."""
    a, b = _per_case(base, size, metric), _per_case(new, size, metric)
    diffs = [b[case] - a[case] for case in sorted(a) if case in b]
    if not diffs:
        return 0.0, 0.0, 0.0, 0
    return (sum(diffs) / len(diffs), *bootstrap(diffs), len(diffs))


def kind_pass(rows: list[dict[str, Any]], size: str) -> dict[str, float]:
    by_kind: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["n_agents"] == size and row["outcome"] != "error":
            by_kind[row["kind"]][row["case_id"]].append(row["outcome"] == "correct")
    return {kind: round(sum(sum(v) / len(v) for v in cases.values()) / len(cases), 2)
            for kind, cases in sorted(by_kind.items())}


def _mean(rows: list[dict[str, Any]], size: str, key: Callable[[dict[str, Any]], float]) -> float | None:
    values = [key(row) for row in rows if row["n_agents"] == size and row["outcome"] != "error"]
    return sum(values) / len(values) if values else None


def report(base_name: str, new_name: str) -> str:
    base = json.loads((RESULTS / base_name).read_text(encoding="utf-8"))["routing_rows"]
    new = json.loads((RESULTS / new_name).read_text(encoding="utf-8"))["routing_rows"]
    lines = [f"BASE {base_name}", f"NEW  {new_name}"]
    for size in SIZES:
        if not any(r["n_agents"] == size for r in new):
            continue
        lines.append(f"\n{size}")
        for metric in METRICS:
            mean, low, high, n = paired(base, new, size, metric)
            counts = f"cases={n} (base {len(_per_case(base, size, metric))}, new {len(_per_case(new, size, metric))})"
            if n == 0:
                lines.append(f"  {metric:10} n/a  {counts}")
            else:
                lines.append(f"  {metric:10} {mean:+.3f} [{low:+.3f}, {high:+.3f}]  {counts}")
        errors = [sum(1 for r in rows if r["n_agents"] == size and r["outcome"] == "error") for rows in (base, new)]
        lines.append(f"  error rows: base {errors[0]}, new {errors[1]}")
        if errors[1] > errors[0]:
            lines.append(f"  WARNING: NEW has more error rows than BASE at {size}; errored cases are dropped from the comparison")
        tokens = [_mean(rows, size, lambda r: r["prompt_tokens"]) for rows in (base, new)]
        looks = _mean(new, size, lambda r: bool(r.get("second_looks")))
        lines.append(f"  tokens     {tokens[0] or 0:,.0f} -> {tokens[1] or 0:,.0f}   second-look rate {looks or 0:.3f}")
        kb, kn = kind_pass(base, size), kind_pass(new, size)
        for kind in sorted(set(kb) | set(kn)):
            lines.append(f"  {kind:24} {kb.get(kind, float('nan')):.2f} -> {kn.get(kind, float('nan')):.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("new")
    args = parser.parse_args(argv)
    print(report(args.base, args.new))


if __name__ == "__main__":
    main()
