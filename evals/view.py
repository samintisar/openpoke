"""Render one result JSON as a static HTML report.

    python -m evals.view evals/results/<run>.json   ->  evals/results/<run>.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from .grading import CORRECT, ERROR

_STYLE = """
body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b2430;background:#fff}
table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccd5dd;padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}details{border:1px solid #ccd5dd;border-radius:6px;margin:6px 0;padding:6px 10px}
summary{cursor:pointer}pre{white-space:pre-wrap;background:#f2f5f7;padding:8px;border-radius:4px}.bad{color:#b4400c}
"""


def _cell(value: Any) -> str:
    return "–" if value is None else str(value)


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(_cell(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


_ROUTING_KEYS = (
    "cases", "planned_cases", "incomplete_cases", "pass_at_1", "macro_pass_at_1", "pass_hat_k", "sample_std",
    "duplicate_rate", "duplicate_rate_all", "error_count", "error_rate", "hit_cap_rate", "mean_prompt_tokens",
    "total_cost_usd", "fresh_cost_usd", "cache_hit_rate",
)


def render(data: dict[str, Any]) -> str:
    meta, summary = data.get("meta", {}), data.get("summary", {})
    routing, retrieval = summary.get("routing", {}), summary.get("retrieval", {})
    parts = [f"<!doctype html><meta charset='utf-8'><title>{escape(meta.get('name', '–'))}</title><style>{_STYLE}</style>"]
    parts.append(f"<h1>{escape(meta.get('name', '–'))} · {escape(meta.get('system', '–'))} · {escape(meta.get('split', '–'))}</h1>")
    validity = "valid" if meta.get("valid") else "<span class='bad'>INVALID: protected eval files changed</span>"
    parts.append(
        f"<p>model {escape(meta.get('model', '–'))} · k={meta.get('k', '–')} · git {escape(str(meta.get('git_sha', '–')))} · {validity} · "
        f"spent {_money(meta.get('budget_spent_usd'))} (fresh {_money(meta.get('fresh_cost_usd'))}, "
        f"unattributed {_money(meta.get('unattributed_cost_usd'))})</p>"
    )
    if meta.get("budget_exhausted"):
        parts.append("<p class='bad'>Budget exhausted: unrun cases are counted as incomplete.</p>")
    elif meta.get("aborted"):
        parts.append(f"<p class='bad'>Run aborted: {escape(str(meta['aborted']))}</p>")

    parts.append("<h2>Routing (L2)</h2>")
    parts.append(_table(
        ["n agents", "cases", "planned", "incomplete", "pass@1", "macro pass@1", "pass^k", "sample std",
         "duplicate rate", "duplicate rate (all)", "errors", "error rate", "hit cap rate", "prompt tokens",
         "cost $", "fresh $", "cache hit rate"],
        [[n, *[g.get(key) for key in _ROUTING_KEYS]] for n, g in routing.items()],
    ))
    kinds = sorted({kind for g in routing.values() for kind in g.get("by_kind", {})})
    parts.append("<h3>pass@1 by kind</h3>")
    parts.append(_table(["kind", *routing.keys()], [[kind, *[g.get("by_kind", {}).get(kind) for g in routing.values()]] for kind in kinds]))
    outcomes = sorted({o for g in routing.values() for o in g.get("outcome_rates", {})})
    parts.append("<h3>Outcome rates</h3>")
    parts.append(_table(["outcome", *routing.keys()], [[o, *[g.get("outcome_rates", {}).get(o, 0) for g in routing.values()]] for o in outcomes]))

    parts.append("<h2>Retrieval (L1)</h2>")
    parts.append(_table(
        ["n agents", "cases", "recall", "MRR", "shown", "roster tokens"],
        [[n, g.get("cases"), g.get("recall"), g.get("mrr"), g.get("mean_shown"), g.get("mean_roster_tokens")] for n, g in retrieval.items()],
    ))

    parts.append("<h2>Failing samples</h2>")
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data.get("routing_rows", []):
        if row["outcome"] != CORRECT:
            failures[str(row["n_agents"])].append(row)
    for n, rows in failures.items():
        parts.append(f"<h3>n = {escape(n)} ({len(rows)} failing)</h3>")
        for row in rows:
            detail = {
                "gold": row["gold"], "acceptable": row["acceptable"], "dispatched_ids": row["dispatched_ids"],
                "created_names": row["created_names"], "user_messages": row["user_messages"],
                "tool_calls": row["tool_calls"], "iterations": row["iterations"],
                "hit_iteration_cap": row.get("hit_iteration_cap"), "cost": row["cost"],
                "fresh_cost": row.get("fresh_cost"), "cache_keys": row["cache_keys"],
            }
            error = ""
            if row["outcome"] == ERROR:
                detail = {"error": row.get("error"), **detail}
                error = f" · <span class='bad'>{escape(str(row.get('error')))}</span>"
            parts.append(
                f"<details><summary><b class='bad'>{escape(row['outcome'])}</b> · {escape(row['kind'])} · "
                f"{escape(row['case_id'])} · sample {row['sample']} — {escape(row['message'])}{error}</summary>"
                f"<pre>{escape(json.dumps(detail, indent=2))}</pre></details>"
            )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.result.read_text(encoding="utf-8"))
    out = args.result.with_suffix(".html")
    out.write_text(render(data), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
