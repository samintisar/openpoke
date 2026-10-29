"""Sanity gate for a new eval model (spec 2026-10-29 v2, §5): does it use our tools reliably?

    python -m evals.model_gate --routing evals/results/<ts>-model-gate-dev.json --e2e evals/results/<ts>-model-gate-e2e-dev.json

Fails (exit 1) when malformed tool calls exceed 2% of routing tool calls, wait loops (3+ waits or an
iteration-cap hit) exceed 10% of routing rows, or error rows exceed 2% of all rows. Also prints fresh
cost per routing decision and the share of prompt tokens served from the provider's cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAX_MALFORMED = 0.02
MAX_WAIT_LOOPS = 0.10
MAX_ERRORS = 0.02
MALFORMED_PREFIXES = (
    "invalid json", "decoded arguments", "unsupported argument type", "Missing required arguments", "Unknown tool",
)


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def check(routing: dict[str, Any], e2e: dict[str, Any]) -> dict[str, Any]:
    rows = routing["routing_rows"]
    scenarios = e2e["scenario_rows"]
    calls = [call for row in rows for call in row.get("tool_calls", [])]
    malformed = sum(1 for call in calls if str(call.get("error") or "").startswith(MALFORMED_PREFIXES))
    loops = sum(
        1 for row in rows
        if row.get("hit_iteration_cap") or sum(1 for call in row.get("tool_calls", []) if call.get("name") == "wait") >= 3
    )
    errors = sum(1 for row in rows if row.get("outcome") == "error") + sum(1 for s in scenarios if s.get("status") == "error")
    prompt = sum(int(row.get("prompt_tokens", 0)) for row in rows)
    report = {
        "routing_rows": len(rows),
        "e2e_rows": len(scenarios),
        "malformed_rate": _share(malformed, len(calls)),
        "wait_loop_rate": _share(loops, len(rows)),
        "error_rate": _share(errors, len(rows) + len(scenarios)),
        "fresh_cost_per_decision": round(sum(float(row.get("fresh_cost", 0.0)) for row in rows) / len(rows), 6) if rows else 0.0,
        "cached_share": _share(sum(int(row.get("cached_prompt_tokens", 0)) for row in rows), prompt),
    }
    report["passed"] = (
        report["malformed_rate"] <= MAX_MALFORMED
        and report["wait_loop_rate"] <= MAX_WAIT_LOOPS
        and report["error_rate"] <= MAX_ERRORS
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--routing", type=Path, required=True, help="a routing result file (evals.run)")
    parser.add_argument("--e2e", type=Path, required=True, help="an L3 result file (evals.e2e_run)")
    args = parser.parse_args(argv)
    report = check(
        json.loads(args.routing.read_text(encoding="utf-8")), json.loads(args.e2e.read_text(encoding="utf-8"))
    )
    for key, value in report.items():
        print(f"{key:24} {value}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
