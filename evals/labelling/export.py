"""Export L3 transcripts for human labelling (spec §8: judge calibration on hand-labelled transcripts).

Picks two samples per dev scenario from one L3 run, preferring a failing sample so failures are seen,
and writes what the user saw (and, collapsed, what happened behind the scenes) as JSON. The code-check
verdicts ride along so agreement with the human labels can be computed later; the page hides them
until a transcript is labelled.

    python -m evals.labelling.export evals/results/<run>-e2e-dev.json evals/labelling/transcripts.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

SCENARIOS = Path(__file__).resolve().parents[1] / "worlds" / "e2e_scenarios.yaml"
PER_SCENARIO = 2


def _pick(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Up to PER_SCENARIO samples, failing ones first, then by sample index."""
    ranked = sorted(rows, key=lambda r: (bool(r["success"]), r["sample"]))
    return sorted(ranked[:PER_SCENARIO], key=lambda r: r["sample"])


def _turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "origin": turn["origin"],
        "input": turn["input"],
        "messages": turn["messages"],  # what the user saw, in order: replies and shown drafts
        "tools": [{"name": t["name"], "arguments": t["arguments"]} for t in turn["tools"]],
    }


def _setup(scenario: dict[str, Any]) -> dict[str, Any]:
    setup = scenario.get("setup", {})
    return {
        "mailbox": [{k: m.get(k) for k in ("from", "subject", "body")} for m in setup.get("mailbox", [])],
        "conversation": setup.get("conversation", []),
    }


def export(run_path: Path) -> list[dict[str, Any]]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    scenarios = {s["id"]: s for s in yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]}
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in run["scenario_rows"]:
        by_id.setdefault(row["scenario_id"], []).append(row)
    out = []
    for scenario_id in sorted(by_id):
        scenario = scenarios[scenario_id]
        for row in _pick(by_id[scenario_id]):
            trace = row["trace"]
            out.append({
                "id": f"{scenario_id}#{row['sample']}",
                "scenario": scenario_id,
                "kind": row["kind"],
                "description": scenario.get("description", ""),
                "setup": _setup(scenario),
                "turns": [_turn(t) for t in trace["turns"]],
                "sends": [{k: s.get(k) for k in ("tool", "to", "cc", "bcc", "subject", "body")} for s in trace["sends"]],
                "blocked": [
                    {k: d.get(k) for k in ("kind", "recipients", "reason")}
                    for d in trace["decisions"] if not d.get("would_allow")
                ],
                "code": {"success": row["success"], "checks": row["checks"], "unauthorized_sends": row["unauthorized_sends"],
                         "off_script": row["off_script"]},
            })
    return out


def main(argv: list[str]) -> None:
    run_path, out_path = Path(argv[0]), Path(argv[1])
    data = export(run_path)
    out_path.write_text(json.dumps({"run": run_path.name, "transcripts": data}, indent=1), encoding="utf-8")
    print(f"wrote {len(data)} transcripts from {run_path.name} to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
