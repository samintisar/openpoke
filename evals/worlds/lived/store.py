"""Save and load the frozen lived-in world: one roster and its probe cases per size."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ...cases import Case
from ..model import AgentSpec, World
from .model import Snapshot, day_to_datetime

DEFAULT_PATH = Path(__file__).resolve().parent / "world-s1.json"
FORMAT = 1


def _case_dict(case: Case) -> dict[str, Any]:
    return {f.name: getattr(case, f.name) for f in fields(Case)}


def _agent(data: dict[str, Any]) -> AgentSpec:
    return AgentSpec(**{**data, "history": [tuple(h) for h in data.get("history", [])]})


def _case(data: dict[str, Any]) -> Case:
    return Case(**{**data, "context": [tuple(c) for c in data.get("context", [])]})


def save(path: Path, *, seed: int, units: list[tuple[Snapshot, list[Case]]], meta: dict[str, Any]) -> None:
    """Write each (snapshot, probes) as a size entry labelled lived-{size}."""
    sizes = [
        {
            "label": f"lived-{snapshot.size}",
            "size": snapshot.size,
            "day": snapshot.day,
            "agents": [asdict(agent) for agent in snapshot.agents],
            "thread_agents": snapshot.thread_agents,
            "cases": [_case_dict(case) for case in cases],
        }
        for snapshot, cases in units
    ]
    body = {"format": FORMAT, "seed": seed, "meta": meta, "sizes": sizes}
    path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_lived(path: Path = DEFAULT_PATH) -> list[tuple[str, World, list[Case]]]:
    """(label, world, cases) per saved size, smallest first."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported lived world format {data.get('format')!r}")
    units = []
    for entry in sorted(data["sizes"], key=lambda e: e["size"]):
        world = World(
            seed=data["seed"], agents=[_agent(a) for a in entry["agents"]], now=day_to_datetime(entry["day"])
        )
        units.append((entry["label"], world, [_case(c) for c in entry["cases"]]))
    return units
