"""Load the hand-written world and cases, validating every referenced agent id."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..cases import NEW, Case
from .model import AgentSpec, World

HANDWRITTEN_PATH = Path(__file__).resolve().parent / "handwritten_cases.yaml"


def load_handwritten(path: Path = HANDWRITTEN_PATH) -> tuple[World, list[Case]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    agents = [
        AgentSpec(
            id=entry["id"],
            name=entry["name"],
            purpose=entry["purpose"],
            entities=list(entry.get("entities", [])),
            last_active_days=float(entry["last_active_days"]),
            created_days=float(entry["created_days"]),
            person=entry.get("person"),
            topic=entry.get("topic"),
            core=True,
        )
        for entry in raw["agents"]
    ]
    world = World(seed=0, agents=agents)
    cases = [
        Case(
            id=entry["id"],
            kind=entry["kind"],
            world_seed=0,
            message=entry["message"],
            gold_targets=list(entry.get("gold_targets", [])),
            message_type=entry.get("message_type", "user"),
            context=[(str(tag), str(text)) for tag, text in entry.get("context", [])],
            acceptable_ids=list(entry.get("acceptable_ids", [])),
            clarify_ok=bool(entry.get("clarify_ok", False)),
            dispatch_optional=bool(entry.get("dispatch_optional", False)),
        )
        for entry in raw["cases"]
    ]
    known = world.by_id()
    for case in cases:
        for target in [*case.gold_targets, *case.acceptable_ids]:
            if target != NEW and target not in known:
                raise ValueError(f"{case.id}: unknown agent id {target!r}")
    return world, cases
