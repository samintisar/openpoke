"""Load the hand-written world and cases, validating every referenced agent id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..cases import NEW, Case
from .model import AgentSpec, World

HANDWRITTEN_PATH = Path(__file__).resolve().parent / "handwritten_cases.yaml"

AGENT_KEYS = ("id", "name", "purpose", "last_active_days", "created_days")
CASE_KEYS = ("id", "kind", "message")


def _require(entry: dict[str, Any], keys: tuple[str, ...], what: str, index: int) -> None:
    """Raise a ValueError naming the entry if any required key is missing."""
    missing = [key for key in keys if key not in entry]
    if missing:
        label = f"{what} {entry['id']!r}" if "id" in entry else f"{what} #{index}"
        raise ValueError(f"{label} is missing required key(s) {', '.join(map(repr, missing))}")


def _check_unique(ids: list[str], what: str) -> None:
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            raise ValueError(f"duplicate {what} id {item_id!r}")
        seen.add(item_id)


def _agent(entry: dict[str, Any], index: int) -> AgentSpec:
    _require(entry, AGENT_KEYS, "agent", index)
    return AgentSpec(
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


def _case(entry: dict[str, Any], index: int) -> Case:
    _require(entry, CASE_KEYS, "case", index)
    return Case(
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


def load_handwritten(path: Path = HANDWRITTEN_PATH) -> tuple[World, list[Case]]:
    """The hand-written world and its cases; raises ValueError on any malformed entry."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: expected a mapping with 'agents' and 'cases'")
    for key in ("agents", "cases"):
        if key not in raw:
            raise ValueError(f"{path.name} is missing required top-level key {key!r}")
    agents = [_agent(entry, index) for index, entry in enumerate(raw["agents"])]
    _check_unique([agent.id for agent in agents], "agent")
    cases = [_case(entry, index) for index, entry in enumerate(raw["cases"])]
    _check_unique([case.id for case in cases], "case")
    world = World(seed=0, agents=agents)
    known = world.by_id()
    for case in cases:
        if not case.gold_targets and not case.clarify_ok:
            raise ValueError(f"{case.id}: empty gold_targets requires clarify_ok")
        for target in [*case.gold_targets, *case.acceptable_ids]:
            if target != NEW and target not in known:
                raise ValueError(f"{case.id}: unknown agent id {target!r}")
    return world, cases
