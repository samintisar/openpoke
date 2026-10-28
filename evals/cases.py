"""Eval case schema, dev/test split and capability/regression suites."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

NEW = "NEW"  # gold target meaning "a new agent should be created"

ROUTING_KINDS = (
    "follow_up",
    "new_task",
    "same_person_new_topic",
    "vague_reference",
    "multi_task",
    "same_name_people",
    "cross_agent",
    "dormant_revival",
    "typo",
    "agent_result",
)
SCENARIO_KINDS = ("injection", "correction_midrun")  # L3, built in Plan 4
KINDS = ROUTING_KINDS + SCENARIO_KINDS

REGRESSION_FILE = Path(__file__).resolve().parent / "suites" / "regression.txt"


@dataclass
class Case:
    """One message plus what a correct router must do with it."""

    id: str
    kind: str
    world_seed: int
    message: str
    gold_targets: list[str]
    message_type: str = "user"  # "user" or "agent"
    context: list[tuple[str, ...]] = field(default_factory=list)  # (tag, text) or (tag, text, ISO timestamp)
    acceptable_ids: list[str] = field(default_factory=list)
    clarify_ok: bool = False
    dispatch_optional: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown case kind {self.kind!r}")
        if self.message_type not in ("user", "agent"):
            raise ValueError(f"unknown message_type {self.message_type!r}")

    @property
    def expected_ids(self) -> list[str]:
        return [target for target in self.gold_targets if target != NEW]

    @property
    def expects_new(self) -> bool:
        return NEW in self.gold_targets

    @property
    def expects_reuse(self) -> bool:
        return bool(self.expected_ids) and not self.dispatch_optional

    @property
    def split(self) -> str:
        """Stable 70/30 dev/test split derived from the id, so it never drifts."""
        digest = int(hashlib.sha1(self.id.encode("utf-8")).hexdigest(), 16)
        return "dev" if digest % 10 < 7 else "test"


def load_regression_ids(path: Path = REGRESSION_FILE) -> set[str]:
    """Case ids listed one per line; blank lines and # comments are skipped."""
    if not path.exists():
        return set()
    stripped = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in stripped if line and not line.startswith("#")}


def filter_cases(
    cases: Iterable[Case],
    *,
    split: str | None = None,
    suite: str = "all",
    kinds: Iterable[str] | None = None,
    regression_ids: set[str] | None = None,
) -> list[Case]:
    if split not in (None, "dev", "test"):
        raise ValueError(f"unknown split {split!r}")
    if suite not in ("all", "capability", "regression"):
        raise ValueError(f"unknown suite {suite!r}")
    regression = load_regression_ids() if regression_ids is None else regression_ids
    wanted_kinds = set(kinds) if kinds else None
    selected: list[Case] = []
    for case in cases:
        if split and case.split != split:
            continue
        if wanted_kinds and case.kind not in wanted_kinds:
            continue
        if suite == "regression" and case.id not in regression:
            continue
        if suite == "capability" and case.id in regression:
            continue
        selected.append(case)
    return selected
