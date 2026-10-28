"""Agent lifecycle: stale agents go dormant, then archived, so they stop competing for the roster.

Nothing is deleted. Dormant cards leave the hot set but stay retrievable per turn; archived cards are
found only through `search_agents`. Any routing to an agent (`touch`) or a trigger fire revives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

from .cards import AgentCard, as_utc
from .directory import AgentDirectory

ACTIVE = "active"
DORMANT = "dormant"
ARCHIVED = "archived"


@dataclass(frozen=True)
class LifecycleConfig:
    dormant_after_days: float = 7
    archive_after_days: float = 30


DEFAULT_CONFIG = LifecycleConfig()


def next_status(
    card: AgentCard, now: datetime, has_open_work: bool, cfg: Optional[LifecycleConfig] = None
) -> str:
    """`active` while the agent has open work; otherwise by time since it was last active (boundaries inclusive)."""
    cfg = cfg or DEFAULT_CONFIG
    if has_open_work:
        return ACTIVE
    idle = as_utc(now) - as_utc(card.last_active_at)
    if idle >= timedelta(days=cfg.archive_after_days):
        return ARCHIVED
    if idle >= timedelta(days=cfg.dormant_after_days):
        return DORMANT
    return ACTIVE


def sweep(
    directory: AgentDirectory,
    now: datetime,
    open_work_ids: Iterable[str] = (),
    cfg: Optional[LifecycleConfig] = None,
) -> dict[str, str]:
    """Apply `next_status` to every card; returns {agent_id: new_status} for the cards that changed.

    `open_work_ids` are agents with an active trigger due soon. Plan 4 adds agents with pending drafts.
    """
    open_ids = set(open_work_ids)
    changed: dict[str, str] = {}
    for card in directory.list_cards():
        status = next_status(card, now, card.id in open_ids, cfg)
        if status != card.status:
            directory.update_card(card.id, status=status)
            changed[card.id] = status
    return changed


__all__ = ["ACTIVE", "ARCHIVED", "DORMANT", "LifecycleConfig", "next_status", "sweep"]
