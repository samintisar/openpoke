"""`send_message_to_agent` and `search_agents` semantics, shared by the server and the eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .cards import AgentCard
from .dedup import find_duplicate
from .directory import AgentDirectory
from .render import render_cards
from .retrieval import RetrievalConfig, search

NEAREST_ON_UNKNOWN = 3


@dataclass
class DispatchResult:
    """`action` is `reused`, `created`, `possible_duplicate` or `error`."""

    action: str
    card: Optional[AgentCard] = None
    payload: dict[str, Any] = field(default_factory=dict)


def _submitted(card: AgentCard, created: bool) -> dict[str, Any]:
    return {
        "status": "submitted",
        "agent_id": card.id,
        "agent_name": card.name,
        "new_agent_created": created,
    }


def dispatch(
    directory: AgentDirectory,
    *,
    instructions: str,
    agent_id: Optional[str] = None,
    name: Optional[str] = None,
    purpose: Optional[str] = None,
    force_new: bool = False,
    now: datetime,
    cfg: Optional[RetrievalConfig] = None,
) -> DispatchResult:
    """Reuse the agent `agent_id`, or create one from `name` and `purpose` unless it looks like a duplicate."""
    agent_id = (agent_id or "").strip()
    if agent_id:
        if directory.get(agent_id) is not None:
            card = directory.touch(agent_id, instructions, now)
            return DispatchResult("reused", card, _submitted(card, created=False))
        cards = directory.list_cards()
        nearest = search(agent_id.replace("-", " "), cards, now, cfg, limit=NEAREST_ON_UNKNOWN)
        return DispatchResult(
            "error",
            None,
            {
                "error": f"Unknown agent_id '{agent_id}'. Use an id from the agents listed, or create a new "
                "agent with name and purpose.",
                "nearest_agents": render_cards(nearest, len(cards), now),
            },
        )

    name = (name or "").strip()
    purpose = (purpose or "").strip()
    if not name or not purpose:
        return DispatchResult(
            "error",
            None,
            {"error": "Pass agent_id to reuse an agent, or both name and purpose to create a new one."},
        )

    # Models sometimes send booleans as strings; "false" must not count as true.
    force_new = force_new is True or str(force_new).strip().lower() == "true"
    if not force_new:
        cards = directory.list_cards()
        duplicate = find_duplicate(name, purpose, cards, now, cfg)
        if duplicate is not None:
            return DispatchResult(
                "possible_duplicate",
                duplicate,
                {
                    "status": "possible_duplicate",
                    "candidate": render_cards([duplicate], len(cards), now),
                    "instructions": (
                        f"Nothing was sent. This existing agent works with the same person on the same subject. "
                        f"Resend with agent_id=\"{duplicate.id}\", even if this is a new step (an invoice after "
                        "scheduling, a sign-off after a draft). Resend with force_new=true only if the subject "
                        "is unrelated to this agent's work."
                    ),
                },
            )

    card = directory.create(name, purpose, instructions, now)
    return DispatchResult("created", card, _submitted(card, created=True))


def search_payload(
    directory: AgentDirectory,
    query: str,
    now: datetime,
    cfg: Optional[RetrievalConfig] = None,
) -> str:
    """Rendered `search_agents` results over every card, archived included."""
    cards = directory.list_cards()
    return render_cards(search(query, cards, now, cfg), len(cards), now)


__all__ = ["DispatchResult", "dispatch", "search_payload"]
