"""Agent cards built from an eval world, the way the registry would hold them after the same history.

Only what a real registry could know is read: id, name, purpose, the requests each agent was sent,
and when it was created and last active. Gold labels (`threads`, `person`, `topic`, `core`, the
world's `roles`) are never touched.

Statuses follow the lifecycle at `world.now` (no world has open work), unless `lifecycle=False`.

The summary is the production updater's output over the agent's instructions and growth's reports
(evals/worlds/lived/summaries.py), unless `summaries=False`. As in production, it is not copied into the
card's entities: retrieval reads the people it names itself.
"""

from __future__ import annotations

from datetime import timedelta

from server.services.agents.cards import AgentCard, extract_entities
from server.services.agents.lifecycle import next_status

from ..worlds.model import AgentSpec, World


def card_for(agent: AgentSpec, world: World, lifecycle: bool = True, summaries: bool = True) -> AgentCard:
    requests = [request for request, _ in agent.history]
    card = AgentCard(
        id=agent.id,
        name=agent.name,
        purpose=agent.purpose,
        entities=extract_entities([agent.name, agent.purpose, *requests]),
        last_status="",
        status="active",
        created_at=world.now - timedelta(days=agent.created_days),
        last_active_at=world.now - timedelta(days=agent.last_active_days),
        run_count=len(agent.history),
        summary=agent.summary if summaries else "",
    )
    if lifecycle:
        card.status = next_status(card, world.now, has_open_work=False)
    return card


def cards_for(world: World, lifecycle: bool = True, summaries: bool = True) -> list[AgentCard]:
    """One card per agent, in the world's agent order."""
    return [card_for(agent, world, lifecycle, summaries) for agent in world.agents]


__all__ = ["card_for", "cards_for"]
