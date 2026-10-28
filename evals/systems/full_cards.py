"""Ablation of `retrieved`: the same cards, prompt and tools, but every card is shown.

Separates what the richer cards and dedup-guarded dispatch buy from what retrieval buys.
"""

from __future__ import annotations

from server.services.agents.cards import AgentCard

from ..cases import Case
from ..worlds.model import World
from .retrieved import RetrievedSystem


class FullCardsSystem(RetrievedSystem):
    """Every card, in roster order (oldest-created first)."""

    name = "full_cards"

    def selected(self, world: World, case: Case | None) -> list[AgentCard]:
        by_id = {card.id: card for card in self.cards(world)}
        return [by_id[agent.id] for agent in world.roster_order()]


__all__ = ["FullCardsSystem"]
