"""The agent directory interface, plus an in-memory implementation for tests and evals."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Iterable, Optional, Protocol

from .cards import AgentCard, extract_entities, slug_id


class AgentDirectory(Protocol):
    def list_cards(self) -> list[AgentCard]: ...

    def get(self, agent_id: str) -> Optional[AgentCard]: ...

    def create(self, name: str, purpose: str, instructions: str, now: datetime) -> AgentCard: ...

    def touch(self, agent_id: str, instructions: str, now: datetime) -> AgentCard: ...

    def last_routed(self) -> list[str]: ...

    def set_last_routed(self, ids: Iterable[str]) -> None: ...


def merge_entities(existing: Iterable[str], texts: Iterable[str]) -> list[str]:
    merged = list(existing)
    merged.extend(e for e in extract_entities(texts) if e not in merged)
    return merged


class InMemoryDirectory:
    """`AgentDirectory` over a dict; copies the given cards so callers' cards are never mutated."""

    def __init__(self, cards: Iterable[AgentCard] = (), last_routed: Iterable[str] = ()) -> None:
        self._cards: dict[str, AgentCard] = {
            card.id: replace(card, entities=list(card.entities)) for card in cards
        }
        self._last_routed = list(last_routed)

    def list_cards(self) -> list[AgentCard]:
        return list(self._cards.values())

    def get(self, agent_id: str) -> Optional[AgentCard]:
        return self._cards.get(agent_id)

    def create(self, name: str, purpose: str, instructions: str, now: datetime) -> AgentCard:
        card = AgentCard(
            id=slug_id(name, self._cards),
            name=name,
            purpose=purpose,
            entities=extract_entities([name, purpose, instructions]),
            created_at=now,
            last_active_at=now,
            run_count=1,
        )
        self._cards[card.id] = card
        return card

    def touch(self, agent_id: str, instructions: str, now: datetime) -> AgentCard:
        card = self._cards[agent_id]
        card.entities = merge_entities(card.entities, [instructions])
        card.last_active_at = now
        card.run_count += 1
        card.status = "active"
        return card

    def last_routed(self) -> list[str]:
        return list(self._last_routed)

    def set_last_routed(self, ids: Iterable[str]) -> None:
        self._last_routed = list(ids)


__all__ = ["AgentDirectory", "InMemoryDirectory", "merge_entities"]
