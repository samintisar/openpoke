"""Create-time duplicate guard: is there already an agent for this work?"""

from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, Sequence

from .cards import AgentCard, extract_entities, normalize_name
from .retrieval import RetrievalConfig, content_tokens, score_cards

NAME_SIMILARITY = 0.85
ENTITY_SCORE = 0.5
MIN_ENTITY_LEN = 3


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def _shared_entities(new_entities: set[str], card: AgentCard) -> set[str]:
    return {e for e in new_entities & set(card.entities) if len(e) >= MIN_ENTITY_LEN}


def find_duplicate(
    name: str,
    purpose: str,
    cards: Sequence[AgentCard],
    now: datetime,
    cfg: Optional[RetrievalConfig] = None,
) -> Optional[AgentCard]:
    """The existing card (archived included) that the proposed agent most likely duplicates, if any.

    A duplicate has a near-identical name, or shares an entity with the new agent, scores at least
    `ENTITY_SCORE` against its purpose, and has a topic word of the new agent in its own name (or shares two
    entities). The topic check keeps "same person, new topic" from being flagged.
    """
    cards = list(cards)
    if not cards:
        return None
    by_name = max(cards, key=lambda card: name_similarity(name, card.name))
    if name_similarity(name, by_name.name) >= NAME_SIMILARITY:
        return by_name

    new_entities = set(extract_entities([name, purpose]))
    if not new_entities:
        return None
    new_tokens = set(content_tokens(f"{name} {purpose}"))
    for card, score in score_cards(purpose, cards, now, (), cfg):
        if score < ENTITY_SCORE:
            break
        shared = _shared_entities(new_entities, card)
        if not shared:
            continue
        # Topic check against the card's concise name: purposes are long and share filler words.
        topic = (new_tokens & set(content_tokens(card.name))) - set(content_tokens(" ".join(shared)))
        if topic or len(shared) >= 2:
            return card
    return None


__all__ = ["find_duplicate", "name_similarity"]
