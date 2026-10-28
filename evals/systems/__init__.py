"""Routing systems under test, by name."""

from __future__ import annotations

from .base import RoutingSystem
from .baseline import BaselineSystem
from .full_cards import FullCardsSystem
from .retrieved import RetrievedNoLifecycleSystem, RetrievedSystem

SYSTEMS: dict[str, type] = {
    "baseline": BaselineSystem,
    "retrieved": RetrievedSystem,
    "retrieved_nolife": RetrievedNoLifecycleSystem,
    "full_cards": FullCardsSystem,
}


def get_system(name: str) -> RoutingSystem:
    try:
        return SYSTEMS[name]()
    except KeyError:
        raise ValueError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}") from None


__all__ = ["RoutingSystem", "SYSTEMS", "get_system"]
