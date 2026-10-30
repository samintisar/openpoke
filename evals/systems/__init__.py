"""Routing systems under test, by name."""

from __future__ import annotations

import inspect
from typing import Any

from .base import RoutingSystem
from .baseline import BaselineSystem
from .full_cards import FullCardsSystem
from .retrieved import (
    RetrievedNoFuzzySystem,
    RetrievedNoGuardSystem,
    RetrievedNoLastRoutedSystem,
    RetrievedNoLifecycleSystem,
    RetrievedNoSearchSystem,
    RetrievedSystem,
)

SYSTEMS: dict[str, type] = {
    "baseline": BaselineSystem,
    "retrieved": RetrievedSystem,
    "retrieved_nolife": RetrievedNoLifecycleSystem,
    "retrieved_noguard": RetrievedNoGuardSystem,
    "retrieved_nosearch": RetrievedNoSearchSystem,
    "retrieved_nofuzzy": RetrievedNoFuzzySystem,
    "retrieved_nolastrouted": RetrievedNoLastRoutedSystem,
    "full_cards": FullCardsSystem,
}


def get_system(name: str, options: dict[str, Any] | None = None) -> RoutingSystem:
    """The system `name`, built with keyword `options` (eval switches such as {"summaries": False})."""
    try:
        cls = SYSTEMS[name]
    except KeyError:
        raise ValueError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}") from None
    try:
        inspect.signature(cls).bind(**(options or {}))
    except TypeError as exc:
        raise ValueError(f"system {name!r} does not take options {sorted(options or {})}: {exc}") from exc
    return cls(**(options or {}))  # outside the try: a real constructor bug keeps its traceback


__all__ = ["RoutingSystem", "SYSTEMS", "get_system"]
