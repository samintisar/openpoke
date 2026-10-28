"""The interface every routing system under test implements."""

from __future__ import annotations

from typing import Protocol

from ..cases import Case
from ..worlds.model import World


class RoutingSystem(Protocol):
    name: str

    def render_roster(self, world: World, case: Case | None) -> str:
        """The full <active_agents>…</active_agents> section for the IA prompt."""

    def shown_ids(self, world: World, case: Case | None) -> list[str]:
        """Agent ids visible in that section, in display order (for L1)."""

    def resolve_name(self, world: World, name: str) -> str | None:
        """Map a name the IA passed to send_message_to_agent to an agent id; None = creates."""
