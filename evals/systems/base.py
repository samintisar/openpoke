"""The interface every routing system under test implements."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from ..cases import Case
from ..grading import Decision
from ..worlds.model import World


class TurnHandler(Protocol):
    def handle(self, name: str, args: dict[str, Any], decision: Decision) -> dict[str, Any]:
        """Record a valid tool call's effects on the Decision and return the tool result the IA sees:
        {"tool", "status", "arguments", "result"}, as production's _format_tool_result builds it."""


class RoutingSystem(Protocol):
    name: str

    def render_roster(self, world: World, case: Case | None) -> str:
        """The full <active_agents>…</active_agents> section for the IA prompt."""

    def shown_ids(self, world: World, case: Case | None) -> list[str]:
        """Agent ids visible in that section, in display order (for L1)."""

    def system_prompt(self) -> str:
        """The IA system prompt this system runs with."""

    def tool_schemas(self) -> list[dict[str, Any]]:
        """The IA tool schemas this system runs with."""

    def tool_signatures(self) -> dict[str, Callable[..., Any]]:
        """No-op shims by tool name, with the real tools' signatures: calling one with the model's arguments
        raises the same TypeError, word for word, that production's tool(**args) raises."""

    def new_turn(self, world: World, case: Case) -> TurnHandler:
        """Fresh per-turn tool state (e.g. agents created earlier in the turn)."""
