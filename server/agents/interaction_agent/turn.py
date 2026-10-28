"""The interaction-agent turn in progress: what started it and, for user turns, when the user wrote."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class TurnContext:
    """`origin` is "user" for a user turn, else the inbox kind ("agent_result", "trigger_result", "watcher")
    or "agent" for a mix of them. `user_message_at` (timezone-aware UTC) is when the newest user message of
    the turn was posted; None for non-user turns."""

    origin: str
    user_message_at: Optional[datetime] = None


# Set by InteractionAgentRuntime for the duration of each turn so IA tools can tell who started it.
current_turn: contextvars.ContextVar[Optional[TurnContext]] = contextvars.ContextVar("current_turn", default=None)


__all__ = ["TurnContext", "current_turn"]
