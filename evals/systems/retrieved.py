"""Plan 2 routing: agent cards, retrieval of the most relevant few, and the dedup-guarded dispatch tool.

Everything that decides routing is production code from `server/services/agents`, run over an
in-memory directory built from the world, so the eval measures what the server runs. The prompt is
the server's `build_system_prompt()` and the tool schemas are `schemas.TOOL_SCHEMAS`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from server.agents.interaction_agent.agent import build_system_prompt
from server.agents.interaction_agent.tools import SEND_DRAFT_ARGUMENTS_ERROR
from server.services.agents.cards import AgentCard, as_utc
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.dispatch import dispatch, search_payload
from server.services.agents.render import render_cards
from server.services.agents.retrieval import (
    RetrievalConfig,
    build_query,
    pinned_from_agent_message,
    select,
)
from server.services.agents.schemas import TOOL_SCHEMAS

from ..cases import Case
from ..grading import Decision
from ..worlds.model import World
from .cards_from_world import cards_for

# Context entries fed to build_query: the last 3 turns, a user and an assistant entry each.
QUERY_CONTEXT_ENTRIES = 6
# Stand-in for the previous turn's routing: agents active this recently.
LAST_ROUTED_WINDOW = timedelta(hours=1)


# No-op shims with the new tools' signatures and names: calling one with the model's arguments
# raises the same TypeError, word for word, that production's tool(**args) raises.
def send_message_to_agent(
    instructions: str,
    agent_id: str | None = None,
    name: str | None = None,
    purpose: str | None = None,
    force_new: bool = False,
) -> None:
    return None


def search_agents(query: str) -> None:
    return None


def send_message_to_user(message: str) -> None:
    return None


def send_draft(
    to: str | None = None, subject: str | None = None, body: str | None = None, action_id: Any = None
) -> None:
    if action_id is None and not (to and subject and body):
        raise TypeError(SEND_DRAFT_ARGUMENTS_ERROR)
    return None


def approve_send(action_id: Any) -> None:
    return None


def approve_recurring_send(trigger_id: Any, recipients: Any) -> None:
    return None


def wait(reason: str) -> None:
    return None


_SHIMS = {
    f.__name__: f
    for f in (send_message_to_agent, search_agents, send_message_to_user, send_draft, approve_send, approve_recurring_send, wait)
}


def _query(case: Case | None) -> str:
    if case is None:
        return ""
    texts = [entry[1] for entry in case.context[-QUERY_CONTEXT_ENTRIES:]]
    return build_query(case.message, texts)


def _last_routed(cards: list[AgentCard], world: World) -> list[str]:
    now = as_utc(world.now)
    return [card.id for card in cards if now - as_utc(card.last_active_at) <= LAST_ROUTED_WINDOW]


def _success(name: str, args: dict[str, Any], result: Any) -> dict[str, Any]:
    return {"tool": name, "status": "success", "arguments": args, "result": result}


def _error(name: str, args: dict[str, Any], payload: Any) -> dict[str, Any]:
    """Production's _format_tool_result for a failed ToolResult: the payload goes under "error"."""
    return {"tool": name, "status": "error", "arguments": args, "error": payload}


class RetrievedTurn:
    """Per-run tool state: a fresh directory over the world's cards, so creates are seen within the turn."""

    def __init__(self, world: World, cards: list[AgentCard], cfg: RetrievalConfig) -> None:
        self.world = world
        self.cfg = cfg
        self.directory = InMemoryDirectory(cards, last_routed=_last_routed(cards, world))
        self.created_ids: set[str] = set()

    def handle(self, name: str, args: dict[str, Any], decision: Decision) -> dict[str, Any]:
        """Record a valid call's effect and return the tool result production would give."""
        if name == "send_message_to_agent":
            return self._send_to_agent(name, args, decision)
        if name == "search_agents":
            payload = search_payload(self.directory, str(args.get("query", "")), self.world.now, self.cfg)
            return _success(name, args, payload)
        if name == "send_message_to_user":
            decision.user_messages.append(str(args.get("message", "")))
            result: dict[str, Any] = {"status": "delivered"}
        elif name == "approve_send":
            # Routing worlds have no pending sends, so production's gate refuses every id.
            return _error(name, args, f"No pending send with action_id {args.get('action_id')!r}")
        elif name == "approve_recurring_send":
            # Nor triggers.
            return _error(name, args, f"Unknown trigger {args.get('trigger_id')!r}.")
        elif name == "send_draft":
            result = {"status": "draft_recorded", "to": args.get("to"), "subject": args.get("subject")}
        else:  # wait; unknown names never reach here
            result = {"status": "waiting", "reason": args.get("reason")}
        return _success(name, args, result)

    def _send_to_agent(self, name: str, args: dict[str, Any], decision: Decision) -> dict[str, Any]:
        result = dispatch(
            self.directory,
            instructions=str(args.get("instructions", "")),
            agent_id=args.get("agent_id"),
            name=args.get("name"),
            purpose=args.get("purpose"),
            force_new=bool(args.get("force_new", False)),
            now=self.world.now,
            cfg=self.cfg,
        )
        if result.action == "created":
            self.created_ids.add(result.card.id)
            decision.created_names.append(result.card.name)
        elif result.action == "reused":
            # Reusing an agent created earlier in this turn is still that one new agent.
            agent_id = result.card.id
            if agent_id not in self.created_ids and agent_id not in decision.dispatched_ids:
                decision.dispatched_ids.append(agent_id)
        else:  # possible_duplicate or error: nothing was sent
            return _error(name, args, result.payload)
        return _success(name, args, result.payload)


class RetrievedSystem:
    """Shows the cards `select` picks for the turn: pinned, hot, then the top K by relevance."""

    name = "retrieved"

    def __init__(self, cfg: RetrievalConfig | None = None, lifecycle: bool = True) -> None:
        self.cfg = cfg or RetrievalConfig()
        # False keeps every card active, to ablate the lifecycle (dormant / archived statuses).
        self.lifecycle = lifecycle
        # id(world) -> (world, cards); holding the world keeps its id from being reused.
        self._cards_cache: dict[int, tuple[World, list[AgentCard]]] = {}

    def cards(self, world: World) -> list[AgentCard]:
        """The world's cards, built once per world (every case and turn shares them; directories copy)."""
        cached = self._cards_cache.get(id(world))
        if cached is None or cached[0] is not world:
            cached = self._cards_cache[id(world)] = (world, cards_for(world, self.lifecycle))
        return cached[1]

    def selected(self, world: World, case: Case | None) -> list[AgentCard]:
        cards = self.cards(world)
        pinned: list[AgentCard] = []
        if case is not None and case.message_type == "agent":
            pinned = pinned_from_agent_message(case.message, cards)
        return select(_query(case), cards, world.now, _last_routed(cards, world), self.cfg, pinned=pinned)

    def render_roster(self, world: World, case: Case | None) -> str:
        return render_cards(self.selected(world, case), len(self.cards(world)), world.now)

    def shown_ids(self, world: World, case: Case | None) -> list[str]:
        return [card.id for card in self.selected(world, case)]

    def system_prompt(self) -> str:
        return build_system_prompt()

    def tool_schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def tool_signatures(self) -> dict[str, Callable[..., Any]]:
        return _SHIMS

    def new_turn(self, world: World, case: Case) -> RetrievedTurn:
        return RetrievedTurn(world, self.cards(world), self.cfg)


class RetrievedNoLifecycleSystem(RetrievedSystem):
    """`retrieved` with every card active: the lifecycle ablation."""

    name = "retrieved_nolife"

    def __init__(self, cfg: RetrievalConfig | None = None) -> None:
        super().__init__(cfg, lifecycle=False)


__all__ = ["RetrievedNoLifecycleSystem", "RetrievedSystem", "RetrievedTurn"]
