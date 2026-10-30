"""Plan 2 routing: agent cards, retrieval of the most relevant few, and the dedup-guarded dispatch tool.

Everything that decides routing is production code from `server/services/agents`, run over an
in-memory directory built from the world, so the eval measures what the server runs. The prompt is
the server's `build_system_prompt()` and the tool schemas are `schemas.TOOL_SCHEMAS`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from typing import Any

from server.agents.interaction_agent.agent import build_system_prompt
from server.agents.interaction_agent.second_look_gate import gate
from server.services.agents.second_look import DEFAULT_SECOND_LOOK
from server.agents.interaction_agent.tools import SEND_DRAFT_ARGUMENTS_ERROR
from server.services.agents.cards import AgentCard, as_utc
from server.services.agents.directory import InMemoryDirectory
from server.services.agents.dispatch import SEARCH_BUDGET, SEARCH_BUDGET_EXHAUSTED, dispatch, search_payload
from server.services.agents.render import render_cards
from server.services.agents.retrieval import (
    RetrievalConfig,
    build_query,
    context_query,
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
# The eval's second look waits out the client's own timeout and retries, so results never depend on wall-clock.
EVAL_SECOND_LOOK = replace(DEFAULT_SECOND_LOOK, timeout_s=180.0)
SEARCH_TOOL = "search_agents"


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


def without_tool_lines(prompt: str, tool: str) -> str:
    """The prompt minus every line that names `tool`: strips a removed tool's rule, nothing else."""
    return "\n".join(line for line in prompt.split("\n") if f"`{tool}`" not in line)


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

    def __init__(
        self, world: World, cards: list[AgentCard], cfg: RetrievalConfig, guard: bool = True,
        case: Case | None = None, second_look: bool = False, search_budget: int | None = None,
    ) -> None:
        self.world = world
        self.cfg = cfg
        self.guard = guard
        self.case = case
        self.second_look = second_look
        self.second_look_cfg = EVAL_SECOND_LOOK
        self.search_budget = search_budget  # None: unlimited
        self._searches = 0
        self.directory = InMemoryDirectory(cards, last_routed=_last_routed(cards, world))
        self.created_ids: set[str] = set()
        self._llm: Any = None  # bound by the routing layer: the second look uses the run's client
        self._model = ""
        self._sample = 0
        self._looked = False  # at most one second look per turn

    def bind_llm(self, llm: Any, *, model: str, sample: int) -> None:
        self._llm, self._model, self._sample = llm, model, sample

    async def handle(self, name: str, args: dict[str, Any], decision: Decision) -> dict[str, Any]:
        """Record a valid call's effect and return the tool result production would give."""
        if name == "send_message_to_agent":
            return await self._send_to_agent(name, args, decision)
        if name == "search_agents":
            if self.search_budget is not None and self._searches >= self.search_budget:
                return _error(name, args, SEARCH_BUDGET_EXHAUSTED)
            self._searches += 1
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

    async def _send_to_agent(self, name: str, args: dict[str, Any], decision: Decision) -> dict[str, Any]:
        if (self.second_look and not self._looked and self._llm is not None and self.case is not None
                and self.case.message_type == "user"):
            context = [entry[1] for entry in self.case.context[-QUERY_CONTEXT_ENTRIES:]]
            args, held, record = await gate(dict(args), message=self.case.message, context=context,
                                            directory=self.directory, now=self.world.now,
                                            complete=self._complete(decision), cfg=self.second_look_cfg,
                                            rcfg=self.cfg, guard=self.guard)
            if record is not None:
                self._looked = True
                decision.second_looks.append(record)
            if held is not None:
                return _error(name, args, held)
        result = dispatch(
            self.directory,
            instructions=str(args.get("instructions", "")),
            agent_id=args.get("agent_id"),
            name=args.get("name"),
            purpose=args.get("purpose"),
            # No guard: every create goes through as if force_new, so none returns possible_duplicate.
            force_new=bool(args.get("force_new", False)) or not self.guard,
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

    def _complete(self, decision: Decision):
        """The second look's model call, through the run's client, accounted on the decision like an IA call."""
        async def complete(system: str, user: str) -> str:
            response = await self._llm.complete(model=self._model, system=system,
                                                messages=[{"role": "user", "content": user}], tools=[],
                                                sample=self._sample)
            meta = response.get("_meta", {})
            cost = float(meta.get("cost", 0.0))
            decision.cost += cost
            if meta.get("cached"):
                decision.cached_calls += 1
            else:
                decision.fresh_cost += cost
            decision.latency_s += float(meta.get("latency_s", 0.0))
            decision.prompt_tokens += int(meta.get("prompt_tokens", 0))
            decision.cached_prompt_tokens += int(meta.get("cached_tokens", 0))
            decision.completion_tokens += int(meta.get("completion_tokens", 0))
            if meta.get("cache_key"):
                decision.cache_keys.append(meta["cache_key"])
            message = ((response.get("choices") or [{}])[0].get("message") or {})
            return (message.get("content") or "").strip()
        return complete


_STATUS_SEARCH_CLAUSE = ", or with `search_agents` if none fits"


class RetrievedSystem:
    """Shows the cards `select` picks for the turn: pinned, hot, then the top K by relevance."""

    name = "retrieved"

    def __init__(
        self, cfg: RetrievalConfig | None = None, lifecycle: bool = True, guard: bool = True, search: bool = True,
        summaries: bool = True, status_policy: bool = True, second_look: bool = True,
        context_query: bool = False, search_budget: int | None = SEARCH_BUDGET,
    ) -> None:
        self.cfg = cfg or RetrievalConfig()
        # False keeps every card active, to ablate the lifecycle (dormant / archived statuses).
        self.lifecycle = lifecycle
        # Eval-only ablation switches; the defaults are production behaviour.
        self.guard = guard  # False: creates skip the duplicate guard
        self.search = search  # False: no search_agents tool, no prompt rule for it, and no search clause in the status rule
        self.summaries = summaries  # False: cards carry no living summary (the summaries ablation)
        self.status_policy = status_policy  # False: no status-question rule
        self.second_look = second_look  # False: no second look before risky dispatches
        self.context_query = context_query  # off by default: no gold-card gain on dev (Plan 6c Task 10)
        self.search_budget = search_budget  # searches per turn before a refusal; None = unlimited
        # id(world) -> (world, cards); holding the world keeps its id from being reused.
        self._cards_cache: dict[int, tuple[World, list[AgentCard]]] = {}

    def cards(self, world: World) -> list[AgentCard]:
        """The world's cards, built once per world (every case and turn shares them; directories copy)."""
        cached = self._cards_cache.get(id(world))
        if cached is None or cached[0] is not world:
            cached = self._cards_cache[id(world)] = (world, cards_for(world, self.lifecycle, self.summaries))
        return cached[1]

    def selected(self, world: World, case: Case | None) -> list[AgentCard]:
        cards = self.cards(world)
        pinned: list[AgentCard] = []
        if case is not None and case.message_type == "agent":
            pinned = pinned_from_agent_message(case.message, cards)
        extra = None
        if self.context_query and case is not None and case.message_type == "user":
            texts = [entry[1] for entry in case.context[-QUERY_CONTEXT_ENTRIES:]]
            extra = context_query(case.message, texts, cards, world.now, self.cfg, _last_routed(cards, world))
        return select(_query(case), cards, world.now, _last_routed(cards, world), self.cfg, pinned=pinned,
                      context_query=extra)

    def render_roster(self, world: World, case: Case | None) -> str:
        return render_cards(self.selected(world, case), len(self.cards(world)), world.now)

    def shown_ids(self, world: World, case: Case | None) -> list[str]:
        return [card.id for card in self.selected(world, case)]

    def system_prompt(self) -> str:
        prompt = build_system_prompt(status_policy=self.status_policy)
        if not self.search:
            if self.status_policy:
                if _STATUS_SEARCH_CLAUSE not in prompt:
                    raise ValueError("the status rule's search clause moved; update _STATUS_SEARCH_CLAUSE")
                prompt = prompt.replace(_STATUS_SEARCH_CLAUSE, "")
            return without_tool_lines(prompt, SEARCH_TOOL)
        return prompt

    def tool_schemas(self) -> list[dict[str, Any]]:
        if not self.search:
            return [schema for schema in TOOL_SCHEMAS if schema["function"]["name"] != SEARCH_TOOL]
        return TOOL_SCHEMAS

    def tool_signatures(self) -> dict[str, Callable[..., Any]]:
        if not self.search:
            # Without its shim the routing layer treats a search_agents call as an unknown tool.
            return {name: f for name, f in _SHIMS.items() if name != SEARCH_TOOL}
        return _SHIMS

    def new_turn(self, world: World, case: Case) -> RetrievedTurn:
        return RetrievedTurn(world, self.cards(world), self.cfg, self.guard, case=case, second_look=self.second_look,
                             search_budget=self.search_budget)


class RetrievedNoLifecycleSystem(RetrievedSystem):
    """`retrieved` with every card active: the lifecycle ablation."""

    name = "retrieved_nolife"

    def __init__(self, cfg: RetrievalConfig | None = None, **options: Any) -> None:
        super().__init__(cfg, lifecycle=False, **options)


class RetrievedNoGuardSystem(RetrievedSystem):
    """`retrieved` without the duplicate guard: creates never return possible_duplicate."""

    name = "retrieved_noguard"

    def __init__(self, cfg: RetrievalConfig | None = None, **options: Any) -> None:
        super().__init__(cfg, guard=False, **options)


class RetrievedNoSearchSystem(RetrievedSystem):
    """`retrieved` without `search_agents`: no tool schema, no shim, no prompt rule."""

    name = "retrieved_nosearch"

    def __init__(self, cfg: RetrievalConfig | None = None, **options: Any) -> None:
        super().__init__(cfg, search=False, **options)


class RetrievedNoFuzzySystem(RetrievedSystem):
    """`retrieved` with exact-only entity matching: no typo matching in retrieval, search or dedup."""

    name = "retrieved_nofuzzy"

    def __init__(self, cfg: RetrievalConfig | None = None, **options: Any) -> None:
        super().__init__(replace(cfg or RetrievalConfig(), fuzzy=False), **options)


class RetrievedNoLastRoutedSystem(RetrievedSystem):
    """`retrieved` without the last-routed boost."""

    name = "retrieved_nolastrouted"

    def __init__(self, cfg: RetrievalConfig | None = None, **options: Any) -> None:
        super().__init__(replace(cfg or RetrievalConfig(), last_routed_boost=0.0), **options)


__all__ = [
    "RetrievedNoFuzzySystem",
    "RetrievedNoGuardSystem",
    "RetrievedNoLastRoutedSystem",
    "RetrievedNoLifecycleSystem",
    "RetrievedNoSearchSystem",
    "RetrievedSystem",
    "RetrievedTurn",
    "without_tool_lines",
]
