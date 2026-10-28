"""L2 routing eval: one real Interaction Agent turn, with its tools intercepted.

The prompt is assembled with the server's own helpers, so the model sees exactly what
production sends, except that the roster section comes from the system under test.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from html import escape
from typing import Any

from server.agents.interaction_agent.agent import (
    _render_conversation_history,
    _render_current_turn,
    build_system_prompt,
)
from server.agents.interaction_agent.tools import get_tool_schemas

from ..cases import Case
from ..grading import Decision
from ..llm import LLMClient
from ..systems.base import RoutingSystem
from ..worlds.model import World

# Production allows 8 iterations; routing decisions happen in the first few.
MAX_ITERATIONS = 4


CONTEXT_BASE_TIME = datetime(2026, 10, 24, 10, 0, 0)


def render_context(case: Case) -> str:
    parts: list[str] = []
    for index, (tag, text) in enumerate(case.context):
        safe = escape(text, quote=False)
        if tag == "conversation_summary":
            parts.append(f"<conversation_summary>{safe}</conversation_summary>")
        else:
            stamp = CONTEXT_BASE_TIME + timedelta(days=index // 2, minutes=index)
            parts.append(f'<{tag} timestamp="{stamp:%Y-%m-%d %H:%M:%S}">{safe}</{tag}>')
    return "\n".join(parts)


def build_messages(case: Case, world: World, system: RoutingSystem) -> list[dict[str, Any]]:
    sections = [
        _render_conversation_history(render_context(case)),
        system.render_roster(world, case),
        _render_current_turn(case.message, case.message_type),
    ]
    return [{"role": "user", "content": "\n\n".join(sections)}]


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Arguments as a dict plus an error, mirroring production's _parse_tool_arguments."""
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        if not raw.strip():
            return {}, None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"invalid json: {exc}"
        if isinstance(parsed, dict):
            return parsed, None
        return {}, "decoded arguments were not an object"
    return {}, f"unsupported argument type: {type(raw).__name__}"


def _schema_error(name: str, args: dict[str, Any]) -> str | None:
    """The error production's handle_tool_call gives when calling tool(**args) raises TypeError."""
    for schema in get_tool_schemas():
        function = schema["function"]
        if function["name"] != name:
            continue
        parameters = function.get("parameters", {})
        missing = [key for key in parameters.get("required", []) if key not in args]
        unknown = [key for key in args if key not in parameters.get("properties", {})]
        if unknown:
            return f"Missing required arguments: {name}() got an unexpected keyword argument {unknown[0]!r}"
        if missing:
            listed = ", ".join(repr(key) for key in missing)
            return f"Missing required arguments: {name}() missing {len(missing)} required argument(s): {listed}"
        return None
    return f"Unknown tool: {name}"


def _error_result(name: str, args: dict[str, Any], error: str) -> dict[str, Any]:
    """Production's _format_tool_result for a failed ToolResult(payload={"error": ...})."""
    return {"tool": name, "status": "error", "arguments": args, "error": {"error": error}}


class _Turn:
    """Per-run tool state: the stub has to remember agents created earlier in the turn."""

    def __init__(self, world: World, system: RoutingSystem, decision: Decision) -> None:
        self.world = world
        self.system = system
        self.decision = decision
        self.created: set[str] = set()

    def handle(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Record a valid call's effect and return the stubbed tool result."""
        if name == "send_message_to_agent":
            agent_name = str(args.get("agent_name", ""))  # raw, like production: no normalization
            agent_id = self.system.resolve_name(self.world, agent_name)
            is_new = agent_id is None and agent_name not in self.created
            if agent_id is None:
                if is_new:
                    self.created.add(agent_name)
                    self.decision.created_names.append(agent_name)
            elif agent_id not in self.decision.dispatched_ids:
                self.decision.dispatched_ids.append(agent_id)
            result = {"status": "submitted", "agent_name": agent_name, "new_agent_created": is_new}
        elif name == "send_message_to_user":
            self.decision.user_messages.append(str(args.get("message", "")))
            result = {"status": "delivered"}
        elif name == "send_draft":
            result = {"status": "draft_recorded", "to": args.get("to"), "subject": args.get("subject")}
        else:  # wait; unknown names never reach here
            result = {"status": "waiting", "reason": args.get("reason")}
        return {"tool": name, "status": "success", "arguments": args, "result": result}


async def run_routing_case(
    case: Case, world: World, system: RoutingSystem, llm: LLMClient, *, model: str, sample: int
) -> Decision:
    messages = build_messages(case, world, system)
    system_prompt = build_system_prompt()
    tools = get_tool_schemas()
    decision = Decision()
    turn = _Turn(world, system, decision)
    last_assistant_text = ""

    for iteration in range(MAX_ITERATIONS):
        response = await llm.complete(model=model, system=system_prompt, messages=messages, tools=tools, sample=sample)
        meta = response.get("_meta", {})
        cost = float(meta.get("cost", 0.0))
        decision.cost += cost
        if meta.get("cached"):
            decision.cached_calls += 1
        else:
            decision.fresh_cost += cost
        decision.latency_s += float(meta.get("latency_s", 0.0))
        decision.prompt_tokens += int(meta.get("prompt_tokens", 0))
        decision.completion_tokens += int(meta.get("completion_tokens", 0))
        if meta.get("cache_key"):
            decision.cache_keys.append(meta["cache_key"])

        message = (response.get("choices") or [{}])[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if content:
            last_assistant_text = content
        raw_calls = message.get("tool_calls") or []
        entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
        if raw_calls:
            entry["tool_calls"] = raw_calls
        messages.append(entry)
        decision.iterations = iteration + 1
        if not raw_calls:
            break

        for raw in raw_calls:
            function = raw.get("function") or {}
            name = function.get("name") or ""
            args, error = _parse_arguments(function.get("arguments"))
            error = error or _schema_error(name, args)
            if error:
                decision.tool_calls.append({"name": name, "arguments": args, "error": error})
                result = _error_result(name, args, error)
            else:
                decision.tool_calls.append({"name": name, "arguments": args})
                result = turn.handle(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": raw.get("id") or name,
                "content": json.dumps(result, default=str),
            })
    else:
        decision.hit_iteration_cap = True  # production raises here after its own, higher cap

    # Production's _finalize_response shows the last assistant text when no message was sent.
    if not decision.user_messages and last_assistant_text:
        decision.user_messages.append(last_assistant_text)
    decision.clarified = (
        not decision.dispatched_ids
        and not decision.created_names
        and bool(decision.user_messages)
        and "?" in decision.user_messages[-1]
    )
    return decision
