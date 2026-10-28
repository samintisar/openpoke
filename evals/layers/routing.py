"""L2 routing eval: one real Interaction Agent turn, with its tools intercepted.

The prompt is assembled with the server's own helpers, so the model sees exactly what
production sends, except that the roster section comes from the system under test.
"""

from __future__ import annotations

import json
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


def render_context(case: Case) -> str:
    parts: list[str] = []
    for index, (tag, text) in enumerate(case.context):
        safe = escape(text, quote=False)
        if tag == "conversation_summary":
            parts.append(f"<conversation_summary>{safe}</conversation_summary>")
        else:
            timestamp = f"2026-10-{24 + index // 2} 10:{index:02d}:00"
            parts.append(f'<{tag} timestamp="{timestamp}">{safe}</{tag}>')
    return "\n".join(parts)


def build_messages(case: Case, world: World, system: RoutingSystem) -> list[dict[str, Any]]:
    sections = [
        _render_conversation_history(render_context(case)),
        system.render_roster(world, case),
        _render_current_turn(case.message, case.message_type),
    ]
    return [{"role": "user", "content": "\n\n".join(sections)}]


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _stub_result(name: str, args: dict[str, Any], world: World, system: RoutingSystem) -> dict[str, Any]:
    """Mirror the payloads the real tools return, without side effects."""
    if name == "send_message_to_agent":
        agent_name = str(args.get("agent_name", ""))
        result = {
            "status": "submitted",
            "agent_name": agent_name,
            "new_agent_created": system.resolve_name(world, agent_name) is None,
        }
    elif name == "send_message_to_user":
        result = {"status": "delivered"}
    elif name == "send_draft":
        result = {"status": "draft_recorded", "to": args.get("to"), "subject": args.get("subject")}
    elif name == "wait":
        result = {"status": "waiting", "reason": args.get("reason")}
    else:
        return {"tool": name, "status": "error", "arguments": args, "error": f"Unknown tool: {name}"}
    return {"tool": name, "status": "success", "arguments": args, "result": result}


def _record(decision: Decision, name: str, args: dict[str, Any], world: World, system: RoutingSystem) -> None:
    if name == "send_message_to_agent":
        agent_name = str(args.get("agent_name", ""))  # raw, like production: no normalization
        agent_id = system.resolve_name(world, agent_name)
        if agent_id is None:
            decision.created_names.append(agent_name)
        elif agent_id not in decision.dispatched_ids:
            decision.dispatched_ids.append(agent_id)
    elif name == "send_message_to_user":
        decision.user_messages.append(str(args.get("message", "")))


async def run_routing_case(
    case: Case, world: World, system: RoutingSystem, llm: LLMClient, *, model: str, sample: int
) -> Decision:
    messages = build_messages(case, world, system)
    system_prompt = build_system_prompt()
    tools = get_tool_schemas()
    decision = Decision()

    for iteration in range(MAX_ITERATIONS):
        response = await llm.complete(model=model, system=system_prompt, messages=messages, tools=tools, sample=sample)
        meta = response.get("_meta", {})
        decision.cost += float(meta.get("cost", 0.0))
        decision.latency_s += float(meta.get("latency_s", 0.0))
        decision.prompt_tokens += int(meta.get("prompt_tokens", 0))
        decision.completion_tokens += int(meta.get("completion_tokens", 0))
        if meta.get("cache_key"):
            decision.cache_keys.append(meta["cache_key"])

        message = (response.get("choices") or [{}])[0].get("message") or {}
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
            args = _parse_arguments(function.get("arguments"))
            decision.tool_calls.append({"name": name, "arguments": args})
            _record(decision, name, args, world, system)
            messages.append({
                "role": "tool",
                "tool_call_id": raw.get("id") or name,
                "content": json.dumps(_stub_result(name, args, world, system), default=str),
            })

    decision.clarified = (
        not decision.dispatched_ids
        and not decision.created_names
        and any("?" in text for text in decision.user_messages)
    )
    return decision
