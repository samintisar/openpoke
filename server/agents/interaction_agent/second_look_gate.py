"""Apply the second look to one `send_message_to_agent` call, before it reaches `dispatch` (Plan 6c).

Shared by the production runtime and the eval. Returns the arguments to send (possibly redirected to another
agent), a held payload when nothing should be sent, and a record of what happened (None when no trigger fired).
A failed or timed-out call never blocks the dispatch: the arguments come back unchanged with an "error" record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

from ...config import get_settings
from ...openrouter_client import request_chat_completion
from ...services.agents.dedup import find_duplicate
from ...services.agents.directory import AgentDirectory
from ...services.agents.retrieval import RetrievalConfig
from ...services.agents.second_look import (
    DEFAULT_SECOND_LOOK,
    TOP_CARDS,
    Complete,
    SecondLookConfig,
    SecondLookError,
    create_trigger,
    effect,
    reuse_trigger,
    review,
    top_cards,
)


def _force_new(value: Any) -> bool:
    """dispatch's reading of `force_new`."""
    return value is True or str(value).strip().lower() == "true"


async def gate(
    arguments: dict[str, Any],
    *,
    message: str,
    context: Sequence[str],
    directory: AgentDirectory,
    now: datetime,
    complete: Complete,
    cfg: SecondLookConfig = DEFAULT_SECOND_LOOK,
    rcfg: Optional[RetrievalConfig] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    cards = directory.list_cards()
    agent_id = str(arguments.get("agent_id") or "").strip()
    target = directory.get(agent_id) if agent_id else None
    if target is not None:
        proposal, reason = "reuse", reuse_trigger(message, target, cards, now, cfg, rcfg)
        described = f"Send the message to existing agent {target.id} ({target.name})."
    elif not agent_id and arguments.get("name") and arguments.get("purpose"):
        name, purpose = str(arguments["name"]), str(arguments["purpose"])
        if not _force_new(arguments.get("force_new")) and find_duplicate(name, purpose, cards, now, rcfg):
            return arguments, None, None  # the duplicate guard answers first, for free
        proposal, reason = "create", create_trigger(message, cards, now, cfg, rcfg)
        described = f"Create a new agent named {name!r} (purpose: {purpose})."
    else:
        return arguments, None, None  # unknown ids and malformed calls are dispatch's to report
    if reason is None:
        return arguments, None, None

    record: dict[str, Any] = {"trigger": proposal, "reason": reason}
    shown = [card for card, _ in top_cards(message, cards, now, rcfg)]
    if target is not None and all(card.id != target.id for card in shown):
        shown = [target, *shown[: TOP_CARDS - 1]]
    try:
        verdict = await review(message=message, context=context, cards=shown, proposal=described, complete=complete,
                               now=now, total=len(cards), timeout_s=cfg.timeout_s)
    except SecondLookError as exc:
        return arguments, None, {**record, "effect": "error", "error": str(exc)}
    outcome = effect(verdict, proposal=proposal, target_id=target.id if target else None)
    record.update(decision=verdict.decision, agent_id=verdict.agent_id, why=verdict.reason, effect=outcome.action)
    if outcome.action == "hold":
        return arguments, outcome.payload, record
    if outcome.action == "redirect":
        return {"agent_id": outcome.agent_id, "instructions": arguments.get("instructions", "")}, None, record
    return arguments, None, record


async def openrouter_complete(system: str, user: str) -> str:
    """Production's second-look call: the cheap-tier model, like the other auxiliary calls."""
    settings = get_settings()
    response = await request_chat_completion(
        model=settings.summarizer_model,
        messages=[{"role": "user", "content": user}],
        system=system,
        api_key=settings.openrouter_api_key,
    )
    choices = response.get("choices") or []
    return ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""


__all__ = ["gate", "openrouter_complete"]
