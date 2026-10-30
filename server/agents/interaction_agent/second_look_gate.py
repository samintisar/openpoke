"""Apply the second look to one `send_message_to_agent` call, before it reaches `dispatch` (Plan 6c).

Shared by the production runtime and the eval. Returns the arguments to send (possibly redirected to another
agent), a held payload when nothing should be sent, and a record of what happened (None when no trigger fired).
A failed or timed-out call never blocks the dispatch: the arguments come back unchanged with an "error" record.
The default reviewer is Jev, split-only (`jev_review.py`): creates are never reviewed, and only a confident "new"
on a reuse holds it. Production builds one `JevClient` (TYPESAFE_API_KEY; without the key every look is an error
record and the call stands, warned about once); the eval passes its own client as `jev`. `cfg.reviewer == "llm"`
asks the LLM reviewer through `complete` instead.
`guard=False` (the eval's no-guard ablation) looks at creates the duplicate guard would have caught, too.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

from ...config import get_settings
from ...logging_config import logger
from ...openrouter_client import request_chat_completion
from ...services.agents.dedup import find_duplicate
from ...services.agents.directory import AgentDirectory
from ...services.agents.jev import JevClient, JevError, api_key_from_env
from ...services.agents.jev_review import AsksJev, jev_review, split_only_effect
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
    guard: bool = True,
    jev: Optional[AsksJev] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    cards = directory.list_cards()
    agent_id = str(arguments.get("agent_id") or "").strip()
    target = directory.get(agent_id) if agent_id else None
    if target is not None:
        proposal, reason = "reuse", reuse_trigger(message, target, cards, now, cfg, rcfg)
        described = f"Send the message to existing agent {target.id} ({target.name})."
    elif not agent_id and arguments.get("name") and arguments.get("purpose"):
        if cfg.reviewer == "jev":
            return arguments, None, None  # Jev only splits reuses; creates are never reviewed
        name, purpose = str(arguments["name"]), str(arguments["purpose"])
        guarded = guard and not _force_new(arguments.get("force_new"))
        if guarded and find_duplicate(name, purpose, cards, now, rcfg):
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
    if cfg.reviewer == "jev":
        return await _jev_look(arguments, record, message=message, context=context, cards=shown, total=len(cards),
                               now=now, target_id=target.id if target else None, cfg=cfg, jev=jev)
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


async def _jev_look(arguments: dict[str, Any], record: dict[str, Any], *, message: str, context: Sequence[str],
                    cards: Sequence[Any], total: int, now: datetime, target_id: Optional[str],
                    cfg: SecondLookConfig, jev: Optional[AsksJev]) -> tuple[dict[str, Any], Optional[dict[str, Any]],
                                                                             dict[str, Any]]:
    """The Jev reviewer on a reuse: hold on a confident "new"; any other answer or failure leaves the call."""
    record = {**record, "reviewer": "jev"}
    client = jev if jev is not None else production_jev(cfg.timeout_s)
    if client is None:
        return arguments, None, {**record, "effect": "error", "error": JEV_NO_KEY, "jev_unavailable": True}
    try:
        verdict = await jev_review(message=message, context=context, cards=cards, jev=client, now=now, total=total,
                                   timeout_s=cfg.timeout_s)
    except JevError as exc:
        return arguments, None, {**record, "effect": "error", "error": str(exc)}
    outcome = split_only_effect(verdict, proposal="reuse", target_id=target_id,
                                min_confidence=cfg.jev_min_confidence)
    record.update(decision=verdict.decision, agent_id=None if verdict.decision != "reuse" else verdict.choice,
                  why=f"jev chose {verdict.choice} at confidence {verdict.confidence:.2f}", effect=outcome.action,
                  jev_choice=verdict.choice, jev_top3=verdict.top(), jev_confidence=verdict.confidence,
                  jev_model=verdict.model, jev_input_tokens=verdict.input_tokens, jev_cost_usd=verdict.cost_usd,
                  jev_min_confidence=cfg.jev_min_confidence)
    if "cached" in verdict.meta:  # the eval's cache: an answer from it cost nothing this run
        record["jev_cached"] = bool(verdict.meta["cached"])
    if outcome.action == "hold":
        return arguments, outcome.payload, record
    return arguments, None, record


JEV_NO_KEY = "TYPESAFE_API_KEY is not set: no Jev second look"
_PRODUCTION_JEV: Optional[JevClient] = None
_NO_KEY_WARNED = False


def production_jev(timeout_s: float) -> Optional[JevClient]:
    """Production's Jev client, made once per process; None without TYPESAFE_API_KEY (warned about once)."""
    global _PRODUCTION_JEV, _NO_KEY_WARNED
    if _PRODUCTION_JEV is None:
        key = api_key_from_env()
        if not key:
            if not _NO_KEY_WARNED:
                _NO_KEY_WARNED = True
                logger.warning("TYPESAFE_API_KEY is not set: the second look's Jev reviewer is off; calls stand")
            return None
        _PRODUCTION_JEV = JevClient(api_key=key, timeout_s=timeout_s)
    return _PRODUCTION_JEV


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


__all__ = ["JEV_NO_KEY", "gate", "openrouter_complete", "production_jev"]
