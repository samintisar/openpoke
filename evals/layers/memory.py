"""L2 memory eval: does an execution agent still recall its matter's facts after compaction?

Per agent, two conditions answer the same questions:
- full: the whole raw transcript, as production renders it without a memory;
- compacted: the production `compact` folds the over-budget prefix into a memory, then `render_history` gives the
  memory, the identifiers and the raw tail, as `ExecutionAgent.build_system_prompt_with_history` does.

QA is one plain call per question, with the execution agent's real system prompt plus that history. When compaction
fails, production keeps the full history, so the compacted condition is scored on the full history (same answers)
and the failure is counted separately.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any

from server.agents.execution_agent.agent import SYSTEM_PROMPT_TEMPLATE
from server.services.agents.memory import (
    CompactionError,
    CompactionResult,
    compact,
    estimate_tokens,
    extract_identifiers,
    render_entries,
    render_history,
    split_for_budget,
)

from ..llm import LLMClient, RunAborted, ScriptedLLM
from ..worlds.memory_gen import FACT_KINDS, AgentHistory, Fact

CONDITIONS = ("full", "compacted")
QA_PROMPT = "Answer from your execution history only, as briefly as possible: {question}"


def normalize(text: str) -> str:
    """Lowercase with every non-alphanumeric character ($ , . - @ whitespace ...) removed."""
    return re.sub(r"[\W_]+", "", (text or "").lower())


def is_recalled(answer: str, gold: Sequence[str]) -> bool:
    answer_norm = normalize(answer)
    return any(normalize(option) and normalize(option) in answer_norm for option in gold)


def system_prompt(name: str, purpose: str, history: str) -> str:
    """The execution agent's prompt with its history, built as production builds it (no registry lookup)."""
    base = SYSTEM_PROMPT_TEMPLATE.format(agent_name=name, agent_purpose=purpose)
    return f"{base}\n\n# Execution History\n\n{history}"


def _content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    return (message.get("content") or "").strip()


class _Completer:
    """`complete(system, user)` for the production `compact`, backed by the eval client.

    `compact` wraps every exception in CompactionError, so a run-ending error is kept here and re-raised by the caller.
    """

    def __init__(self, llm: LLMClient, model: str) -> None:
        self.llm, self.model = llm, model
        self.metas: list[dict[str, Any]] = []
        self.aborted: RunAborted | None = None

    async def __call__(self, system: str, user: str) -> str:
        try:
            data = await self.llm.complete(
                model=self.model, system=system, messages=[{"role": "user", "content": user}], tools=[], sample=0
            )
        except RunAborted as exc:
            self.aborted = exc
            raise
        self.metas.append(data.get("_meta") or {})
        content = _content(data)
        if not content:
            raise CompactionError("compaction response has no content")
        return content


def _meta_sums(metas: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cost": sum(float(m.get("cost", 0.0)) for m in metas),
        "fresh_cost": sum(float(m.get("cost", 0.0)) for m in metas if not m.get("cached")),
        "prompt_tokens": sum(int(m.get("prompt_tokens", 0)) for m in metas),
        "completion_tokens": sum(int(m.get("completion_tokens", 0)) for m in metas),
        "cached_calls": sum(1 for m in metas if m.get("cached")),
        "cache_keys": [m.get("cache_key", "") for m in metas],
    }


async def _ask(llm: LLMClient, model: str, system: str, fact: Fact) -> dict[str, Any]:
    data = await llm.complete(
        model=model, system=system, messages=[{"role": "user", "content": QA_PROMPT.format(question=fact.question)}],
        tools=[], sample=0,
    )
    meta = data.get("_meta") or {}
    answer = _content(data)
    correct = is_recalled(answer, fact.gold)
    return {
        "kind": fact.kind, "question": fact.question, "gold": list(fact.gold), "answer": answer, "correct": correct,
        # a wrong answer that gives a superseded or confusable value (e.g. the earlier amount)
        "stale": not correct and bool(fact.distractors) and is_recalled(answer, fact.distractors),
        "prompt_tokens": int(meta.get("prompt_tokens", 0)), "cost": float(meta.get("cost", 0.0)),
        "cached": bool(meta.get("cached")), "cache_key": meta.get("cache_key", ""),
    }


async def _condition(llm: LLMClient, model: str, system: str, history: str, facts: Sequence[Fact]) -> dict[str, Any]:
    answers = await asyncio.gather(*(_ask(llm, model, system, fact) for fact in facts))
    return {"history_tokens": estimate_tokens(history), "system_tokens": estimate_tokens(system), "facts": list(answers)}


async def run_memory_agent(agent: AgentHistory, llm: LLMClient, *, model: str, budget_tokens: int) -> dict[str, Any]:
    """Compact one agent's history, then ask every fact question under both conditions."""
    entries = list(agent.entries)
    full_history = render_entries(entries)
    prefix, tail = split_for_budget(entries, budget_tokens)
    prefix_tokens = estimate_tokens(render_entries(prefix))
    identifiers = extract_identifiers(entry.text for entry in prefix)

    completer = _Completer(llm, model)
    result: CompactionResult | None = None
    status, error = "skipped", None  # production only compacts a prefix over the budget
    if prefix_tokens > budget_tokens:
        try:
            result = await compact(prefix, "", agent.purpose, complete=completer)
            status = "ok"
        except CompactionError as exc:
            if completer.aborted is not None:
                raise completer.aborted from exc
            status, error = "failed", str(exc)

    full = await _condition(llm, model, system_prompt(agent.name, agent.purpose, full_history), full_history, agent.facts)
    if result is not None:
        history = render_history(result.memory, identifiers, tail)
        # Compaction refreshes the card's purpose, which the agent's prompt then uses.
        compacted = await _condition(
            llm, model, system_prompt(agent.name, result.purpose, history), history, agent.facts
        )
        compacted["fallback"] = False
    else:
        compacted = {**full, "fallback": True}  # same prompt as full, so the same answers

    return {
        "agent_id": agent.agent_id, "name": agent.name, "purpose": agent.purpose, "error": None,
        "entries": len(entries), "prefix_entries": len(prefix), "tail_entries": len(tail),
        "full_history_tokens": estimate_tokens(full_history), "prefix_tokens": prefix_tokens,
        "compacted_history_tokens": compacted["history_tokens"],
        "identifiers": identifiers,
        "compaction": {
            "status": status, "error": error,
            "memory": result.memory if result else None, "purpose": result.purpose if result else None,
            "last_status": result.last_status if result else None,
            "memory_tokens": estimate_tokens(result.memory) if result else None,
            **_meta_sums(completer.metas),
        },
        "conditions": {"full": full, "compacted": compacted},
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize_memory(rows: list[dict[str, Any]], *, planned: int) -> dict[str, Any]:
    done = [row for row in rows if row.get("error") is None]
    summary: dict[str, Any] = {
        "planned_agents": planned, "agents": len(done), "errors": len(rows) - len(done),
        "incomplete_agents": planned - len(rows),
    }
    for condition in CONDITIONS:
        answers = [a for row in done for a in row["conditions"][condition]["facts"]]
        # QA calls actually made: a fallback reuses the full condition's answers
        calls = [a for row in done if not row["conditions"][condition].get("fallback") for a in
                 row["conditions"][condition]["facts"]]
        summary[condition] = {
            "recall": _mean([float(a["correct"]) for a in answers]),
            "by_kind": {kind: _mean([float(a["correct"]) for a in answers if a["kind"] == kind]) for kind in FACT_KINDS},
            "stale_rate": _mean([float(a["stale"]) for a in answers]),
            "mean_history_tokens": _mean([row["conditions"][condition]["history_tokens"] for row in done]),
            "mean_prompt_tokens": _mean([a["prompt_tokens"] for a in calls]),
            "qa_cost_usd": round(sum(a["cost"] for a in calls), 4),
        }
    summary["compacted"]["fallback_agents"] = sum(1 for row in done if row["conditions"]["compacted"].get("fallback"))
    compactions = [row["compaction"] for row in done]
    summary["compaction"] = {
        "calls": sum(1 for c in compactions if c["status"] != "skipped"),
        "failures": sum(1 for c in compactions if c["status"] == "failed"),
        "skipped": sum(1 for c in compactions if c["status"] == "skipped"),
        "cost_usd": round(sum(c["cost"] for c in compactions), 4),
        "mean_prefix_tokens": _mean([row["prefix_tokens"] for row in done]),
        "mean_memory_tokens": _mean([c["memory_tokens"] for c in compactions if c["memory_tokens"] is not None]),
    }
    summary["total_cost_usd"] = round(
        summary["compaction"]["cost_usd"] + sum(summary[c]["qa_cost_usd"] for c in CONDITIONS), 4
    )
    return summary


def fake_memory_llm() -> ScriptedLLM:
    """Offline smoke model: a fixed, valid compaction and an empty QA answer."""

    def script(messages: list[dict[str, Any]], sample: int) -> dict[str, Any]:
        if "<new_entries>" in messages[-1]["content"]:
            return {"content": json.dumps({
                "memory": "Offline fake memory: no facts kept.", "purpose": "Offline fake purpose.",
                "last_status": "Offline fake status.",
            })}
        return {"content": "I don't know."}

    return ScriptedLLM(script)


__all__ = [
    "CONDITIONS", "QA_PROMPT", "fake_memory_llm", "is_recalled", "normalize", "run_memory_agent", "summarize_memory",
    "system_prompt",
]
