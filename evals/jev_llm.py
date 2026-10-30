"""Jev access for evals: a disk cache keyed by the request body, a cost meter and the run's budget cap.

Modelled on `evals.llm.OpenRouterLLM`: entries live in the same cache directory under a `jev-` prefix, a cached
answer is free, and `Budget.check()` runs before every uncached call. Cost is usage.input_tokens at $0.042 per
million (output is free). The model version each response reports is counted, so a write-up can say which Jev
answered. `ScriptedJev` is the offline stand-in for tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from server.services.agents.jev import (
    JEV_MODEL,
    JevError,
    build_body,
    choice_question,
    cost_usd,
    input_tokens,
    noul_question,
    parse_choice,
    parse_noul,
)

from .llm import Budget, RateLimiter, _read_cache, _write_cache

CACHE_PREFIX = "jev-"
JEV_REQUESTS_PER_MINUTE = 1200


class AsksJev(Protocol):
    model: str

    async def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]: ...


def body_key(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def estimate_tokens(body: dict[str, Any]) -> int:
    """About 4 characters per token over the state and the questions (what Jev bills as input)."""
    return len(json.dumps(body.get("state"), default=str) + json.dumps(body.get("questions"), default=str)) // 4


class CachedJev:
    """Wraps a Jev client (real or scripted) with the eval cache, the cost meter and the budget."""

    def __init__(self, client: AsksJev, *, cache_dir: Path, budget: Budget, concurrency: int = 8,
                 rate_limiter: Optional[RateLimiter] = None) -> None:
        self.client = client
        self.model = getattr(client, "model", JEV_MODEL)
        self.cache_dir = cache_dir
        self.budget = budget
        self._semaphore = asyncio.Semaphore(concurrency)
        self._rate_limiter = rate_limiter
        self.calls = 0
        self.cached_calls = 0
        self.input_tokens = 0  # billed: uncached calls only
        self.cost_usd = 0.0
        self.models: Counter[str] = Counter()  # the version each answer came from, cached ones included

    def path(self, body: dict[str, Any]) -> Path:
        return self.cache_dir / f"{CACHE_PREFIX}{body_key(body)}.json"

    async def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        body = build_body(self.model, state, questions)
        path = self.path(body)
        cached = _read_cache(path)
        if cached is not None:
            self.cached_calls += 1
            self.models[str(cached.get("model", "?"))] += 1
            cached["_meta"] = {"cached": True, "cost": 0.0}
            return cached
        async with self._semaphore:
            self.budget.check()
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            data = await self.client.ask(state, questions)
        cost = cost_usd(data)
        self.budget.add(cost)
        self.calls += 1
        self.input_tokens += input_tokens(data)
        self.cost_usd += cost
        self.models[str(data.get("model", "?"))] += 1
        stored = {k: v for k, v in data.items() if k != "_meta"}
        try:
            _write_cache(path, stored)
        except OSError:  # the call is paid for; losing the entry only costs a rerun
            pass
        return {**stored, "_meta": {"cached": False, "cost": cost}}

    async def choice(self, state: Any, question_id: str, instructions: Any,
                     criteria: Mapping[str, Any]) -> tuple[str, dict[str, float], float]:
        response = await self.ask(state, {question_id: choice_question(instructions, criteria)})
        return parse_choice(response, question_id, set(criteria))

    async def noul(self, state: Any, instructions: Any, question_id: str = "q") -> float:
        response = await self.ask(state, {question_id: noul_question(instructions)})
        return parse_noul(response, question_id)

    def meter(self) -> dict[str, Any]:
        return {"calls": self.calls, "cached_calls": self.cached_calls, "input_tokens": self.input_tokens,
                "cost_usd": round(self.cost_usd, 8), "budget_spent_usd": round(self.budget.spent_usd, 8),
                "models": dict(self.models)}


Script = Callable[[Any, str, dict[str, Any]], dict[str, Any]]


def _uniform(state: Any, qid: str, question: dict[str, Any]) -> dict[str, Any]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.5}
    options = list(question["criteria"])
    p = 1.0 / len(options)
    return {"type": question["type"], "choice": options[0], "probabilities": {o: p for o in options},
            "confidence": p}


class ScriptedJev:
    """Offline Jev: `script(state, question_id, question)` returns each answer. Records every request body."""

    def __init__(self, script: Script = _uniform, *, model: str = JEV_MODEL, version: str = "jev-fake") -> None:
        self.script = script
        self.model = model
        self.version = version
        self.requests: list[dict[str, Any]] = []

    async def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        body = build_body(self.model, state, questions)
        self.requests.append(body)
        answers = {qid: self.script(state, qid, q) for qid, q in body["questions"].items()}
        return {"model": self.version, "answers": answers,
                "usage": {"input_tokens": estimate_tokens(body), "output_tokens": 0}}


__all__ = ["CACHE_PREFIX", "CachedJev", "JEV_REQUESTS_PER_MINUTE", "JevError", "ScriptedJev", "body_key",
           "estimate_tokens"]
