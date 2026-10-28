"""LLM access for evals: OpenRouter with a response cache and a hard budget cap.

Responses are cached by a hash of (model, system, messages, tools, sample), so reruns are free
and reproducible, and only changed prompts cost money. `sample` separates the k repeated runs
of one case.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# USD per million tokens (input, output), used only when OpenRouter omits usage.cost.
FALLBACK_PRICING: dict[str, tuple[float, float]] = {"anthropic/claude-sonnet-4": (3.0, 15.0)}
DEFAULT_PRICING = (3.0, 15.0)
RETRY_STATUS = {429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    """Raised before an uncached call once spending reaches the cap."""


class Budget:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0

    def check(self) -> None:
        if self.spent_usd >= self.cap_usd:
            raise BudgetExceeded(f"budget of ${self.cap_usd:.2f} reached (spent ${self.spent_usd:.4f})")

    def add(self, cost: float) -> None:
        self.spent_usd += cost


class LLMClient(Protocol):
    async def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int
    ) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


def cache_key(*, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int) -> str:
    blob = json.dumps(
        {"model": model, "system": system, "messages": messages, "tools": tools, "sample": sample},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def estimate_cost(model: str, usage: dict[str, Any]) -> float:
    price_in, price_out = FALLBACK_PRICING.get(model, DEFAULT_PRICING)
    return usage.get("prompt_tokens", 0) * price_in / 1e6 + usage.get("completion_tokens", 0) * price_out / 1e6


class OpenRouterLLM:
    def __init__(
        self,
        *,
        api_key: str,
        cache_dir: Path,
        budget: Budget,
        concurrency: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        retry_delay_s: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.budget = budget
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(transport=transport, timeout=120.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int
    ) -> dict[str, Any]:
        key = cache_key(model=model, system=system, messages=messages, tools=tools, sample=sample)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_meta"] = {**data.get("_meta", {}), "cached": True, "cache_key": key}
            return data

        async with self._semaphore:
            self.budget.check()
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system}, *messages],
                "tools": tools,
                "stream": False,
                "usage": {"include": True},
            }
            started = time.perf_counter()
            data = await self._post(payload)
            latency = time.perf_counter() - started

        usage = data.get("usage") or {}
        cost = float(usage["cost"]) if usage.get("cost") is not None else estimate_cost(model, usage)
        self.budget.add(cost)
        data["_meta"] = {
            "cost": cost,
            "latency_s": round(latency, 3),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "cached": False,
            "cache_key": key,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = self.retry_delay_s
        for attempt in range(self.max_retries + 1):
            response = await self._client.post(OPENROUTER_URL, headers=headers, json=payload)
            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(f"OpenRouter error: {data['error']}")
            return data
        raise RuntimeError("unreachable")


_ZERO_META = {"cost": 0.0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cached": False, "cache_key": ""}


def tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class ScriptedLLM:
    """Deterministic stand-in: script(messages, sample) returns the assistant message."""

    def __init__(self, script: Callable[[list[dict[str, Any]], int], dict[str, Any]]) -> None:
        self.script = script

    async def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int
    ) -> dict[str, Any]:
        message = {"role": "assistant", **self.script(messages, sample)}
        return {"choices": [{"message": message}], "_meta": dict(_ZERO_META)}

    async def aclose(self) -> None:
        return None


_FIRST_AGENT = re.compile(r'<agent name="([^"]+)"')


def fake_llm() -> ScriptedLLM:
    """Offline smoke model: acknowledges, then reuses the first listed agent."""

    def script(messages: list[dict[str, Any]], sample: int) -> dict[str, Any]:
        if any(message["role"] == "tool" for message in messages):
            return {"content": ""}
        match = _FIRST_AGENT.search(messages[0]["content"])
        name = html.unescape(match.group(1)) if match else "New task"
        return {
            "content": "",
            "tool_calls": [
                tool_call("send_message_to_user", message="on it"),
                tool_call("send_message_to_agent", agent_name=name, instructions="handle it"),
            ],
        }

    return ScriptedLLM(script)
