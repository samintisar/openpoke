"""LLM access for evals: OpenRouter with a response cache and a hard budget cap.

Responses are cached by a hash of the exact request payload plus `sample`, so reruns are free
and reproducible, and only changed prompts cost money. `sample` separates the k repeated runs
of one case. Only validated responses are cached.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# USD per million tokens (input, output), used only when OpenRouter omits usage.cost.
FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
}
DEFAULT_PRICING = (3.0, 15.0)
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_STATUS = {401, 402, 403}  # bad key, no credits, forbidden: no later call can succeed
MAX_RETRY_AFTER_S = 60.0

logger = logging.getLogger(__name__)


class RunAborted(RuntimeError):
    """Stops the whole run: continuing with other cases cannot succeed."""


class BudgetExceeded(RunAborted):
    """Raised before an uncached request attempt once spending reaches the cap."""


class FatalAPIError(RunAborted):
    """OpenRouter refused the account (401/402/403); retrying or moving on would only repeat it."""


class DailyQuotaExhausted(RunAborted):
    """A free model's per-day request cap is used up. Finished calls are cached, so rerunning
    the same config tomorrow resumes where this run stopped."""


def _is_daily_quota(text: str) -> bool:
    """OpenRouter's 429 for the free-tier daily cap names the limit, e.g. "free-models-per-day"."""
    lowered = text.lower()
    return "per-day" in lowered or "per day" in lowered


class RetryableResponseError(RuntimeError):
    """A 200 response that is unusable (no assistant message, or finish_reason "error")."""


class Budget:
    """A spending cap checked before each uncached request attempt.

    The check happens before a request is sent and the cost is added when it returns, so up
    to about `concurrency` in-flight calls can pass the check together: the cap can be
    overshot by roughly that many calls' cost.

    Only costs reported in a response body are charged. A request that times out on our side
    (or whose body is lost or unreadable) may still be billed by OpenRouter, but it isn't
    charged here, so real spend can exceed `spent_usd`.
    """

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0

    def check(self) -> None:
        if self.spent_usd >= self.cap_usd:
            raise BudgetExceeded(f"budget of ${self.cap_usd:.2f} reached (spent ${self.spent_usd:.4f})")

    def add(self, cost: float) -> None:
        self.spent_usd += cost


class RateLimiter:
    """Spaces request starts at least 60/rpm seconds apart (free models are capped per minute)."""

    def __init__(
        self,
        requests_per_minute: int,
        *,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval_s = 60.0 / requests_per_minute
        self._sleep = sleep
        self._clock = clock
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            start = max(now, self._next_start)
            self._next_start = start + self.interval_s
        if start > now:
            await self._sleep(start - now)


class LLMClient(Protocol):
    async def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int
    ) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


def build_payload(
    *, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """The exact JSON body sent to OpenRouter."""
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": tools,
        "stream": False,
        "usage": {"include": True},
    }


def _payload_key(payload: dict[str, Any], sample: int) -> str:
    blob = json.dumps({"payload": payload, "sample": sample}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_key(*, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int) -> str:
    """Hash of the sent payload plus sample, so every request parameter is part of the key."""
    return _payload_key(build_payload(model=model, system=system, messages=messages, tools=tools), sample)


def estimate_cost(model: str, usage: dict[str, Any]) -> float:
    if model.endswith(":free"):
        return 0.0
    price_in, price_out = FALLBACK_PRICING.get(model, DEFAULT_PRICING)
    return usage.get("prompt_tokens", 0) * price_in / 1e6 + usage.get("completion_tokens", 0) * price_out / 1e6


def _validation_error(data: dict[str, Any]) -> str | None:
    """Why a 200 body is unusable, or None if it carries a proper assistant message."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "response has no choices"
    if not isinstance(choices[0].get("message"), dict):
        return "choices[0].message is not an object"
    if choices[0].get("finish_reason") == "error":
        return "finish_reason is 'error'"
    return None


def _error_code(error: Any) -> int | None:
    code = error.get("code") if isinstance(error, dict) else None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _retry_after(response: httpx.Response | None) -> float | None:
    """A numeric Retry-After header in seconds, capped; None if absent or not a number."""
    if response is None or "Retry-After" not in response.headers:
        return None
    try:
        return min(float(response.headers["Retry-After"]), MAX_RETRY_AFTER_S)
    except ValueError:
        return None


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None  # a torn or corrupt entry is a miss; the next success overwrites it


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


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
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.budget = budget
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._sleep = sleep  # every retry wait, backoff or Retry-After, goes through this hook
        self._semaphore = asyncio.Semaphore(concurrency)
        self._rate_limiter = rate_limiter
        self._client = httpx.AsyncClient(transport=transport, timeout=120.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int
    ) -> dict[str, Any]:
        payload = build_payload(model=model, system=system, messages=messages, tools=tools)
        key = _payload_key(payload, sample)
        path = self.cache_dir / f"{key}.json"
        cached = _read_cache(path)
        if cached is not None:
            cached["_meta"] = {**cached.get("_meta", {}), "cached": True, "cache_key": key}
            return cached

        async with self._semaphore:
            started = time.perf_counter()
            data, wasted = await self._post(payload)
            latency = time.perf_counter() - started

        usage = data.get("usage") or {}
        cost = float(usage["cost"]) if usage.get("cost") is not None else estimate_cost(model, usage)
        self.budget.add(cost)
        data["_meta"] = {
            "cost": cost + wasted,  # includes charged attempts that returned unusable bodies
            "latency_s": round(latency, 3),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "cached": False,
            "cache_key": key,
        }
        try:
            _write_cache(path, data)
        except OSError as exc:  # the call is paid for; losing the cache entry only costs a rerun
            logger.warning("could not write LLM cache entry %s: %s", path, exc)
        return data

    async def _backoff(self, attempt: int, response: httpx.Response | None = None) -> None:
        retry_after = _retry_after(response)
        await self._sleep(retry_after if retry_after is not None else self.retry_delay_s * 2**attempt)

    async def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """POST with retries; returns the validated body and the cost charged for unusable bodies."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        wasted = 0.0
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            self.budget.check()
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            try:
                response = await self._client.post(OPENROUTER_URL, headers=headers, json=payload)
            except httpx.TransportError:
                if last:
                    raise
                await self._backoff(attempt)
                continue
            if response.status_code in FATAL_STATUS:
                raise FatalAPIError(f"OpenRouter returned {response.status_code}: {response.text[:200]}")
            if response.status_code == 429 and _is_daily_quota(response.text):
                raise DailyQuotaExhausted(f"OpenRouter daily quota reached: {response.text[:200]}")
            if response.status_code in RETRY_STATUS and not last:
                await self._backoff(attempt, response)
                continue
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError:  # a 200 with an HTML or truncated body; usage is unknown
                if last:
                    raise RetryableResponseError("unusable OpenRouter response: body is not JSON") from None
                await self._backoff(attempt, response)
                continue
            if "error" in data:
                if _error_code(data["error"]) in FATAL_STATUS:
                    raise FatalAPIError(f"OpenRouter error: {data['error']}")
                if _error_code(data["error"]) == 429 and _is_daily_quota(json.dumps(data["error"])):
                    raise DailyQuotaExhausted(f"OpenRouter daily quota reached: {data['error']}")
                if _error_code(data["error"]) in RETRY_STATUS and not last:
                    await self._backoff(attempt, response)
                    continue
                raise RuntimeError(f"OpenRouter error: {data['error']}")
            problem = _validation_error(data)
            if problem is None:
                return data, wasted
            charge = (data.get("usage") or {}).get("cost")
            if charge is not None:
                self.budget.add(float(charge))
                wasted += float(charge)
            if last:
                raise RetryableResponseError(f"unusable OpenRouter response: {problem}")
            await self._backoff(attempt, response)
        raise RuntimeError("unreachable")


_ZERO_META = {"cost": 0.0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cached": False, "cache_key": ""}


def tool_call(name: str, index: int = 0, **arguments: Any) -> dict[str, Any]:
    """An OpenAI-style tool call with the deterministic id call_{name}_{index}.

    `index` is the call's position in its assistant message; ScriptedLLM renumbers ids of this
    form by position, so scripts can leave it at 0.
    """
    return {
        "id": f"call_{name}_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _numbered(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give tool_call()-style ids their position in the message (0..n-1); keep custom ids."""
    out = []
    for index, call in enumerate(calls):
        name = (call.get("function") or {}).get("name", "")
        if re.fullmatch(rf"call_{re.escape(name)}_\d+", str(call.get("id", ""))):
            call = {**call, "id": f"call_{name}_{index}"}
        out.append(call)
    return out


class ScriptedLLM:
    """Deterministic stand-in: script(messages, sample) returns the assistant message.

    Tool-call ids made by tool_call() are numbered 0..n-1 within each message.
    """

    def __init__(self, script: Callable[[list[dict[str, Any]], int], dict[str, Any]]) -> None:
        self.script = script

    async def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], sample: int
    ) -> dict[str, Any]:
        message = {"role": "assistant", **self.script(messages, sample)}
        if message.get("tool_calls"):
            message["tool_calls"] = _numbered(message["tool_calls"])
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
                tool_call("send_message_to_user", 0, message="on it"),
                tool_call("send_message_to_agent", 1, agent_name=name, instructions="handle it"),
            ],
        }

    return ScriptedLLM(script)
