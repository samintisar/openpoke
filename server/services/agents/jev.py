"""Client for Jev, TypeSafe's "System One" decision model (https://docs.typesafe.ai).

One POST answers a set of typed questions about a `state` (text or JSON):
- `choice`: pick one of up to 255 options, each described by `criteria`; returns the choice, a probability per
  option and a confidence;
- `score`: like choice, over graded options;
- `noul`: a probability (0..1) that a statement about the state holds.

Production-safe: every failure (no key, transport, timeout, non-2xx, a body of the wrong shape) is raised as
`JevError`, never as anything else, so a caller can fall back with one `except`. The key comes from
`TYPESAFE_API_KEY` (the repo-root `.env` is loaded by `server.config`) and is never logged or put in an error.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_USD_PER_M_INPUT = 0.042  # output tokens are free
MAX_CHOICE_OPTIONS = 255
QUESTION_TYPES = ("choice", "score", "noul")
_ERROR_BODY_CHARS = 300


class JevError(Exception):
    """A Jev call failed or returned something unusable."""


def api_key_from_env() -> Optional[str]:
    """TYPESAFE_API_KEY from the environment, loading the repo-root `.env` like the rest of the server."""
    if os.getenv("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    try:
        from ... import config  # noqa: F401  (importing it loads .env into os.environ)
    except Exception:  # pragma: no cover - a broken config must not break the caller
        return None
    return os.getenv("TYPESAFE_API_KEY") or None


def build_body(model: str, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The exact JSON body sent to Jev (also the eval cache's key)."""
    if not isinstance(questions, Mapping) or not questions:
        raise JevError("a Jev request needs at least one question")
    for qid, question in questions.items():
        kind = question.get("type") if isinstance(question, Mapping) else None
        if kind not in QUESTION_TYPES:
            raise JevError(f"question {qid!r} has type {kind!r}, not one of {QUESTION_TYPES}")
        if kind in ("choice", "score"):
            criteria = question.get("criteria")
            if not isinstance(criteria, Mapping) or not criteria:
                raise JevError(f"{kind} question {qid!r} needs criteria (option -> description)")
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise JevError(f"{kind} question {qid!r} has {len(criteria)} options; the limit is {MAX_CHOICE_OPTIONS}")
    return {"model": model, "state": state, "questions": {str(k): dict(v) for k, v in questions.items()}}


def choice_question(instructions: Any, criteria: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul_question(instructions: Any) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def _answer(response: Any, question_id: str, *kinds: str) -> dict[str, Any]:
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        raise JevError("Jev response has no answers object")
    answer = answers.get(question_id)
    if not isinstance(answer, dict):
        raise JevError(f"Jev response has no answer for {question_id!r}")
    if answer.get("type", kinds[0]) not in kinds:
        raise JevError(f"Jev answered {question_id!r} as {answer.get('type')!r}, not {' or '.join(kinds)}")
    return answer


def _unit(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise JevError(f"Jev {what} {value!r} is not a number") from None
    if not 0.0 <= number <= 1.0:
        raise JevError(f"Jev {what} {number} is outside 0..1")
    return number


def parse_choice(response: Any, question_id: str,
                 options: Optional[set[str]] = None) -> tuple[str, dict[str, float], float]:
    """(choice, probabilities, confidence) of a choice or score answer; the choice must be one of `options`."""
    try:
        answer = _answer(response, question_id, "choice", "score")
        pick = answer.get("choice")
        if not isinstance(pick, str):
            raise JevError(f"Jev choice for {question_id!r} is {pick!r}, not an option name")
        if options is not None and pick not in options:
            raise JevError(f"Jev chose {pick!r} for {question_id!r}, which is not an option")
        probs = answer.get("probabilities") or {}
        if not isinstance(probs, dict):
            raise JevError(f"Jev probabilities for {question_id!r} are not an object")
        probabilities = {str(k): _unit(v, "probability") for k, v in probs.items()}
        confidence = _unit(answer.get("confidence", probabilities.get(pick, 0.0)), "confidence")
        return pick, probabilities, confidence
    except JevError:
        raise
    except Exception as exc:  # any surprise in the body's shape
        raise JevError(f"unusable Jev choice answer: {type(exc).__name__}: {exc}") from exc


def parse_noul(response: Any, question_id: str) -> float:
    try:
        return _unit(_answer(response, question_id, "noul").get("noul"), "noul")
    except JevError:
        raise
    except Exception as exc:
        raise JevError(f"unusable Jev noul answer: {type(exc).__name__}: {exc}") from exc


def input_tokens(response: Any) -> int:
    usage = response.get("usage") if isinstance(response, dict) else None
    try:
        return int((usage or {}).get("input_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def cost_usd(response: Any) -> float:
    return input_tokens(response) * JEV_USD_PER_M_INPUT / 1e6


class JevClient:
    """Async Jev client. `ask` returns the raw response; `choice` and `noul` ask one typed question each."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = JEV_URL, model: str = JEV_MODEL,
                 timeout_s: float = 10.0, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._api_key = api_key if api_key is not None else api_key_from_env()
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)

    def __repr__(self) -> str:  # never shows the key
        return f"JevClient(base_url={self.base_url!r}, model={self.model!r}, timeout_s={self.timeout_s})"

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # pragma: no cover
            pass

    async def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        try:
            return await self._ask(state, questions)
        except JevError:
            raise
        except Exception as exc:
            raise JevError(f"Jev call failed: {type(exc).__name__}") from exc

    async def _ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        if not self._api_key:
            raise JevError("TYPESAFE_API_KEY is not set")
        body = build_body(self.model, state, questions)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            response = await self._client.post(self.base_url, headers=headers, json=body)
        except httpx.TimeoutException:
            raise JevError(f"Jev call timed out after {self.timeout_s}s") from None
        except httpx.HTTPError as exc:
            raise JevError(f"Jev transport error: {type(exc).__name__}") from None
        if not 200 <= response.status_code < 300:
            raise JevError(f"Jev returned HTTP {response.status_code}: {response.text[:_ERROR_BODY_CHARS]}")
        try:
            data = response.json()
        except ValueError:
            raise JevError("Jev returned a body that is not JSON") from None
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise JevError("Jev response has no answers object")
        missing = [qid for qid in body["questions"] if qid not in data["answers"]]
        if missing:
            raise JevError(f"Jev response has no answer for {missing}")
        return data

    async def choice(self, state: Any, question_id: str, instructions: Any,
                     criteria: Mapping[str, Any]) -> tuple[str, dict[str, float], float]:
        response = await self.ask(state, {question_id: choice_question(instructions, criteria)})
        return parse_choice(response, question_id, set(criteria))

    async def noul(self, state: Any, instructions: Any, question_id: str = "q") -> float:
        response = await self.ask(state, {question_id: noul_question(instructions)})
        return parse_noul(response, question_id)


__all__ = [
    "JEV_MODEL", "JEV_URL", "JEV_USD_PER_M_INPUT", "JevClient", "JevError", "api_key_from_env", "build_body",
    "choice_question", "cost_usd", "input_tokens", "noul_question", "parse_choice", "parse_noul",
]
