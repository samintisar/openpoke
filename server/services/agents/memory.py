"""Per-agent memory compaction (spec §5).

An execution agent's prompt history is its latest `<agent_memory>`, the `<agent_identifiers>` seen so far, and the
raw log entries after the memory. Once the un-compacted entries before the newest `history_token_budget` worth
exceed that budget, one cheap-model call folds them into the memory. Raw log entries are never rewritten.

The pure parts (`split_for_budget`, `extract_identifiers`, `compact`, `render_history`) take injected callables and
do no I/O; `maybe_compact` wires them to the log store, the agent directory and OpenRouter.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from html import escape
from typing import Optional

from ...config import get_settings
from ...logging_config import logger
from ...openrouter_client import request_chat_completion
from ..execution.log_store import ExecutionAgentLogStore, get_execution_agent_logs

Complete = Callable[[str, str], Awaitable[str]]


def estimate_tokens(text: str) -> int:
    """Rough token count: four characters per token."""
    return len(text) // 4


@dataclass(frozen=True)
class Entry:
    """One raw execution-log entry."""

    tag: str
    text: str
    timestamp: str = ""


def load_entries(log_store: ExecutionAgentLogStore, agent_id: str) -> list[Entry]:
    """The agent's raw log entries in order (memory entries excluded)."""
    return [Entry(tag, payload, timestamp) for tag, timestamp, payload in log_store.iter_entries(agent_id)]


def render_entry(entry: Entry) -> str:
    """Render an entry exactly as `ExecutionAgentLogStore.load_transcript` does."""
    escaped = escape(entry.text, quote=False)
    if entry.timestamp:
        return f"<{entry.tag} timestamp=\"{entry.timestamp}\">{escaped}</{entry.tag}>"
    return f"<{entry.tag}>{escaped}</{entry.tag}>"


def render_entries(entries: Iterable[Entry]) -> str:
    return "\n".join(render_entry(entry) for entry in entries)


def split_for_budget(
    entries: Sequence[Entry], budget_tokens: int, upto: int = 0
) -> tuple[list[Entry], list[Entry]]:
    """Split the entries after the first `upto` (already in memory) into `(prefix, tail)`.

    The tail is the newest entries whose rendered size fits in `budget_tokens`; the prefix is everything between
    the memory and the tail, i.e. what the next compaction would fold in.
    """
    start = min(max(upto, 0), len(entries))
    tail_start = len(entries)
    used = 0
    for index in range(len(entries) - 1, start - 1, -1):
        cost = estimate_tokens(render_entry(entries[index])) + 1  # +1 for the joining newline
        if used + cost > budget_tokens:
            break
        used += cost
        tail_start = index
    return list(entries[start:tail_start]), list(entries[tail_start:])


# -- identifiers ---------------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
# `thread_id: x`, `"threadId": "x"`, `draft_id=x`, and the backslash-escaped forms inside JSON-in-JSON payloads.
_KEYED_ID_RE = re.compile(
    r"""(?i)\b(?:draft|thread|message)_?id\\*["']?\s*[:=]\s*\\*["']?([A-Za-z0-9_\-]{4,})"""
)
# Gmail message/thread ids are 16+ lowercase hex characters (with at least one digit).
_HEX_ID_RE = re.compile(r"(?<![\w.%+@\-])(?=[0-9a-f]*\d)[0-9a-f]{16,}(?![\w@\-])")
# Gmail draft ids look like `r-1234567890123456789` or `r1234567890`.
_DRAFT_ID_RE = re.compile(r"(?<![\w\-])r-?\d{8,}(?![\w\-])")
_NOT_IDS = {"none", "null", "true", "false", "undefined"}


def extract_identifiers(texts: Iterable[str]) -> list[str]:
    """Email addresses, thread/message/draft ids, in order of first appearance, deduped (case-insensitively)."""
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        matches: list[tuple[int, str]] = []
        matches.extend((m.start(), m.group(0)) for m in _EMAIL_RE.finditer(text))
        matches.extend((m.start(1), m.group(1)) for m in _KEYED_ID_RE.finditer(text))
        matches.extend((m.start(), m.group(0)) for m in _HEX_ID_RE.finditer(text))
        matches.extend((m.start(), m.group(0)) for m in _DRAFT_ID_RE.finditer(text))
        for _, value in sorted(matches, key=lambda match: match[0]):
            key = value.lower()
            if key in _NOT_IDS or key in seen:
                continue
            seen.add(key)
            found.append(value)
    return found


def merge_identifiers(existing: Iterable[str], new: Iterable[str]) -> list[str]:
    merged = list(existing)
    seen = {value.lower() for value in merged}
    for value in new:
        if value.lower() not in seen:
            seen.add(value.lower())
            merged.append(value)
    return merged


# -- compaction -----------------------------------------------------------------------------------------------

COMPACTION_PROMPT = """You maintain the long-term memory of an execution agent. The agent owns one matter for the \
user (an email thread, a booking, a negotiation, a recurring chore) and acts through Gmail tools. You are given the \
agent's purpose, its existing memory (possibly empty) and older log entries that are about to leave its prompt. \
Fold the new entries into the memory.

Return only a JSON object, with no other text and no code fences:
{"memory": "...", "purpose": "...", "last_status": "..."}

- memory: under 300 words. Keep the people involved and their roles; decisions made (amounts, dates, choices); \
commitments and deadlines (who owes what, by when); the user's preferences (tone, sign-off, timing, what to avoid); \
and open loops still waiting on someone. Keep everything from the existing memory that still matters, update what \
changed, and drop chit-chat, tool mechanics and anything superseded.
- purpose: one sentence describing the agent's whole matter, not just its latest request.
- last_status: one line on where the matter stands now.

Email addresses and thread, message and draft ids are preserved separately, verbatim, so you need not repeat them."""


class CompactionError(Exception):
    """The compaction call failed or returned unusable output."""


@dataclass(frozen=True)
class CompactionResult:
    memory: str
    purpose: str
    last_status: str


def build_compaction_input(prefix: Sequence[Entry], existing_memory: str, purpose: str) -> str:
    return (
        f"Purpose: {purpose}\n\n"
        f"<existing_memory>\n{existing_memory.strip() or '(none)'}\n</existing_memory>\n\n"
        f"<new_entries>\n{render_entries(prefix)}\n</new_entries>"
    )


def parse_compaction(raw: str, purpose: str) -> CompactionResult:
    """Parse the model's JSON; a missing purpose keeps the old one. Raises `CompactionError`."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", text).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise CompactionError("compaction output has no JSON object")
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise CompactionError(f"compaction output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CompactionError("compaction output is not a JSON object")
    memory = data.get("memory")
    if not isinstance(memory, str) or not memory.strip():
        raise CompactionError("compaction output has an empty memory")
    new_purpose = data.get("purpose")
    last_status = data.get("last_status")
    return CompactionResult(
        memory=memory.strip(),
        purpose=new_purpose.strip() if isinstance(new_purpose, str) and new_purpose.strip() else purpose,
        last_status=" ".join(last_status.split()) if isinstance(last_status, str) else "",
    )


async def compact(
    prefix: Sequence[Entry], existing_memory: str, purpose: str, *, complete: Complete
) -> CompactionResult:
    """Fold `prefix` into `existing_memory` with one `complete(system, user)` call."""
    try:
        raw = await complete(COMPACTION_PROMPT, build_compaction_input(prefix, existing_memory, purpose))
    except CompactionError:
        raise
    except Exception as exc:
        raise CompactionError(f"compaction call failed: {exc}") from exc
    return parse_compaction(raw, purpose)


_CONTEXT_BEFORE = 70
_CONTEXT_AFTER = 30
_SPACE_RE = re.compile(r"\s+")


def annotate_identifiers(identifiers: Sequence[str], texts: Sequence[str]) -> list[str]:
    """Each identifier with the text around its last appearance in `texts`, e.g.
    `119649bb5f34b724 (Calling reply_to_thread with: {"thread_id": "… "subject": "Re: Lease 4B")`.

    A bare list of ids does not say which is the main thread and which a side thread or a deleted
    draft (on the memory eval, unlabelled ids cut thread-id recall to 0.50), so the context goes with them.
    """
    lowered = [text.lower() for text in texts]
    annotated: list[str] = []
    for identifier in identifiers:
        needle = identifier.lower()
        context = ""
        for text, low in zip(reversed(texts), reversed(lowered)):
            pos = low.rfind(needle)
            if pos >= 0:
                before = text[max(0, pos - _CONTEXT_BEFORE):pos]
                after = text[pos + len(identifier):pos + len(identifier) + _CONTEXT_AFTER]
                context = _SPACE_RE.sub(" ", f"{before}…{after}").strip()
                break
        annotated.append(f"{identifier} ({context})" if context else identifier)
    return annotated


def render_history(
    memory: str, identifiers: Sequence[str], tail_entries: Sequence[Entry], context_entries: Sequence[Entry] = ()
) -> str:
    """`<agent_memory>`, `<agent_identifiers>`, then the raw tail rendered as the log store renders it.

    With `context_entries` (the compacted prefix) each identifier gets its own line with the text around
    its last use there; without them, the identifiers are a comma-separated list.
    """
    if context_entries:
        ids = "\n" + "\n".join(annotate_identifiers(identifiers, [entry.text for entry in context_entries])) + "\n"
    else:
        ids = ", ".join(identifiers)
    head = (
        f"<agent_memory>{escape(memory, quote=False)}</agent_memory>\n"
        f"<agent_identifiers>{escape(ids, quote=False)}</agent_identifiers>\n"
    )
    return head + render_entries(tail_entries)


# -- orchestration --------------------------------------------------------------------------------------------

_LOCKS: dict[str, asyncio.Lock] = {}


def compaction_lock(agent_id: str) -> asyncio.Lock:
    """The per-agent lock compaction runs under."""
    lock = _LOCKS.get(agent_id)
    if lock is None:
        lock = _LOCKS[agent_id] = asyncio.Lock()
    return lock


async def _openrouter_complete(system: str, user: str) -> str:
    settings = get_settings()
    response = await request_chat_completion(
        model=settings.summarizer_model,
        messages=[{"role": "user", "content": user}],
        system=system,
        api_key=settings.openrouter_api_key,
    )
    choices = response.get("choices") or []
    content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not content:
        raise CompactionError("OpenRouter response missing content")
    return content


async def maybe_compact(
    agent_id: str,
    *,
    log_store: Optional[ExecutionAgentLogStore] = None,
    directory=None,
    complete: Optional[Complete] = None,
    budget_tokens: Optional[int] = None,
) -> bool:
    """Compact the agent's history if its un-compacted prefix exceeds the budget. Never raises.

    Returns True when a new memory was recorded. On failure the agent keeps its full history and the next run
    retries.
    """
    try:
        logs = log_store or get_execution_agent_logs()
        budget = budget_tokens if budget_tokens is not None else get_settings().history_token_budget

        async with compaction_lock(agent_id):
            existing = logs.latest_memory(agent_id)
            upto = existing.upto if existing else 0
            prefix, _ = split_for_budget(load_entries(logs, agent_id), budget, upto)
            if estimate_tokens(render_entries(prefix)) <= budget:
                return False

            if directory is None:
                from .registry import get_agent_directory

                directory = get_agent_directory()
            card = directory.get(agent_id)
            purpose = (card.purpose or card.name) if card else agent_id
            result = await compact(
                prefix, existing.memory if existing else "", purpose, complete=complete or _openrouter_complete
            )
            identifiers = merge_identifiers(
                existing.identifiers if existing else [], extract_identifiers(entry.text for entry in prefix)
            )
            logs.record_memory(agent_id, result.memory, identifiers, upto + len(prefix))
            logger.info(f"[{agent_id}] Compacted {len(prefix)} log entries into memory")
            if card is not None:
                try:
                    directory.update_card(agent_id, purpose=result.purpose, last_status=result.last_status)
                except Exception as exc:
                    logger.warning(f"[{agent_id}] Failed to refresh card after compaction: {exc}")
            return True
    except Exception as exc:
        logger.warning(f"[{agent_id}] Memory compaction failed; keeping full history: {exc}")
        return False


__all__ = [
    "COMPACTION_PROMPT",
    "CompactionError",
    "CompactionResult",
    "Entry",
    "compact",
    "compaction_lock",
    "estimate_tokens",
    "extract_identifiers",
    "load_entries",
    "maybe_compact",
    "merge_identifiers",
    "render_entries",
    "render_entry",
    "render_history",
    "split_for_budget",
]
