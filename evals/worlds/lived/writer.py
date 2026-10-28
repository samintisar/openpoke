"""Turns an Intent into the text of one message, using an LLM, and checks the required mentions."""

from __future__ import annotations

import re
from collections.abc import Sequence

from evals.llm import LLMClient
from evals.worlds.lived.model import Intent

ATTEMPTS = 3
TAIL = 6

USER_SYSTEM = (
    "You write the exact chat message a busy person texts their AI assistant. "
    "Usually one line, casual, often lowercase, sometimes a small typo, but never misspell a required word. "
    "Vary how messages open; most skip greetings. Never announce what kind of request it is "
    "(no \"something new\", \"a new task\", \"two things\"); just say it. "
    "Refer to people by first name only unless the hard rules require more. "
    "Assume the assistant knows the person's life; don't over-explain. "
    "Output only the message itself: no quotes, no preamble, no explanation."
)

AGENT_SYSTEM = (
    "You write the body of a status report that an AI assistant's background worker sends back. "
    "If a contact is involved, say what they replied, for example "
    '"Alice replied: she can do Thursday at 3". If no contact is involved, say what the worker found '
    'or did, for example "Found 3 flights under $400; cheapest is Tuesday 7am". '
    "One or two sentences, plain and factual. Output only the report body: no prefix, no quotes, no preamble."
)

_KIND_PHRASES = {
    "start": "Ask the assistant to do this. It's the first time you mention it, but don't say so",
    "follow_up": "Add the next step to something the assistant is already handling for you",
    "revival": "Pick back up something you last discussed weeks ago",
    "multi": "Ask for two unrelated things in one message; keep them clearly separate, don't blend them",
    "result": "Report the outcome of the task",
}


def _about(intent: Intent) -> str:
    """The intent in plain words: what the message does, who and what it concerns."""
    phrase = _KIND_PHRASES.get(intent.kind, "Send a message to the assistant")
    parts = [phrase + "."]
    if intent.contacts:
        parts.append("It concerns: " + ", ".join(intent.contacts) + " (the user knows them; use first names).")
    if intent.topic:
        parts.append(f"Topic: {intent.topic}.")
    if intent.detail:
        parts.append(f"Specifically: {intent.detail}")
    return " ".join(parts)


def _render_tail(transcript_tail: Sequence[tuple[str, ...]]) -> str:
    lines = [f"{entry[0]}: {entry[1]}" for entry in list(transcript_tail)[-TAIL:]]
    return "\n".join(lines) if lines else "(no earlier messages)"


def build_prompt(intent: Intent, *, transcript_tail: Sequence[tuple[str, ...]], persona: str) -> tuple[str, str]:
    """The (system, user) prompt pair for one intent."""
    system = AGENT_SYSTEM if intent.message_type == "agent" else USER_SYSTEM
    rules = []
    if intent.mention:
        rules.append("Must include (exactly as written): " + ", ".join(f'"{m}"' for m in intent.mention))
    if intent.forbid:
        rules.append("Must not include: " + ", ".join(f'"{f}"' for f in intent.forbid))
    sections = [
        f"Who is sending this: {persona}",
        "Recent conversation:\n" + _render_tail(transcript_tail),
        "What to write: " + _about(intent),
    ]
    if rules:
        sections.append("Hard rules:\n" + "\n".join(rules))
    return system, "\n\n".join(sections)


_QUOTES = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’"}


def _clean(content: str | None) -> str:
    """Strip whitespace and any quotes wrapped around the whole message."""
    text = (content or "").strip()
    while len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    return text


# The model sometimes appends notes about the rules to the message; a real text never mentions them.
_META = re.compile(r"must (not )?include|hard rules?|\bthe rules?\b|required words?|\bthe requirement", re.IGNORECASE)


def _valid(text: str, intent: Intent) -> bool:
    lowered = text.lower()
    return (
        bool(text)
        and not _META.search(text)
        and all(m.lower() in lowered for m in intent.mention)
        and not any(f.lower() in lowered for f in intent.forbid)
    )


async def write(
    intent: Intent,
    *,
    transcript_tail: Sequence[tuple[str, ...]],
    persona: str,
    llm: LLMClient,
    model: str,
    sample: int = 0,
) -> str | None:
    """The message for `intent`, or None if three samples all break the mention rules."""
    system, user = build_prompt(intent, transcript_tail=transcript_tail, persona=persona)
    for s in range(sample, sample + ATTEMPTS):
        data = await llm.complete(
            model=model, system=system, messages=[{"role": "user", "content": user}], tools=[], sample=s
        )
        text = _clean(data["choices"][0]["message"].get("content"))
        if _valid(text, intent):
            return text
    return None
