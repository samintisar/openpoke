"""Agent cards: the compact identity shown to the interaction agent for routing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Function words and pronouns: never entities, never useful for matching.
STOPWORDS = frozenset(
    """
    a an the and or but nor so yet if then than that this these those there here of to in on at by for from
    with without about into onto over under after before between through during up down out off as is am are
    was were be been being do does did done have has had having can could will would shall should may might
    must not no yes i me my mine myself we us our ours you your yours he him his she her hers it its they them
    their theirs who whom whose which what when where why how all any some each every both either neither
    just also too very really still again only even ever please thanks thank hi hey hello ok okay im ill ive
    let lets dont doesnt didnt cant wont isnt arent wasnt
    """.split()
)

MONTHS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
WEEKDAYS = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun "
    "today tomorrow yesterday tonight weekend".split()
)

# Words that are capitalized at the start of instructions or in Title Case names but name no one.
COMMON_WORDS = frozenset(
    """
    ask tell send draft reply write email emails mail call text message ping remind reminder schedule book find
    search check look get make set put go keep give take bring share forward follow followup confirm cancel
    reschedule update add remove create start stop help sort plan organize coordinate arrange review read
    summarize summary reach respond request see let use note notes report status latest recent new old next
    last first quick short brief final draft drafts thread threads meeting meetings invite event events
    calendar contact contacts user agent task tasks question questions info details detail item items list
    buy order return track compare research price prices prioritize pick choose decide figure gather pull
    yes no maybe sure great good thanks best regards dear subject re fw fwd do if
    """.split()
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENTENCE_BREAK_RE = re.compile(r"[.!?:;\n]+")
_CONNECTORS = STOPWORDS


@dataclass
class AgentCard:
    """Registry entry for one execution agent."""

    id: str
    name: str
    purpose: str
    entities: list[str] = field(default_factory=list)
    last_status: str = ""
    status: str = "active"
    created_at: datetime = EPOCH
    last_active_at: datetime = EPOCH
    run_count: int = 0


def as_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC so card and `now` timestamps always compare."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def is_common(word: str) -> bool:
    lowered = word.lower()
    return lowered in STOPWORDS or lowered in MONTHS or lowered in WEEKDAYS or lowered in COMMON_WORDS


def _clean_word(word: str) -> str:
    word = word.replace("’", "'").strip("'-")
    if word.lower().endswith("'s"):
        word = word[:-2]
    return word.replace("'", "")


def _is_title_case(words: list[str]) -> bool:
    """A Title Case label ("Kwame Book Club Dates"): capitalization says nothing about names."""
    content = [w for w in words if w.lower() not in _CONNECTORS]
    return len(content) >= 3 and all(w[0].isupper() for w in content)


def extract_entities(texts: Iterable[str]) -> list[str]:
    """Email addresses and capitalized name-like tokens, lowercased, in order of first appearance."""
    found: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            found.append(value)

    texts = [text for text in texts if text]
    sentences: list[list[str]] = []
    for text in texts:
        for email in _EMAIL_RE.findall(text):
            add(email.lower())
        for sentence in _SENTENCE_BREAK_RE.split(_EMAIL_RE.sub(" ", text)):
            words = [w for w in (_clean_word(w) for w in _WORD_RE.findall(sentence)) if w]
            if words and not _is_title_case(words):
                sentences.append(words)
    # A capitalized sentence opener that also appears in lowercase ("Buy …" / "to buy") is an ordinary word.
    lowercase = {w for words in sentences for w in words if w.islower()}
    for words in sentences:
        for index, word in enumerate(words):
            if len(word) < 2 or not word[0].isupper() or is_common(word):
                continue
            if index == 0 and word.lower() in lowercase:
                continue
            add(word.lower())
    return found


def normalize_name(name: str) -> str:
    """Lowercase, drop apostrophes, turn other punctuation into spaces, collapse whitespace."""
    lowered = (name or "").lower().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^\w\s]|_", " ", lowered).split())


def slug_id(name: str, taken: Iterable[str]) -> str:
    """Slug of `name`, with `-2`, `-3`… appended when it collides with a taken id."""
    base = normalize_name(name).replace(" ", "-") or "agent"
    used = set(taken)
    if base not in used:
        return base
    suffix = 2
    while f"{base}-{suffix}" in used:
        suffix += 1
    return f"{base}-{suffix}"


def find_card(cards: Iterable[AgentCard], agent_id: str) -> Optional[AgentCard]:
    return next((card for card in cards if card.id == agent_id), None)


__all__ = [
    "AgentCard",
    "EPOCH",
    "as_utc",
    "extract_entities",
    "find_card",
    "is_common",
    "normalize_name",
    "slug_id",
]
