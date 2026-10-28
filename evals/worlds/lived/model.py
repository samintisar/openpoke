"""Shared types for the lived-in world: a seeded life plan whose roster the real IA grows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..model import AgentSpec

LIFE_DAYS = 180
LIFE_START = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)


def day_to_datetime(day: float) -> datetime:
    return LIFE_START + timedelta(days=day)


@dataclass(frozen=True)
class Contact:
    first: str
    last: str
    relation: str  # work, client, recruiter, family, friend, service
    company: str = ""

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def email(self) -> str:
        domain = (self.company or "mail").lower().replace(" ", "")
        return f"{self.first.lower()}.{self.last.lower()}@{domain}.com"


@dataclass(frozen=True)
class Thread:
    """One real-world task: a person (or none, for errands) and a topic."""

    id: str
    contact: str | None  # Contact.full
    topic: str
    domain: str  # the relation pool the topic came from, or "errand"


@dataclass(frozen=True)
class Intent:
    """What one message must say. The writer turns it into text; gold comes from thread_ids."""

    kind: str  # event kind (start, follow_up, result, revival, multi) or a routing kind for probes
    thread_ids: tuple[str, ...]
    contacts: tuple[str, ...] = ()  # Contact.full for each person the message is about
    topic: str = ""
    detail: str = ""  # free-text instruction for the writer
    mention: tuple[str, ...] = ()  # must appear (case-insensitive)
    forbid: tuple[str, ...] = ()  # must not appear (case-insensitive)
    message_type: str = "user"  # "user" or "agent"


@dataclass(frozen=True)
class Event:
    id: str
    day: float  # days since the life started
    intent: Intent


@dataclass
class LifePlan:
    seed: int
    persona: str
    contacts: list[Contact]
    threads: list[Thread]
    events: list[Event]
    background: list[Intent] = field(default_factory=list)

    def thread(self, thread_id: str) -> Thread:
        return next(t for t in self.threads if t.id == thread_id)

    def contact(self, full: str) -> Contact:
        return next(c for c in self.contacts if c.full == full)


@dataclass
class Snapshot:
    """The grown roster the first time it reached `size` agents."""

    size: int
    day: float
    agents: list[AgentSpec]  # creation order; created_days / last_active_days are relative to `day`
    transcript: list[tuple[str, str, str]]  # (tag, text, ISO timestamp), oldest first
    thread_agents: dict[str, list[str]]  # thread id -> agent ids that served it, first-served first
    thread_last_day: dict[str, float]  # thread id -> day of its last played event
    last_user_threads: tuple[str, ...] = ()  # threads of the last user event played
