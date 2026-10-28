"""Shared types for the lived-in world: a seeded life plan whose roster the real IA grows."""

from __future__ import annotations

from dataclasses import dataclass, field

LIFE_DAYS = 180


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
