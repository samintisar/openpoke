"""Data model for synthetic eval worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

WORLD_NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Person:
    first: str
    last: str
    company: str

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def email(self) -> str:
        return f"{self.first.lower()}.{self.last.lower()}@{self.company.lower()}.com"


@dataclass
class AgentSpec:
    """An execution agent as it would exist after some days of use."""

    id: str
    name: str
    purpose: str
    entities: list[str]
    last_active_days: float
    created_days: float
    person: str | None = None  # Person.full
    topic: str | None = None
    core: bool = False
    history: list[tuple[str, str]] = field(default_factory=list)  # (request, response)

    def last_active_at(self, now: datetime = WORLD_NOW) -> datetime:
        return now - timedelta(days=self.last_active_days)


@dataclass
class World:
    seed: int
    agents: list[AgentSpec]
    roles: dict[str, str] = field(default_factory=dict)  # role -> agent id
    role_people: dict[str, Person] = field(default_factory=dict)
    reserved_topic: tuple[str, str] | None = None  # never assigned to the main person
    now: datetime = WORLD_NOW

    def by_id(self) -> dict[str, AgentSpec]:
        return {agent.id: agent for agent in self.agents}

    def roster_order(self) -> list[AgentSpec]:
        """Agents oldest-created first, the order roster.json grows in."""
        return sorted(self.agents, key=lambda agent: (-agent.created_days, agent.id))

    def truncated(self, n_agents: int) -> World:
        core = [agent for agent in self.agents if agent.core]
        filler = [agent for agent in self.agents if not agent.core]
        if n_agents < len(core):
            raise ValueError(f"n_agents={n_agents} is smaller than the {len(core)} core agents")
        return World(
            seed=self.seed,
            agents=core + filler[: n_agents - len(core)],
            roles=dict(self.roles),
            role_people=dict(self.role_people),
            reserved_topic=self.reserved_topic,
            now=self.now,
        )
