"""Grows a roster by playing the life plan through the real Interaction Agent turn.

Each event is written as a message, then one IA turn decides which agents get it. Agents the turn
creates join the roster with the event's threads as gold; agents it dispatches gain them. Snapshots
of the roster are taken the first time it reaches each size. Background errands are named by the
same IA turn against an empty roster and merged in afterwards to reach 1,000 agents.
"""

from __future__ import annotations

import asyncio
import copy
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from ...cases import Case
from ...grading import Decision
from ...layers.routing import RoutingCaseError, run_routing_case
from ...llm import LLMClient
from ...systems.baseline import BaselineSystem
from ..model import AgentSpec, World
from .model import LIFE_DAYS, Event, Intent, LifePlan, Snapshot, day_to_datetime
from .writer import write

TAIL = 6  # transcript entries the writer sees
CONTEXT = 20  # transcript entries the IA turn sees
PROGRESS_EVERY = 25


@dataclass
class GrowthResult:
    snapshots: list[Snapshot]
    agents: list[AgentSpec]  # the final roster; created_days / last_active_days relative to the last event played
    transcript: list[tuple[str, str, str]]
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (event id, reason)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "agent"


def _unique_id(name: str, taken: set[str]) -> str:
    base = candidate = _slug(name)
    suffix = 2
    while candidate in taken:
        candidate = f"{base}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _instructions(decision: Decision) -> dict[str, list[str]]:
    """Agent name -> the instructions of each valid send_message_to_agent call, in call order."""
    out: dict[str, list[str]] = {}
    for call in decision.tool_calls:
        if call["name"] != "send_message_to_agent" or "error" in call:
            continue
        args = call["arguments"]
        out.setdefault(str(args.get("agent_name", "")), []).append(str(args.get("instructions", "")))
    return out


def _case(case_id: str, plan: LifePlan, message: str, message_type: str,
          transcript: Sequence[tuple[str, str, str]]) -> Case:
    # Kind follow_up only satisfies the schema; nothing grades growth turns.
    context = [tuple(entry) for entry in transcript[-CONTEXT:]]
    return Case(id=case_id, kind="follow_up", world_seed=plan.seed, message=message, gold_targets=[],
                message_type=message_type, context=context)


class _Roster:
    """Growth state: agents in creation order, with absolute (life-day) created and active days."""

    def __init__(self) -> None:
        self.agents: list[AgentSpec] = []
        self.created_day: dict[str, float] = {}
        self.last_day: dict[str, float] = {}
        self.ids: set[str] = set()
        self.by_id: dict[str, AgentSpec] = {}
        self.thread_agents: dict[str, list[str]] = {}
        self.thread_last_day: dict[str, float] = {}
        self.last_user_threads: tuple[str, ...] = ()
        self.transcript: list[tuple[str, str, str]] = []

    def relative(self, day: float, *, deep: bool = False) -> list[AgentSpec]:
        """Copies of the agents with created_days / last_active_days counted back from `day`."""
        out = []
        for agent in self.agents:
            base = copy.deepcopy(agent) if deep else agent
            out.append(replace(base, created_days=round(day - self.created_day[agent.id], 6),
                               last_active_days=round(day - self.last_day[agent.id], 6)))
        return out

    def _serve(self, thread_id: str, agent_id: str) -> None:
        served = self.thread_agents.setdefault(thread_id, [])
        if agent_id not in served:
            served.append(agent_id)

    def apply(self, event: Event, decision: Decision) -> None:
        threads = list(event.intent.thread_ids)
        calls = _instructions(decision)
        created = list(dict.fromkeys(decision.created_names))
        # A thread already served by an agent this turn dispatched stays with that agent; new agents (and
        # reused agents that had none of the event's threads) only take the threads nobody covered.
        covered = {t for a in decision.dispatched_ids for t in threads if t in self.by_id[a].threads}
        unserved = [t for t in threads if t not in covered]
        per_agent = [[t] for t in unserved] if len(created) == len(unserved) else [unserved] * len(created)
        for name, agent_threads in zip(created, per_agent):
            instructions = calls.get(name) or [""]
            agent = AgentSpec(
                id=_unique_id(name, self.ids), name=name, purpose=instructions[0], entities=[],
                last_active_days=0.0, created_days=0.0, history=[(text, "") for text in instructions],
                threads=list(agent_threads),
            )
            self.agents.append(agent)
            self.by_id[agent.id] = agent
            self.created_day[agent.id] = self.last_day[agent.id] = event.day
            for thread_id in agent_threads:
                self._serve(thread_id, agent.id)
        for agent_id in decision.dispatched_ids:
            agent = self.by_id[agent_id]
            mine = [t for t in threads if t in agent.threads] or unserved
            for thread_id in mine:
                if thread_id not in agent.threads:
                    agent.threads.append(thread_id)
                self._serve(thread_id, agent_id)
            self.last_day[agent_id] = event.day
            agent.history.extend((text, "") for text in calls.get(agent.name, []))
        for thread_id in threads:
            self.thread_last_day[thread_id] = event.day
        if event.intent.message_type == "user":
            self.last_user_threads = tuple(threads)

    def snapshot(self, size: int, day: float) -> Snapshot:
        return Snapshot(
            size=size, day=day, agents=self.relative(day, deep=True), transcript=list(self.transcript),
            thread_agents={t: list(ids) for t, ids in self.thread_agents.items()},
            thread_last_day=dict(self.thread_last_day), last_user_threads=self.last_user_threads,
        )


async def grow(
    plan: LifePlan,
    *,
    llm: LLMClient,
    model: str,
    sizes: Sequence[int] = (10, 50, 200),
    max_agents: int = 200,
    log: Callable[[str], object] = print,
) -> GrowthResult:
    """Play the plan's events in order until the roster reaches `max_agents`.

    Raises ValueError if the events run out before the largest size; RunAborted propagates."""
    roster = _Roster()
    pending = sorted(set(sizes))
    snapshots: list[Snapshot] = []
    skipped: list[tuple[str, str]] = []
    system = BaselineSystem()
    day = 0.0
    total = len(plan.events)

    for index, event in enumerate(plan.events, 1):
        if index > 1 and (index - 1) % PROGRESS_EVERY == 0:
            _progress(log, index - 1, total, roster, skipped, llm)
        intent = event.intent
        owner: AgentSpec | None = None
        if intent.message_type == "agent":
            owner_ids = [a for t in intent.thread_ids for a in roster.thread_agents.get(t, [])]
            if not owner_ids:
                skipped.append((event.id, "result before its thread has an agent"))
                continue
            owner = roster.by_id[owner_ids[0]]
        body = await write(intent, transcript_tail=roster.transcript[-TAIL:], persona=plan.persona,
                           llm=llm, model=model, sample=0)
        if body is None:
            skipped.append((event.id, "writer gave no valid message"))
            continue
        message = f"[SUCCESS] {owner.name}: {body}" if owner else body
        case = _case(f"grow-{event.id}", plan, message, intent.message_type, roster.transcript)
        world = World(seed=plan.seed, agents=roster.relative(event.day), now=day_to_datetime(event.day))
        try:
            decision = await run_routing_case(case, world, system, llm, model=model, sample=0)
        except RoutingCaseError as exc:
            log(f"skipping {event.id}: {exc}")
            skipped.append((event.id, f"routing error: {exc}"))
            continue

        day = event.day
        roster.apply(event, decision)
        stamp = day_to_datetime(event.day).isoformat()
        tag = "agent_message" if intent.message_type == "agent" else "user_message"
        roster.transcript.append((tag, message, stamp))
        roster.transcript.extend(("poke_reply", reply, stamp) for reply in decision.user_messages)

        while pending and len(roster.agents) >= pending[0]:
            snapshots.append(roster.snapshot(pending.pop(0), day))
        if len(roster.agents) >= max_agents:
            break

    if pending:
        raise ValueError(
            f"events ran out at {len(roster.agents)} agents, before sizes {pending} (skipped {len(skipped)})"
        )
    return GrowthResult(snapshots=snapshots, agents=roster.relative(day, deep=True),
                        transcript=list(roster.transcript), skipped=skipped)


def _progress(log: Callable[[str], object], done: int, total: int, roster: _Roster,
              skipped: list[tuple[str, str]], llm: LLMClient) -> None:
    line = f"event {done}/{total} agents {len(roster.agents)} skipped {len(skipped)}"
    budget = getattr(llm, "budget", None)
    if budget is not None:
        line += f" spent ${budget.spent_usd:.2f}"
    log(line)


async def name_background(
    plan: LifePlan, *, llm: LLMClient, model: str, concurrency: int = 8, rng_seed: int | None = None
) -> list[AgentSpec]:
    """One agent per background intent whose IA turn (empty roster) creates one, in intent order.

    The agents' created_days and last_active_days hold the absolute life day the agent was created
    (days since LIFE_START, uniform over LIFE_DAYS), not days ago: merge_background converts them."""
    rng = random.Random(plan.seed if rng_seed is None else rng_seed)
    days = [round(rng.uniform(0, LIFE_DAYS), 3) for _ in plan.background]
    semaphore = asyncio.Semaphore(concurrency)

    async def name_one(index: int, intent: Intent) -> tuple[str, str] | None:
        async with semaphore:
            message = await write(intent, transcript_tail=[], persona=plan.persona, llm=llm, model=model, sample=0)
            if message is None:
                return None
            case = _case(f"bg-{index:04d}", plan, message, "user", [])
            world = World(seed=plan.seed, agents=[], now=day_to_datetime(days[index]))
            try:
                decision = await run_routing_case(case, world, BaselineSystem(), llm, model=model, sample=0)
            except RoutingCaseError:
                return None
            if not decision.created_names:
                return None
            name = decision.created_names[0]
            return name, (_instructions(decision).get(name) or [""])[0]

    named = await asyncio.gather(*(name_one(i, intent) for i, intent in enumerate(plan.background)))
    ids: set[str] = set()
    agents = []
    for index, result in enumerate(named):
        if result is None:
            continue
        name, instructions = result
        agents.append(AgentSpec(
            id=_unique_id(name, ids), name=name, purpose=instructions, entities=[],
            last_active_days=days[index], created_days=days[index], history=[(instructions, "")], threads=[],
        ))
    return agents


def merge_background(snapshot: Snapshot, background: Sequence[AgentSpec], target: int = 1000) -> Snapshot:
    """The snapshot's agents plus background agents, interleaved by created day, up to `target`.

    Background agents are taken in the given order. One whose name equals an existing or already
    kept name is dropped, since production's exact-name match would have reused that agent. Created
    days after the snapshot day are clamped to it. Transcript and thread maps are unchanged."""
    names = {agent.name for agent in snapshot.agents}
    ids = {agent.id for agent in snapshot.agents}
    # (life day created, group, order, agent): snapshot agents sort before background on a tie.
    rows = [(snapshot.day - a.created_days, 0, i, copy.deepcopy(a)) for i, a in enumerate(snapshot.agents)]
    room = target - len(snapshot.agents)
    for i, agent in enumerate(background):
        if room <= 0:
            break
        if agent.name in names:
            continue
        names.add(agent.name)
        created = min(agent.created_days, snapshot.day)
        active = min(max(agent.last_active_days, created), snapshot.day)
        rows.append((created, 1, i, replace(
            copy.deepcopy(agent), id=_unique_id(agent.name, ids),
            created_days=round(snapshot.day - created, 6), last_active_days=round(snapshot.day - active, 6),
        )))
        room -= 1
    rows.sort(key=lambda row: row[:3])
    return Snapshot(
        size=target, day=snapshot.day, agents=[row[3] for row in rows], transcript=list(snapshot.transcript),
        thread_agents={t: list(a) for t, a in snapshot.thread_agents.items()},
        thread_last_day=dict(snapshot.thread_last_day), last_user_threads=snapshot.last_user_threads,
    )
