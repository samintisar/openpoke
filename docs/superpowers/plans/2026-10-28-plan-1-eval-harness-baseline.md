# Plan 1 — Eval Harness and Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline-first eval harness (synthetic worlds, cases, L1 retrieval eval, L2 live routing eval, cache, budget, metrics, viewer, leaderboard, integrity checks) and record OpenPoke's **baseline** routing numbers before any fix lands.

**Architecture:** A top-level `evals/` package, separate from `server/`. A seeded generator builds a 1,000-agent world whose correct answers are known by construction; smaller rosters are prefixes of it. A *routing system* renders the `<active_agents>` block the Interaction Agent (IA) sees. Today's behaviour is the `baseline` system. L2 runs one real IA turn through OpenRouter, using the server's own prompt helpers, with tools intercepted and graded by outcome. Everything except L2 runs offline, and L2 has a scripted fake model for tests.

**Tech Stack:** Python 3.10+ (the repo `.venv` is 3.14), pytest + pytest-asyncio, httpx (`MockTransport` in tests), PyYAML, OpenRouter chat completions.

**Spec:** `docs/superpowers/specs/2026-10-27-agent-overload-design.md` (§8 Evaluation, §9 step 1–2). This is plan 1 of 5.

**Conventions for every task**
- Run all commands from the repo root with the venv active: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux).
- Match the server's style: module docstrings, type hints, `from __future__ import annotations`, small focused functions.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## File structure

```
requirements-dev.txt                 dev/test deps on top of server/requirements.txt
pytest.ini                           test paths, pythonpath, asyncio mode
.gitignore                           + evals/.cache/, evals/results/*.html
server/services/execution/roster.py  MODIFY: cross-platform file lock (temporary; removed in Plan 2)
server/tests/__init__.py
server/tests/test_roster_lock.py
evals/
  __init__.py
  README.md                          how to run and read the harness
  cases.py                           Case schema, kinds, NEW sentinel, split, suites, filtering
  grading.py                         Decision + outcome grading (outcomes, not paths)
  llm.py                             LLMClient protocol, OpenRouterLLM (cache + budget + retries), ScriptedLLM, fake_llm
  metrics.py                         pass@1, pass^k, rates, per-kind, tokens, cost
  integrity.py                       checksum of protected eval files; `--update`
  leaderboard.py                     regenerates results/leaderboard.md
  view.py                            static HTML report per run
  run.py                             CLI runner
  PROTECTED.sha256                   committed checksum of protected files
  suites/regression.txt              case ids that must keep passing
  experiments/smoke.yaml
  experiments/baseline.yaml
  experiments/JOURNAL.md             one hypothesis per experiment
  results/                           result JSON + leaderboard.md (committed); *.html (ignored)
  systems/__init__.py                registry: get_system(name)
  systems/base.py                    RoutingSystem protocol
  systems/baseline.py                today's bare-name roster
  layers/__init__.py
  layers/retrieval.py                L1
  layers/routing.py                  L2
  worlds/__init__.py
  worlds/model.py                    Person, AgentSpec, World
  worlds/generator.py                build_world, generate_cases
  worlds/handwritten.py              loader + validation
  worlds/handwritten_cases.yaml      hand-written world + cases
  tests/__init__.py
  tests/test_cases.py
  tests/test_world_generator.py
  tests/test_handwritten.py
  tests/test_grading.py
  tests/test_baseline_system.py
  tests/test_llm.py
  tests/test_layers.py
  tests/test_metrics.py
  tests/test_integrity.py
  tests/test_run.py
```

---

### Task 0: Dev environment and test scaffold

**Files:**
- Create: `requirements-dev.txt`, `pytest.ini`, `server/tests/__init__.py`, `evals/__init__.py`, `evals/tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create `requirements-dev.txt`**

```text
-r server/requirements.txt
pytest>=8.0
pytest-asyncio>=0.23
pyyaml>=6.0
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = server/tests evals/tests
pythonpath = .
asyncio_mode = auto
```

- [ ] **Step 3: Create the empty package files**

`server/tests/__init__.py` and `evals/tests/__init__.py`: empty files.

`evals/__init__.py`:

```python
"""Evaluation harness for OpenPoke routing (kept outside the server package)."""
```

- [ ] **Step 4: Append to `.gitignore`**

```gitignore

# Eval harness
evals/.cache/
evals/results/*.html
```

- [ ] **Step 5: Install and confirm pytest runs**

Run: `python -m pip install -r requirements-dev.txt`
Then: `python -m pytest -q`
Expected: `no tests ran` (exit code 5). No import errors.

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt pytest.ini .gitignore server/tests/__init__.py evals/__init__.py evals/tests/__init__.py
git commit -m "Add test scaffold and dev requirements"
```

---

### Task 1: Cross-platform roster lock (temporary Windows fix)

`roster.py` imports `fcntl`, which is Unix-only, so the server can't even be imported on Windows. Plan 2 deletes `roster.py`. Until then the eval needs to import the server's prompt code.

**Files:**
- Modify: `server/services/execution/roster.py:1-55`
- Test: `server/tests/test_roster_lock.py`

- [ ] **Step 1: Write the failing test**

```python
"""The roster must import and round-trip on every platform."""

import importlib

from server.services.execution.roster import AgentRoster


def test_roster_round_trip(tmp_path):
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    roster.add_agent("Email to Alice")
    roster.add_agent("Q3 Budget")

    assert AgentRoster(path).get_agents() == ["Email to Alice", "Q3 Budget"]


def test_interaction_agent_module_imports():
    module = importlib.import_module("server.agents.interaction_agent.agent")
    assert callable(module.build_system_prompt)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest server/tests/test_roster_lock.py -v`
Expected on Windows: collection error `ModuleNotFoundError: No module named 'fcntl'`. (On Linux both tests already pass. Continue anyway; the change must keep them passing.)

- [ ] **Step 3: Replace the `fcntl` import and calls**

In `server/services/execution/roster.py`, replace the import block at the top:

```python
"""Simple agent roster management - just a list of agent names."""

import json
import sys
import time
from pathlib import Path
from typing import IO

from ...logging_config import logger

if sys.platform == "win32":
    import msvcrt

    def _try_lock(handle: IO[str]) -> None:
        """Take a non-blocking lock on the first byte; raise BlockingIOError if held."""
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc

    def _unlock(handle: IO[str]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: IO[str]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: IO[str]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
```

Then, inside `AgentRoster.save`, replace the two `fcntl` calls:

```python
                with open(self._roster_path, 'w') as f:
                    _try_lock(f)
                    try:
                        json.dump(self._agents, f, indent=2)
                        return
                    finally:
                        _unlock(f)
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest server/tests/test_roster_lock.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add server/services/execution/roster.py server/tests/test_roster_lock.py
git commit -m "Make roster file lock work on Windows"
```

---

### Task 2: Case schema, splits and suites

**Files:**
- Create: `evals/cases.py`, `evals/suites/regression.txt`
- Test: `evals/tests/test_cases.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from evals.cases import NEW, ROUTING_KINDS, Case, filter_cases


def _case(case_id: str, kind: str = "follow_up", gold=None, **kw) -> Case:
    return Case(id=case_id, kind=kind, world_seed=1, message="m", gold_targets=gold or ["a"], **kw)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        _case("x", kind="nonsense")


def test_expected_ids_and_new():
    case = _case("x", kind="multi_task", gold=["agent-1", NEW])
    assert case.expected_ids == ["agent-1"]
    assert case.expects_new is True
    assert case.expects_reuse is True


def test_agent_result_never_expects_reuse():
    case = _case("x", kind="agent_result", dispatch_optional=True)
    assert case.expects_reuse is False


def test_split_is_stable_and_roughly_70_30():
    cases = [_case(f"case-{i}") for i in range(400)]
    splits = [c.split for c in cases]
    assert splits == [c.split for c in cases]
    dev_share = splits.count("dev") / len(splits)
    assert 0.6 < dev_share < 0.8


def test_filter_by_split_kind_and_suite():
    cases = [_case(f"case-{i}", kind=ROUTING_KINDS[i % 2]) for i in range(50)]
    regression = {cases[0].id, cases[1].id}

    dev = filter_cases(cases, split="dev", regression_ids=regression)
    assert dev and all(c.split == "dev" for c in dev)

    only_follow = filter_cases(cases, kinds=["follow_up"], regression_ids=regression)
    assert {c.kind for c in only_follow} == {"follow_up"}

    reg = filter_cases(cases, suite="regression", regression_ids=regression)
    assert {c.id for c in reg} == regression

    cap = filter_cases(cases, suite="capability", regression_ids=regression)
    assert not ({c.id for c in cap} & regression)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_cases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.cases'`.

- [ ] **Step 3: Write `evals/cases.py`**

```python
"""Eval case schema, dev/test split and capability/regression suites."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

NEW = "NEW"  # gold target meaning "a new agent should be created"

ROUTING_KINDS = (
    "follow_up",
    "new_task",
    "same_person_new_topic",
    "vague_reference",
    "multi_task",
    "same_name_people",
    "cross_agent",
    "dormant_revival",
    "typo",
    "agent_result",
)
SCENARIO_KINDS = ("injection", "correction_midrun")  # L3, built in Plan 4
KINDS = ROUTING_KINDS + SCENARIO_KINDS

REGRESSION_FILE = Path(__file__).resolve().parent / "suites" / "regression.txt"


@dataclass
class Case:
    """One message plus what a correct router must do with it."""

    id: str
    kind: str
    world_seed: int
    message: str
    gold_targets: list[str]
    message_type: str = "user"  # "user" or "agent"
    context: list[tuple[str, str]] = field(default_factory=list)  # (tag, text) pairs
    acceptable_ids: list[str] = field(default_factory=list)
    clarify_ok: bool = False
    dispatch_optional: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown case kind {self.kind!r}")
        if self.message_type not in ("user", "agent"):
            raise ValueError(f"unknown message_type {self.message_type!r}")

    @property
    def expected_ids(self) -> list[str]:
        return [target for target in self.gold_targets if target != NEW]

    @property
    def expects_new(self) -> bool:
        return NEW in self.gold_targets

    @property
    def expects_reuse(self) -> bool:
        return bool(self.expected_ids) and not self.dispatch_optional

    @property
    def split(self) -> str:
        """Stable 70/30 dev/test split derived from the id, so it never drifts."""
        digest = int(hashlib.sha1(self.id.encode("utf-8")).hexdigest(), 16)
        return "dev" if digest % 10 < 7 else "test"


def load_regression_ids(path: Path = REGRESSION_FILE) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def filter_cases(
    cases: Iterable[Case],
    *,
    split: str | None = None,
    suite: str = "all",
    kinds: Iterable[str] | None = None,
    regression_ids: set[str] | None = None,
) -> list[Case]:
    if suite not in ("all", "capability", "regression"):
        raise ValueError(f"unknown suite {suite!r}")
    regression = load_regression_ids() if regression_ids is None else regression_ids
    wanted_kinds = set(kinds) if kinds else None
    selected: list[Case] = []
    for case in cases:
        if split and case.split != split:
            continue
        if wanted_kinds and case.kind not in wanted_kinds:
            continue
        if suite == "regression" and case.id not in regression:
            continue
        if suite == "capability" and case.id in regression:
            continue
        selected.append(case)
    return selected
```

- [ ] **Step 4: Create `evals/suites/regression.txt`**

```text
# Case ids that pass and must keep passing (one per line).
# Every fixed failure is added here. Edited by hand only; protected by evals/PROTECTED.sha256.
```

- [ ] **Step 5: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_cases.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add evals/cases.py evals/suites/regression.txt evals/tests/test_cases.py
git commit -m "Add eval case schema with splits and suites"
```

---

### Task 3: World model and seeded generator

Correct answers are known by construction. Core agents are the ones cases are about. Filler agents are distractors, and they never give a core person an agent on a core topic, so every gold label stays correct at every roster size.

**Files:**
- Create: `evals/worlds/__init__.py` (empty), `evals/worlds/model.py`, `evals/worlds/generator.py`
- Test: `evals/tests/test_world_generator.py`

- [ ] **Step 1: Write the failing test**

```python
from evals.cases import NEW, ROUTING_KINDS
from evals.worlds.generator import MAX_AGENTS, NEW_PEOPLE, build_world, generate_cases


def test_world_is_deterministic():
    assert [a.name for a in build_world(1).agents] == [a.name for a in build_world(1).agents]
    assert [a.name for a in build_world(1).agents] != [a.name for a in build_world(2).agents]


def test_full_world_has_unique_ids_and_names():
    world = build_world(2)
    assert len(world.agents) == MAX_AGENTS
    assert len({a.id for a in world.agents}) == MAX_AGENTS
    assert len({a.name for a in world.agents}) == MAX_AGENTS


def test_smaller_worlds_are_prefixes_that_keep_every_core_agent():
    world = build_world(3)
    small, mid = world.truncated(10), world.truncated(200)
    core_ids = {a.id for a in world.agents if a.core}
    assert len(small.agents) == 10 and len(mid.agents) == 200
    assert core_ids <= {a.id for a in small.agents}
    assert {a.id for a in small.agents} <= {a.id for a in mid.agents}


def test_twins_share_a_first_name_but_not_the_main_topic():
    world = build_world(4)
    main, twin = world.role_people["main"], world.role_people["twin"]
    assert main.first == twin.first and main.last != twin.last
    main_topic = world.by_id()[world.roles["main"]].topic
    assert main_topic not in {a.topic for a in world.agents if a.person == twin.full}


def test_no_filler_agent_competes_with_a_core_person_and_topic():
    world = build_world(5)
    core_pairs = {(a.person, a.topic) for a in world.agents if a.core and a.person}
    filler_pairs = {(a.person, a.topic) for a in world.agents if not a.core and a.person}
    assert not core_pairs & filler_pairs


def test_cases_cover_every_routing_kind_with_valid_gold():
    for seed in (1, 2, 3):
        world = build_world(seed)
        cases = generate_cases(world, per_kind=4)
        assert {c.kind for c in cases} == set(ROUTING_KINDS)
        assert len({c.id for c in cases}) == len(cases)
        ids = world.by_id()
        for case in cases:
            assert all(t == NEW or t in ids for t in case.gold_targets), case.id
            assert all(t in ids for t in case.acceptable_ids), case.id


def test_create_cases_really_have_no_existing_agent():
    world = build_world(6)
    new_people = {f"{first} {last}" for first, last in NEW_PEOPLE}
    assert not any(a.person in new_people for a in world.agents)
    main = world.role_people["main"]
    reserved = world.reserved_topic[0]
    assert not any(a.person == main.full and a.topic == reserved for a in world.agents)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_world_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.worlds'`.

- [ ] **Step 3: Write `evals/worlds/model.py`**

```python
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
```

- [ ] **Step 4: Write `evals/worlds/generator.py`**

```python
"""Seeded synthetic worlds whose correct routing answers are known by construction.

A world has a fixed core (the agents cases are about) plus filler agents that act as
distractors. Filler never gives a core person an agent on a core topic, so gold labels stay
correct at every roster size, and smaller worlds are prefixes of the full one.
"""

from __future__ import annotations

import random
import re

from ..cases import NEW, Case
from .model import AgentSpec, Person, World

MAX_AGENTS = 1000

PEOPLE = [
    ("Marcus", "Lee"), ("Priya", "Shah"), ("Diego", "Ramos"), ("Hana", "Sato"),
    ("Omar", "Haddad"), ("Lena", "Fischer"), ("Ravi", "Iyer"), ("Chloe", "Martin"),
    ("Kenji", "Mori"), ("Zoe", "Turner"), ("Ivan", "Petrov"), ("Maya", "Cohen"),
    ("Tom", "Becker"), ("Sofia", "Rossi"), ("Dana", "Kim"), ("Felix", "Wagner"),
    ("Grace", "Liu"), ("Hugo", "Lambert"), ("Iris", "Novak"), ("Jonas", "Berg"),
    ("Kira", "Nair"), ("Leo", "Costa"), ("Mila", "Horvat"), ("Noah", "Fraser"),
    ("Olga", "Ivanova"), ("Pablo", "Ortega"), ("Quinn", "Harper"), ("Rosa", "Mendez"),
    ("Seth", "Cole"), ("Tara", "Singh"), ("Uma", "Rao"), ("Victor", "Hale"),
    ("Wendy", "Zhou"), ("Yusuf", "Aydin"),
]
TWIN_PAIRS = [
    (("Alice", "Chen"), ("Alice", "Wong")),
    (("Sam", "Ortiz"), ("Sam", "Patel")),
    (("Ben", "Carter"), ("Ben", "Okafor")),
]
NEW_PEOPLE = [("Nadia", "Rahman"), ("Theo", "Brooks"), ("Ines", "Duarte"), ("Kwame", "Mensah")]
COMPANIES = ["Acme", "Globex", "Initech", "Hooli", "Umbrella", "Stark", "Wayne", "Soylent"]
CITIES = ["Tokyo", "Lisbon", "Austin", "Seoul", "Berlin", "Toronto", "Denver", "Oslo", "Lima", "Cairo", "Perth", "Dublin"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# (phrase, short form)
TOPICS = [
    ("contract renewal", "renewal"), ("offsite planning", "offsite"), ("intro call", "intro"),
    ("invoice dispute", "invoice"), ("reference letter", "reference"), ("apartment lease", "lease"),
    ("conference talk", "talk"), ("podcast interview", "podcast"), ("job offer", "offer"),
    ("wedding RSVP", "RSVP"), ("quarterly report", "report"), ("board deck review", "deck"),
]
CROSS_TOPIC = ("pricing approval", "pricing")
RESERVED_TOPICS = [("charity gala", "gala"), ("book club", "book club"), ("mentorship chat", "mentorship")]

NAME_TEMPLATES = (
    "Email to {first}",
    "{first} {short}",
    "{topic} with {first}",
    "Reply to {first} re {short}",
    "{first} {last} {short}",
)
GENERIC_TEMPLATES = (
    "Flight to {city}",
    "{city} hotels",
    "Expense report {month}",
    "{company} invoice {month}",
    "Newsletter cleanup {month}",
    "Dentist appointment {month}",
)
NEW_TASKS = (
    "book a table for four at an Italian place this Friday",
    "find a birthday gift for my mom under $50",
    "look up the return policy for my headphones",
    "find a plumber who can come tomorrow morning",
)
FOLLOW_UPS = (
    "add a P.S. to the {phrase} email to {name} saying I'm free Thursday",
    "can you make the {phrase} email to {name} shorter",
    "tell {name} I need one more day on the {phrase}",
    "update the {phrase} draft for {name} to mention the budget",
)
VAGUE = ("any reply yet?", "did they get back to us?", "anything back on that?")
CROSS = (
    "let {main} know {colleague} signed off on the pricing",
    "tell {main} that {colleague} approved the pricing",
)
DORMANT = ("any update on the {phrase} with {first}?", "whatever happened with {first}'s {phrase}?")
TYPO = ("can you nudge {name} on the {phrase}?", "did {name} answer about the {phrase}?")
TWIN_VAGUE = ("did {first} get back to me?", "what's the latest with {first}?")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "agent"


def _typo(name: str, rng: random.Random) -> str:
    index = rng.randrange(1, len(name) - 2)
    swapped = name[:index] + name[index + 1] + name[index] + name[index + 2 :]
    return swapped if swapped != name else name[:index] + name[index + 1 :]


class _Builder:
    """Accumulates agents with unique names, ids and (person, topic, month) pairs."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.agents: list[AgentSpec] = []
        self._names: set[str] = set()
        self._ids: set[str] = set()
        self._pairs: set[tuple[str, str, str | None]] = set()

    def _unique_id(self, name: str) -> str:
        base = candidate = _slug(name)
        suffix = 2
        while candidate in self._ids:
            candidate = f"{base}-{suffix}"
            suffix += 1
        self._ids.add(candidate)
        return candidate

    def add(
        self,
        *,
        name: str,
        purpose: str,
        entities: list[str],
        days: float,
        person: str | None = None,
        topic: str | None = None,
        core: bool = False,
        history: list[tuple[str, str]] | None = None,
    ) -> AgentSpec | None:
        if name in self._names:
            return None
        agent = AgentSpec(
            id=self._unique_id(name),
            name=name,
            purpose=purpose,
            entities=entities,
            last_active_days=round(days, 2),
            created_days=round(days + self.rng.uniform(0, 30), 2),
            person=person,
            topic=topic,
            core=core,
            history=history or [],
        )
        self._names.add(name)
        self.agents.append(agent)
        return agent

    def add_person_agent(
        self,
        person: Person,
        topic: tuple[str, str],
        *,
        days: float,
        core: bool = False,
        month: str | None = None,
    ) -> AgentSpec | None:
        phrase, short = topic
        key = (person.full, phrase, month)
        if key in self._pairs:
            return None
        templates = list(NAME_TEMPLATES)
        self.rng.shuffle(templates)
        for template in templates:
            name = template.format(first=person.first, last=person.last, short=short, topic=phrase.title())
            if month:
                name = f"{name} ({month})"
            if name in self._names:
                continue
            suffix = f" ({month} thread)" if month else ""
            agent = self.add(
                name=name,
                purpose=f"{phrase.capitalize()} with {person.full} at {person.company}{suffix}",
                entities=[person.full.lower(), person.first.lower(), person.email],
                days=days,
                person=person.full,
                topic=phrase,
                core=core,
                history=[(f"Email {person.full} about the {phrase}", f"Draft created for {person.email} about the {phrase}.")],
            )
            self._pairs.add(key)
            return agent
        return None


def build_world(seed: int, max_agents: int = MAX_AGENTS) -> World:
    rng = random.Random(seed)
    twin_a, twin_b = rng.choice(TWIN_PAIRS)
    companies = list(COMPANIES)
    rng.shuffle(companies)
    pool = list(PEOPLE)
    rng.shuffle(pool)

    main = Person(*twin_a, companies[0])
    twin = Person(*twin_b, companies[1])
    colleague = Person(*pool.pop(), companies[0])
    typo_index = next(i for i, (first, _) in enumerate(pool) if len(first) >= 5)
    typo = Person(*pool.pop(typo_index), rng.choice(companies))
    dormant = Person(*pool.pop(), rng.choice(companies))
    other = Person(*pool.pop(), rng.choice(companies))
    filler_people = [Person(first, last, rng.choice(companies)) for first, last in pool]

    topics = list(TOPICS)
    rng.shuffle(topics)
    t_main, t_main_neg, t_twin, t_dormant, t_typo, t_other = topics[:6]

    builder = _Builder(rng)
    roles: dict[str, str] = {}

    def core(role: str, person: Person, topic: tuple[str, str], low: float, high: float) -> None:
        agent = builder.add_person_agent(person, topic, days=rng.uniform(low, high), core=True)
        assert agent is not None
        roles[role] = agent.id

    core("main", main, t_main, 0.5, 4)
    core("main_neg", main, t_main_neg, 5, 20)
    core("twin", twin, t_twin, 1, 10)
    core("colleague", colleague, CROSS_TOPIC, 1, 6)
    core("dormant", dormant, t_dormant, 35, 60)
    core("typo", typo, t_typo, 1, 8)
    core("other", other, t_other, 0.5, 6)
    city = rng.choice(CITIES)
    flight = builder.add(
        name=f"Flight to {city}", purpose=f"Book a flight to {city}", entities=[city.lower()],
        days=rng.uniform(2, 15), core=True,
    )
    assert flight is not None
    roles["generic"] = flight.id

    core_people = [main, twin, colleague, typo, dormant, other]
    core_names = {person.full for person in core_people}
    core_topics = {t[0] for t in (t_main, t_main_neg, t_twin, t_dormant, t_typo, t_other, CROSS_TOPIC)}
    everyone = filler_people + core_people
    filler_person_agents: list[AgentSpec] = []

    attempts = 0
    while len(builder.agents) < max_agents:
        attempts += 1
        if attempts > max_agents * 200:
            raise RuntimeError("world generator could not reach max_agents")
        days = rng.uniform(0, 30) if rng.random() < 0.6 else rng.uniform(30, 120)
        roll = rng.random()
        if roll < 0.6:
            person, topic = rng.choice(everyone), rng.choice(TOPICS)
            if person.full in core_names and topic[0] in core_topics:
                continue
            month = None if rng.random() < 0.5 else rng.choice(MONTHS)
            agent = builder.add_person_agent(person, topic, days=days, month=month)
            if agent:
                filler_person_agents.append(agent)
        elif roll < 0.65 and filler_person_agents:
            # Near-duplicate of an existing filler agent (failure mode F2), never a gold target.
            source = rng.choice(filler_person_agents)
            variant = source.name.lower() if source.name.lower() != source.name else f"Re: {source.name}"
            builder.add(
                name=variant, purpose=source.purpose, entities=list(source.entities), days=days,
                person=source.person, topic=source.topic, history=list(source.history),
            )
        else:
            name = rng.choice(GENERIC_TEMPLATES).format(
                city=rng.choice(CITIES), month=rng.choice(MONTHS), company=rng.choice(companies)
            )
            builder.add(name=name, purpose=name, entities=[], days=days)

    return World(
        seed=seed,
        agents=builder.agents,
        roles=roles,
        role_people={
            "main": main, "main_neg": main, "twin": twin, "colleague": colleague,
            "dormant": dormant, "typo": typo, "other": other,
        },
        reserved_topic=rng.choice(RESERVED_TOPICS),
    )


def generate_cases(world: World, per_kind: int) -> list[Case]:
    """per_kind cases of every routing kind, all about the world's core agents."""
    rng = random.Random(world.seed * 7919 + 17)
    agents = world.by_id()
    cases: list[Case] = []

    def target(role: str) -> tuple[AgentSpec, Person]:
        return agents[world.roles[role]], world.role_people[role]

    def add(kind: str, index: int, message: str, gold: list[str], **extra) -> None:
        cases.append(Case(
            id=f"{kind}-s{world.seed}-{index:02d}", kind=kind, world_seed=world.seed,
            message=message, gold_targets=gold, **extra,
        ))

    assert world.reserved_topic is not None
    for i in range(per_kind):
        role = ("main", "colleague", "typo", "other")[i % 4]
        agent, person = target(role)
        name = person.full if role == "main" else person.first
        add("follow_up", i, rng.choice(FOLLOW_UPS).format(phrase=agent.topic, name=name), [agent.id])

        if i % 2 == 0:
            first, last = rng.choice(NEW_PEOPLE)
            add("new_task", i, f"email {first} {last} to set up a {rng.choice(TOPICS)[0]}", [NEW])
        else:
            add("new_task", i, rng.choice(NEW_TASKS), [NEW])

        main = world.role_people["main"]
        phrase = world.reserved_topic[0]
        if i % 2 == 0:
            add("same_person_new_topic", i, f"email {main.full} about the {phrase}", [NEW])
        else:
            add("same_person_new_topic", i, f"email {main.first} about the {phrase}", [NEW], clarify_ok=True)

        agent, person = target(("main", "other")[i % 2])
        if i % 4 < 2:
            context = [
                ("user_message", f"email {person.full} about the {agent.topic}"),
                ("poke_reply", f"drafted it for {person.first}, want me to send it?"),
                ("user_message", "yes send it"),
                ("poke_reply", "sent"),
            ]
        else:
            context = [
                ("conversation_summary", f"Earlier the user asked to email {person.full} about the {agent.topic}; the email was sent on Oct 24."),
                ("user_message", "thanks"),
                ("poke_reply", "anytime"),
            ]
        add("vague_reference", i, rng.choice(VAGUE), [agent.id], context=context)

        role = ("main", "other")[i % 2]
        agent, person = target(role)
        name = person.full if role == "main" else person.first
        follow = rng.choice(FOLLOW_UPS).format(phrase=agent.topic, name=name)
        add("multi_task", i, f"{follow}, and also {rng.choice(NEW_TASKS)}", [agent.id, NEW])

        agent, person = target("twin")
        if i % 2 == 0:
            add("same_name_people", i, f"follow up with {person.first} about the {agent.topic}", [agent.id])
        else:
            add("same_name_people", i, rng.choice(TWIN_VAGUE).format(first=person.first), [], clarify_ok=True)

        main_agent, main_person = target("main")
        colleague_agent, colleague_person = target("colleague")
        add(
            "cross_agent", i,
            rng.choice(CROSS).format(main=main_person.full, colleague=colleague_person.first),
            [main_agent.id], acceptable_ids=[colleague_agent.id],
        )

        agent, person = target("dormant")
        add("dormant_revival", i, rng.choice(DORMANT).format(phrase=agent.topic, first=person.first), [agent.id])

        agent, person = target("typo")
        add("typo", i, rng.choice(TYPO).format(name=_typo(person.first, rng), phrase=agent.topic), [agent.id])

        agent, person = target(("main", "colleague", "other", "dormant")[i % 4])
        add(
            "agent_result", i,
            f"[SUCCESS] {agent.name}: {person.first} replied on the {agent.topic} thread asking to move the call to Thursday.",
            [agent.id], message_type="agent", dispatch_optional=True,
        )
    return cases
```

- [ ] **Step 5: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_world_generator.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add evals/worlds/__init__.py evals/worlds/model.py evals/worlds/generator.py evals/tests/test_world_generator.py
git commit -m "Add seeded world generator with gold labels by construction"
```

---

### Task 4: Hand-written world and cases

A small realistic set that checks whether the synthetic data is realistic, reported separately. This task adds 10 cases, one per routing kind; Samin grows it to about 50 in Plan 5.

**Files:**
- Create: `evals/worlds/handwritten_cases.yaml`, `evals/worlds/handwritten.py`
- Test: `evals/tests/test_handwritten.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from evals.cases import ROUTING_KINDS
from evals.worlds.handwritten import load_handwritten


def test_handwritten_set_loads_and_covers_every_kind():
    world, cases = load_handwritten()
    assert len(world.agents) >= 10
    assert {c.kind for c in cases} == set(ROUTING_KINDS)


def test_unknown_agent_id_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "agents:\n"
        "  - {id: a, name: A, purpose: p, last_active_days: 1, created_days: 2}\n"
        "cases:\n"
        "  - {id: hw-x, kind: follow_up, message: m, gold_targets: [missing]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        load_handwritten(path)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_handwritten.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.worlds.handwritten'`.

- [ ] **Step 3: Write `evals/worlds/handwritten_cases.yaml`**

```yaml
# Hand-written world and cases (synthetic-validity check). Protected file: edit by hand only.
agents:
  - {id: email-to-alice, name: "Email to Alice", purpose: "Contract renewal with Alice Chen at Acme", entities: [alice chen, alice, alice.chen@acme.com], last_active_days: 2, created_days: 9, person: "Alice Chen", topic: "contract renewal"}
  - {id: email-to-alice-2, name: "email to alice", purpose: "Contract renewal with Alice Chen at Acme", entities: [alice chen, alice, alice.chen@acme.com], last_active_days: 20, created_days: 21, person: "Alice Chen", topic: "contract renewal"}
  - {id: alice-offsite, name: "Alice offsite", purpose: "Offsite planning with Alice Chen at Acme", entities: [alice chen, alice, alice.chen@acme.com], last_active_days: 12, created_days: 30, person: "Alice Chen", topic: "offsite planning"}
  - {id: intro-to-alice-wong, name: "Intro to Alice Wong", purpose: "Intro call with Alice Wong at Hooli", entities: [alice wong, alice, alice.wong@hooli.com], last_active_days: 6, created_days: 14, person: "Alice Wong", topic: "intro call"}
  - {id: bob-pricing, name: "Bob pricing", purpose: "Pricing approval with Bob Diaz at Acme", entities: [bob diaz, bob, bob.diaz@acme.com], last_active_days: 3, created_days: 10, person: "Bob Diaz", topic: "pricing approval"}
  - {id: priya-job-offer, name: "Priya job offer", purpose: "Job offer negotiation with Priya Shah at Globex", entities: [priya shah, priya, priya.shah@globex.com], last_active_days: 41, created_days: 55, person: "Priya Shah", topic: "job offer"}
  - {id: marcus-invoice, name: "Marcus invoice", purpose: "Invoice dispute with Marcus Lee at Initech", entities: [marcus lee, marcus, marcus.lee@initech.com], last_active_days: 4, created_days: 18, person: "Marcus Lee", topic: "invoice dispute"}
  - {id: q3-budget, name: "Q3 Budget", purpose: "Quarterly report with the finance team", entities: [finance], last_active_days: 8, created_days: 40}
  - {id: flight-to-tokyo, name: "Flight to Tokyo", purpose: "Book a flight to Tokyo", entities: [tokyo], last_active_days: 15, created_days: 16}
  - {id: tokyo-hotels, name: "Tokyo hotels", purpose: "Find hotels in Tokyo", entities: [tokyo], last_active_days: 15, created_days: 16}
  - {id: weekly-digest, name: "Weekly digest", purpose: "Weekly inbox digest every Monday", entities: [], last_active_days: 1, created_days: 60}
  - {id: dentist-appointment, name: "Dentist appointment", purpose: "Book a dentist cleaning", entities: [dentist], last_active_days: 30, created_days: 31}

cases:
  - id: hw-01-follow-up
    kind: follow_up
    message: "add a P.S. to the renewal email to Alice saying I can do Thursday"
    gold_targets: [email-to-alice]
    acceptable_ids: [email-to-alice-2]
  - id: hw-02-new-task
    kind: new_task
    message: "find a plumber who can come tomorrow morning"
    gold_targets: [NEW]
  - id: hw-03-same-person-new-topic
    kind: same_person_new_topic
    message: "email Alice Chen about joining our book club"
    gold_targets: [NEW]
  - id: hw-04-vague-reference
    kind: vague_reference
    message: "any reply yet?"
    context:
      - [user_message, "send the renewal email to Alice Chen"]
      - [poke_reply, "sent it"]
    gold_targets: [email-to-alice]
    acceptable_ids: [email-to-alice-2]
  - id: hw-05-multi-task
    kind: multi_task
    message: "tell Marcus we'll pay the undisputed part of the invoice, and find me a birthday gift for my mom under $50"
    gold_targets: [marcus-invoice, NEW]
  - id: hw-06-same-name-people
    kind: same_name_people
    message: "did Alice get back to me?"
    gold_targets: []
    clarify_ok: true
  - id: hw-07-cross-agent
    kind: cross_agent
    message: "let Alice Chen know Bob signed off on the pricing"
    gold_targets: [email-to-alice]
    acceptable_ids: [email-to-alice-2, bob-pricing]
  - id: hw-08-dormant-revival
    kind: dormant_revival
    message: "any news on Priya's offer?"
    gold_targets: [priya-job-offer]
  - id: hw-09-typo
    kind: typo
    message: "nudge Marcos about the invoice"
    gold_targets: [marcus-invoice]
  - id: hw-10-agent-result
    kind: agent_result
    message_type: agent
    message: "[SUCCESS] Bob pricing: Bob replied that pricing is approved through Q2."
    gold_targets: [bob-pricing]
    dispatch_optional: true
```

- [ ] **Step 4: Write `evals/worlds/handwritten.py`**

```python
"""Load the hand-written world and cases, validating every referenced agent id."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..cases import NEW, Case
from .model import AgentSpec, World

HANDWRITTEN_PATH = Path(__file__).resolve().parent / "handwritten_cases.yaml"


def load_handwritten(path: Path = HANDWRITTEN_PATH) -> tuple[World, list[Case]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    agents = [
        AgentSpec(
            id=entry["id"],
            name=entry["name"],
            purpose=entry["purpose"],
            entities=list(entry.get("entities", [])),
            last_active_days=float(entry["last_active_days"]),
            created_days=float(entry["created_days"]),
            person=entry.get("person"),
            topic=entry.get("topic"),
            core=True,
        )
        for entry in raw["agents"]
    ]
    world = World(seed=0, agents=agents)
    cases = [
        Case(
            id=entry["id"],
            kind=entry["kind"],
            world_seed=0,
            message=entry["message"],
            gold_targets=list(entry.get("gold_targets", [])),
            message_type=entry.get("message_type", "user"),
            context=[(str(tag), str(text)) for tag, text in entry.get("context", [])],
            acceptable_ids=list(entry.get("acceptable_ids", [])),
            clarify_ok=bool(entry.get("clarify_ok", False)),
            dispatch_optional=bool(entry.get("dispatch_optional", False)),
        )
        for entry in raw["cases"]
    ]
    known = world.by_id()
    for case in cases:
        for target in [*case.gold_targets, *case.acceptable_ids]:
            if target != NEW and target not in known:
                raise ValueError(f"{case.id}: unknown agent id {target!r}")
    return world, cases
```

- [ ] **Step 5: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_handwritten.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add evals/worlds/handwritten_cases.yaml evals/worlds/handwritten.py evals/tests/test_handwritten.py
git commit -m "Add hand-written eval world and cases"
```

---

### Task 5: Outcome grading

**Files:**
- Create: `evals/grading.py`
- Test: `evals/tests/test_grading.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from evals.cases import NEW, Case
from evals.grading import Decision, grade


def case(kind="follow_up", gold=("a",), **kw):
    return Case(id="c", kind=kind, world_seed=1, message="m", gold_targets=list(gold), **kw)


@pytest.mark.parametrize(
    "the_case, decision, expected",
    [
        (case(), Decision(dispatched_ids=["a"]), "correct"),
        (case(), Decision(dispatched_ids=["b"]), "wrong_reuse"),
        (case(), Decision(created_names=["Email to Alice v2"]), "duplicate_create"),
        (case(), Decision(dispatched_ids=["a"], created_names=["x"]), "extra_create"),
        (case(), Decision(), "no_action"),
        (case(), Decision(clarified=True), "unnecessary_clarify"),
        (case(kind="new_task", gold=[NEW]), Decision(created_names=["Plumber"]), "correct"),
        (case(kind="new_task", gold=[NEW]), Decision(dispatched_ids=["a"]), "missed_create"),
        (case(kind="multi_task", gold=["a", NEW]), Decision(dispatched_ids=["a"], created_names=["Gift"]), "correct"),
        (case(kind="multi_task", gold=["a", NEW]), Decision(dispatched_ids=["a"]), "partial"),
        (case(kind="same_name_people", gold=[], clarify_ok=True), Decision(clarified=True), "correct"),
        (case(kind="same_name_people", gold=[], clarify_ok=True), Decision(dispatched_ids=["a"]), "should_clarify"),
        (case(kind="same_name_people", gold=[], clarify_ok=True), Decision(), "should_clarify"),
        (case(kind="same_person_new_topic", gold=[NEW], clarify_ok=True), Decision(clarified=True), "correct"),
        (case(kind="cross_agent", acceptable_ids=["b"]), Decision(dispatched_ids=["b"]), "correct"),
        (case(kind="agent_result", dispatch_optional=True), Decision(), "correct"),
        (case(kind="agent_result", dispatch_optional=True), Decision(dispatched_ids=["a"]), "correct"),
        (case(kind="agent_result", dispatch_optional=True), Decision(dispatched_ids=["b"]), "wrong_reuse"),
        (case(kind="agent_result", dispatch_optional=True), Decision(created_names=["x"]), "duplicate_create"),
    ],
)
def test_grade(the_case, decision, expected):
    assert grade(the_case, decision) == expected
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_grading.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.grading'`.

- [ ] **Step 3: Write `evals/grading.py`**

```python
"""Grade a routing decision by outcome: which agents got work, not how the model got there."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cases import Case

CORRECT = "correct"
WRONG_REUSE = "wrong_reuse"  # sent work to an agent that doesn't own it
DUPLICATE_CREATE = "duplicate_create"  # created a new agent when one should be reused
MISSED_CREATE = "missed_create"  # reused an agent when a new one was needed
EXTRA_CREATE = "extra_create"  # right agent, plus an unneeded new one
PARTIAL = "partial"  # multi-target case only half done
SHOULD_CLARIFY = "should_clarify"  # ambiguous request; guessing or doing nothing is wrong
UNNECESSARY_CLARIFY = "unnecessary_clarify"
NO_ACTION = "no_action"
ERROR = "error"


@dataclass
class Decision:
    """What one Interaction Agent turn did, as seen through intercepted tools."""

    dispatched_ids: list[str] = field(default_factory=list)
    created_names: list[str] = field(default_factory=list)
    clarified: bool = False
    user_messages: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_keys: list[str] = field(default_factory=list)


def grade(case: Case, decision: Decision) -> str:
    predicted = set(decision.dispatched_ids)
    created = bool(decision.created_names)
    expected = set(case.expected_ids)
    acceptable = set(case.acceptable_ids)

    if not predicted and not created:
        if decision.clarified:
            return CORRECT if case.clarify_ok else UNNECESSARY_CLARIFY
        if case.dispatch_optional:
            return CORRECT
        return SHOULD_CLARIFY if case.clarify_ok and not case.gold_targets else NO_ACTION

    if not case.gold_targets:
        return SHOULD_CLARIFY
    if predicted - expected - acceptable:
        return WRONG_REUSE if expected else MISSED_CREATE
    if case.dispatch_optional:
        return DUPLICATE_CREATE if created else CORRECT

    unmatched = [g for g in expected if not predicted & ({g} | acceptable)]
    if unmatched:
        return DUPLICATE_CREATE if created and not case.expects_new else PARTIAL
    if created and not case.expects_new:
        return EXTRA_CREATE
    if case.expects_new and not created:
        return PARTIAL if expected else MISSED_CREATE
    return CORRECT
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_grading.py -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/grading.py evals/tests/test_grading.py
git commit -m "Add outcome-based routing grader"
```

---

### Task 6: Routing systems and the baseline

A *routing system* decides what the IA sees in `<active_agents>`, and how a name it writes back maps to an agent id. The baseline must be byte-identical to production.

**Files:**
- Create: `evals/systems/__init__.py`, `evals/systems/base.py`, `evals/systems/baseline.py`
- Test: `evals/tests/test_baseline_system.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

import server.agents.interaction_agent.agent as ia_agent
from evals.systems import get_system
from evals.systems.baseline import BaselineSystem
from evals.worlds.generator import build_world
from evals.worlds.model import World


class FakeRoster:
    def __init__(self, names):
        self._names = names

    def load(self):
        pass

    def get_agents(self):
        return list(self._names)


def test_baseline_matches_production_rendering(monkeypatch):
    world = build_world(1).truncated(25)
    names = [a.name for a in world.roster_order()]
    monkeypatch.setattr(ia_agent, "get_agent_roster", lambda: FakeRoster(names))
    production = f"<active_agents>\n{ia_agent._render_active_agents()}\n</active_agents>"
    assert BaselineSystem().render_roster(world, None) == production


def test_empty_roster_renders_none():
    assert BaselineSystem().render_roster(World(seed=0, agents=[]), None) == "<active_agents>\nNone\n</active_agents>"


def test_shows_every_agent_in_roster_order():
    world = build_world(1).truncated(30)
    assert BaselineSystem().shown_ids(world, None) == [a.id for a in world.roster_order()]


def test_resolve_name_is_exact_like_production():
    world = build_world(1).truncated(30)
    agent = world.agents[0]
    system = BaselineSystem()
    assert system.resolve_name(world, agent.name) == agent.id
    assert system.resolve_name(world, agent.name.upper() + "!") is None


def test_get_system():
    assert get_system("baseline").name == "baseline"
    with pytest.raises(ValueError):
        get_system("nope")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_baseline_system.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.systems'`.

- [ ] **Step 3: Write `evals/systems/base.py`**

```python
"""The interface every routing system under test implements."""

from __future__ import annotations

from typing import Protocol

from ..cases import Case
from ..worlds.model import World


class RoutingSystem(Protocol):
    name: str

    def render_roster(self, world: World, case: Case | None) -> str:
        """The full <active_agents>…</active_agents> section for the IA prompt."""

    def shown_ids(self, world: World, case: Case | None) -> list[str]:
        """Agent ids visible in that section, in display order (for L1)."""

    def resolve_name(self, world: World, name: str) -> str | None:
        """Map a name the IA passed to send_message_to_agent to an agent id; None = creates."""
```

- [ ] **Step 4: Write `evals/systems/baseline.py`**

```python
"""Today's behaviour: every agent ever created, as a bare name, matched by exact string."""

from __future__ import annotations

from html import escape

from ..cases import Case
from ..worlds.model import World


class BaselineSystem:
    name = "baseline"

    def render_roster(self, world: World, case: Case | None) -> str:
        agents = world.roster_order()
        if not agents:
            body = "None"
        else:
            body = "\n".join(f'<agent name="{escape(a.name or "agent", quote=True)}" />' for a in agents)
        return f"<active_agents>\n{body}\n</active_agents>"

    def shown_ids(self, world: World, case: Case | None) -> list[str]:
        return [agent.id for agent in world.roster_order()]

    def resolve_name(self, world: World, name: str) -> str | None:
        for agent in world.agents:
            if agent.name == name:
                return agent.id
        return None
```

- [ ] **Step 5: Write `evals/systems/__init__.py`**

```python
"""Routing systems under test, by name."""

from __future__ import annotations

from .base import RoutingSystem
from .baseline import BaselineSystem

SYSTEMS: dict[str, type] = {"baseline": BaselineSystem}


def get_system(name: str) -> RoutingSystem:
    try:
        return SYSTEMS[name]()
    except KeyError:
        raise ValueError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}") from None


__all__ = ["RoutingSystem", "SYSTEMS", "get_system"]
```

- [ ] **Step 6: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_baseline_system.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add evals/systems evals/tests/test_baseline_system.py
git commit -m "Add routing system interface and production-identical baseline"
```

---

### Task 7: LLM client with cache, budget and retries

**Files:**
- Create: `evals/llm.py`
- Test: `evals/tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
import json

import httpx
import pytest

from evals.llm import Budget, BudgetExceeded, OpenRouterLLM, ScriptedLLM, cache_key, estimate_cost, tool_call

KW = dict(model="m", system="s", messages=[{"role": "user", "content": "x"}], tools=[], sample=0)


def _ok(cost=0.01):
    return {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": cost},
    }


def _llm(tmp_path, handler, cap=1.0):
    return OpenRouterLLM(
        api_key="k", cache_dir=tmp_path, budget=Budget(cap),
        transport=httpx.MockTransport(handler), retry_delay_s=0,
    )


async def test_identical_call_is_served_from_cache(tmp_path):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_ok())

    llm = _llm(tmp_path, handler)
    first = await llm.complete(**KW)
    second = await llm.complete(**KW)
    await llm.aclose()

    assert len(sent) == 1
    assert first["_meta"]["cached"] is False and second["_meta"]["cached"] is True
    assert llm.budget.spent_usd == pytest.approx(0.01)
    assert sent[0]["usage"] == {"include": True}
    assert sent[0]["messages"][0] == {"role": "system", "content": "s"}


async def test_each_sample_is_a_separate_call(tmp_path):
    sent = []
    llm = _llm(tmp_path, lambda r: sent.append(1) or httpx.Response(200, json=_ok()))
    await llm.complete(**KW)
    await llm.complete(**{**KW, "sample": 1})
    await llm.aclose()
    assert len(sent) == 2


async def test_budget_cap_blocks_uncached_calls(tmp_path):
    llm = _llm(tmp_path, lambda r: httpx.Response(200, json=_ok()), cap=0.0)
    with pytest.raises(BudgetExceeded):
        await llm.complete(**KW)
    await llm.aclose()


async def test_retries_rate_limit_then_succeeds(tmp_path):
    responses = [httpx.Response(429, json={}), httpx.Response(200, json=_ok())]
    llm = _llm(tmp_path, lambda r: responses.pop(0))
    data = await llm.complete(**KW)
    await llm.aclose()
    assert data["choices"][0]["message"]["content"] == "hi"


def test_estimate_cost_when_usage_has_no_cost():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    assert estimate_cost("anthropic/claude-sonnet-4", usage) == pytest.approx(3.0)


def test_cache_key_depends_on_every_input():
    base = cache_key(**KW)
    assert base == cache_key(**KW)
    assert base != cache_key(**{**KW, "model": "other"})
    assert base != cache_key(**{**KW, "messages": [{"role": "user", "content": "y"}]})


async def test_scripted_llm_returns_tool_calls():
    llm = ScriptedLLM(lambda messages, sample: {"content": "", "tool_calls": [tool_call("wait", reason="r")]})
    data = await llm.complete(**KW)
    call = data["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "wait"
    assert json.loads(call["function"]["arguments"]) == {"reason": "r"}
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.llm'`.

- [ ] **Step 3: Write `evals/llm.py`**

```python
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
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_llm.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/llm.py evals/tests/test_llm.py
git commit -m "Add cached, budget-capped OpenRouter client for evals"
```

---

### Task 8: L1 retrieval and L2 routing layers

**Files:**
- Create: `evals/layers/__init__.py` (empty), `evals/layers/retrieval.py`, `evals/layers/routing.py`
- Test: `evals/tests/test_layers.py`

- [ ] **Step 1: Write the failing test**

```python
import json

from evals.grading import grade
from evals.layers.retrieval import run_retrieval_case
from evals.layers.routing import MAX_ITERATIONS, build_messages, run_routing_case
from evals.llm import ScriptedLLM, tool_call
from evals.systems.baseline import BaselineSystem
from evals.worlds.generator import build_world, generate_cases


def setup(kind="follow_up"):
    full = build_world(1)
    world = full.truncated(20)
    case = next(c for c in generate_cases(full, per_kind=2) if c.kind == kind)
    return world, case


def tool_turns(messages):
    return sum(1 for m in messages if m["role"] == "tool")


async def test_dispatch_after_acknowledgement_is_correct():
    world, case = setup()
    target = world.by_id()[case.gold_targets[0]].name

    def script(messages, sample):
        seen = tool_turns(messages)
        if seen == 0:
            return {"content": "", "tool_calls": [tool_call("send_message_to_user", message="on it")]}
        if seen == 1:
            return {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name=target, instructions="do it")]}
        return {"content": ""}

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(script), model="m", sample=0)
    assert decision.dispatched_ids == [case.gold_targets[0]]
    assert decision.created_names == []
    assert decision.user_messages == ["on it"]
    assert decision.iterations == 3
    assert grade(case, decision) == "correct"


async def test_unknown_name_creates_and_stub_says_so():
    world, case = setup()
    captured = []

    def script(messages, sample):
        captured.append(list(messages))
        if tool_turns(messages) == 0:
            return {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name="Totally new", instructions="x")]}
        return {"content": ""}

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(script), model="m", sample=0)
    assert decision.created_names == ["Totally new"]
    assert grade(case, decision) == "duplicate_create"
    stub = json.loads(captured[1][-1]["content"])
    assert stub["result"]["new_agent_created"] is True


async def test_question_without_dispatch_is_a_clarification():
    world, case = setup("same_name_people")

    def script(messages, sample):
        if tool_turns(messages) == 0:
            return {"content": "", "tool_calls": [tool_call("send_message_to_user", message="which one do you mean?")]}
        return {"content": ""}

    decision = await run_routing_case(case, world, BaselineSystem(), ScriptedLLM(script), model="m", sample=0)
    assert decision.clarified is True


async def test_iteration_cap():
    world, case = setup()
    llm = ScriptedLLM(lambda messages, sample: {"content": "", "tool_calls": [tool_call("wait", reason="x")]})
    decision = await run_routing_case(case, world, BaselineSystem(), llm, model="m", sample=0)
    assert decision.iterations == MAX_ITERATIONS


def test_prompt_sections_follow_production_order():
    world, case = setup("vague_reference")
    content = build_messages(case, world, BaselineSystem())[0]["content"]
    assert content.index("<conversation_history>") < content.index("<active_agents>") < content.index("<new_user_message>")
    assert case.message in content


def test_agent_results_use_the_agent_tag():
    world, case = setup("agent_result")
    assert "<new_agent_message>" in build_messages(case, world, BaselineSystem())[0]["content"]


def test_retrieval_baseline_shows_everything():
    world, case = setup()
    row = run_retrieval_case(case, world, BaselineSystem())
    assert row["shown_count"] == 20
    assert row["recall"] == 1.0
    assert row["roster_tokens"] > 0


def test_retrieval_create_cases_have_no_recall():
    world, case = setup("new_task")
    assert run_retrieval_case(case, world, BaselineSystem())["recall"] is None
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_layers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.layers'`.

- [ ] **Step 3: Write `evals/layers/retrieval.py`**

```python
"""L1 retrieval eval: is the right agent among what the system shows? No LLM."""

from __future__ import annotations

from typing import Any

from ..cases import Case
from ..systems.base import RoutingSystem
from ..worlds.model import World


def run_retrieval_case(case: Case, world: World, system: RoutingSystem) -> dict[str, Any]:
    shown = system.shown_ids(world, case)
    block = system.render_roster(world, case)
    recall: float | None = None
    mrr: float | None = None
    targets = case.expected_ids
    if targets:
        positions = {agent_id: index for index, agent_id in enumerate(shown)}
        hits = 0
        best: int | None = None
        for gold in targets:
            found = [positions[option] for option in (gold, *case.acceptable_ids) if option in positions]
            if found:
                hits += 1
                best = min(found) if best is None else min(best, *found)
        recall = hits / len(targets)
        mrr = 0.0 if best is None else 1.0 / (best + 1)
    return {"shown_count": len(shown), "roster_tokens": len(block) // 4, "recall": recall, "mrr": mrr}
```

- [ ] **Step 4: Write `evals/layers/routing.py`**

```python
"""L2 routing eval: one real Interaction Agent turn, with its tools intercepted.

The prompt is assembled with the server's own helpers, so the model sees exactly what
production sends, except that the roster section comes from the system under test.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from server.agents.interaction_agent.agent import (
    _render_conversation_history,
    _render_current_turn,
    build_system_prompt,
)
from server.agents.interaction_agent.tools import get_tool_schemas

from ..cases import Case
from ..grading import Decision
from ..llm import LLMClient
from ..systems.base import RoutingSystem
from ..worlds.model import World

# Production allows 8 iterations; routing decisions happen in the first few.
MAX_ITERATIONS = 4


def render_context(case: Case) -> str:
    parts: list[str] = []
    for index, (tag, text) in enumerate(case.context):
        safe = escape(text, quote=False)
        if tag == "conversation_summary":
            parts.append(f"<conversation_summary>{safe}</conversation_summary>")
        else:
            timestamp = f"2026-10-{24 + index // 2} 10:{index:02d}:00"
            parts.append(f'<{tag} timestamp="{timestamp}">{safe}</{tag}>')
    return "\n".join(parts)


def build_messages(case: Case, world: World, system: RoutingSystem) -> list[dict[str, Any]]:
    sections = [
        _render_conversation_history(render_context(case)),
        system.render_roster(world, case),
        _render_current_turn(case.message, case.message_type),
    ]
    return [{"role": "user", "content": "\n\n".join(sections)}]


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _stub_result(name: str, args: dict[str, Any], world: World, system: RoutingSystem) -> dict[str, Any]:
    """Mirror the payloads the real tools return, without side effects."""
    if name == "send_message_to_agent":
        agent_name = str(args.get("agent_name", ""))
        result = {
            "status": "submitted",
            "agent_name": agent_name,
            "new_agent_created": system.resolve_name(world, agent_name) is None,
        }
    elif name == "send_message_to_user":
        result = {"status": "delivered"}
    elif name == "send_draft":
        result = {"status": "draft_recorded", "to": args.get("to"), "subject": args.get("subject")}
    elif name == "wait":
        result = {"status": "waiting", "reason": args.get("reason")}
    else:
        return {"tool": name, "status": "error", "arguments": args, "error": f"Unknown tool: {name}"}
    return {"tool": name, "status": "success", "arguments": args, "result": result}


def _record(decision: Decision, name: str, args: dict[str, Any], world: World, system: RoutingSystem) -> None:
    if name == "send_message_to_agent":
        agent_name = str(args.get("agent_name", "")).strip()
        agent_id = system.resolve_name(world, agent_name)
        if agent_id is None:
            decision.created_names.append(agent_name)
        elif agent_id not in decision.dispatched_ids:
            decision.dispatched_ids.append(agent_id)
    elif name == "send_message_to_user":
        decision.user_messages.append(str(args.get("message", "")))


async def run_routing_case(
    case: Case, world: World, system: RoutingSystem, llm: LLMClient, *, model: str, sample: int
) -> Decision:
    messages = build_messages(case, world, system)
    system_prompt = build_system_prompt()
    tools = get_tool_schemas()
    decision = Decision()

    for iteration in range(MAX_ITERATIONS):
        response = await llm.complete(model=model, system=system_prompt, messages=messages, tools=tools, sample=sample)
        meta = response.get("_meta", {})
        decision.cost += float(meta.get("cost", 0.0))
        decision.latency_s += float(meta.get("latency_s", 0.0))
        decision.prompt_tokens += int(meta.get("prompt_tokens", 0))
        decision.completion_tokens += int(meta.get("completion_tokens", 0))
        if meta.get("cache_key"):
            decision.cache_keys.append(meta["cache_key"])

        message = (response.get("choices") or [{}])[0].get("message") or {}
        raw_calls = message.get("tool_calls") or []
        entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
        if raw_calls:
            entry["tool_calls"] = raw_calls
        messages.append(entry)
        decision.iterations = iteration + 1
        if not raw_calls:
            break

        for raw in raw_calls:
            function = raw.get("function") or {}
            name = function.get("name") or ""
            args = _parse_arguments(function.get("arguments"))
            decision.tool_calls.append({"name": name, "arguments": args})
            _record(decision, name, args, world, system)
            messages.append({
                "role": "tool",
                "tool_call_id": raw.get("id") or name,
                "content": json.dumps(_stub_result(name, args, world, system), default=str),
            })

    decision.clarified = (
        not decision.dispatched_ids
        and not decision.created_names
        and any("?" in text for text in decision.user_messages)
    )
    return decision
```

- [ ] **Step 5: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_layers.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add evals/layers evals/tests/test_layers.py
git commit -m "Add L1 retrieval and L2 routing eval layers"
```

---

### Task 9: Metrics

**Files:**
- Create: `evals/metrics.py`
- Test: `evals/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from evals.metrics import summarize_retrieval, summarize_routing


def row(case_id, sample, outcome, n=10, kind="follow_up", reuse=True, seed=1):
    return {
        "n_agents": n, "world_seed": seed, "case_id": case_id, "kind": kind, "sample": sample,
        "outcome": outcome, "expects_reuse": reuse, "prompt_tokens": 1000, "cost": 0.01, "latency_s": 1.0,
    }


def test_pass_at_1_and_pass_hat_k():
    rows = [
        row("a", 0, "correct"), row("a", 1, "correct"),
        row("b", 0, "correct"), row("b", 1, "duplicate_create"),
    ]
    summary = summarize_routing(rows)["10"]
    assert summary["k"] == 2
    assert summary["pass_at_1"] == pytest.approx(0.75)
    assert summary["pass_hat_k"] == pytest.approx(0.5)
    assert summary["duplicate_rate"] == pytest.approx(0.25)
    assert summary["outcome_rates"] == {"correct": 0.75, "duplicate_create": 0.25}
    assert summary["total_cost_usd"] == pytest.approx(0.04)


def test_groups_by_roster_size_and_kind():
    rows = [row("a", 0, "correct", n=10), row("a", 0, "wrong_reuse", n=1000, kind="typo")]
    summary = summarize_routing(rows)
    assert summary["10"]["pass_at_1"] == 1.0
    assert summary["1000"]["by_kind"] == {"typo": 0.0}


def test_duplicate_rate_is_none_without_reuse_cases():
    assert summarize_routing([row("a", 0, "correct", reuse=False)])["10"]["duplicate_rate"] is None


def test_retrieval_ignores_create_cases_for_recall():
    rows = [
        {"n_agents": 10, "recall": 1.0, "mrr": 0.5, "shown_count": 10, "roster_tokens": 100},
        {"n_agents": 10, "recall": None, "mrr": None, "shown_count": 10, "roster_tokens": 100},
    ]
    summary = summarize_retrieval(rows)["10"]
    assert summary["recall"] == 1.0 and summary["mrr"] == 0.5 and summary["cases"] == 2
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.metrics'`.

- [ ] **Step 3: Write `evals/metrics.py`**

```python
"""Aggregate per-sample rows into experiment metrics, grouped by roster size."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from .grading import CORRECT, DUPLICATE_CREATE


def _by_size(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["n_agents"])].append(row)
    return groups


def summarize_routing(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {size: _routing_group(group) for size, group in _by_size(rows).items()}


def _routing_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[tuple, list[bool]] = defaultdict(list)
    by_sample: dict[int, list[bool]] = defaultdict(list)
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        ok = row["outcome"] == CORRECT
        by_case[(row["world_seed"], row["case_id"])].append(ok)
        by_sample[row["sample"]].append(ok)
        by_kind[row["kind"]].append(ok)
    reuse = [row for row in rows if row["expects_reuse"]]
    outcomes = Counter(row["outcome"] for row in rows)
    return {
        "cases": len(by_case),
        "samples": len(rows),
        "k": max(len(v) for v in by_case.values()),
        "pass_at_1": round(mean(mean(v) for v in by_case.values()), 4),
        "pass_hat_k": round(mean(all(v) for v in by_case.values()), 4),
        "sample_std": round(pstdev([mean(v) for v in by_sample.values()]), 4) if len(by_sample) > 1 else 0.0,
        "duplicate_rate": (
            round(sum(row["outcome"] == DUPLICATE_CREATE for row in reuse) / len(reuse), 4) if reuse else None
        ),
        "outcome_rates": {name: round(count / len(rows), 4) for name, count in sorted(outcomes.items())},
        "by_kind": {kind: round(mean(v), 4) for kind, v in sorted(by_kind.items())},
        "mean_prompt_tokens": round(mean(row["prompt_tokens"] for row in rows), 1),
        "total_cost_usd": round(sum(row["cost"] for row in rows), 4),
        "mean_latency_s": round(mean(row["latency_s"] for row in rows), 3),
    }


def summarize_retrieval(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for size, group in _by_size(rows).items():
        recalls = [row["recall"] for row in group if row["recall"] is not None]
        mrrs = [row["mrr"] for row in group if row["mrr"] is not None]
        summary[size] = {
            "cases": len(group),
            "recall": round(mean(recalls), 4) if recalls else None,
            "mrr": round(mean(mrrs), 4) if mrrs else None,
            "mean_shown": round(mean(row["shown_count"] for row in group), 1),
            "mean_roster_tokens": round(mean(row["roster_tokens"] for row in group), 1),
        }
    return summary
```

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add evals/metrics.py evals/tests/test_metrics.py
git commit -m "Add eval metrics: pass@1, pass^k, duplicate rate, per kind"
```

---

### Task 10: Integrity checksum of protected eval files

An automated experiment loop must never edit cases, gold labels, grading or metrics to raise its score. Every run records a checksum of those files and is marked invalid if it doesn't match the committed one.

**Files:**
- Create: `evals/integrity.py`, `evals/PROTECTED.sha256` (generated)
- Test: `evals/tests/test_integrity.py`

- [ ] **Step 1: Write the failing test**

```python
from evals.integrity import PROTECTED, compute_checksum, expected_checksum, write_expected


def make_tree(root, body=b"x\n"):
    for rel in PROTECTED:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def test_checksum_changes_when_a_protected_file_changes(tmp_path):
    make_tree(tmp_path)
    before = compute_checksum(tmp_path)
    (tmp_path / PROTECTED[0]).write_bytes(b"y\n")
    assert compute_checksum(tmp_path) != before


def test_line_endings_do_not_change_the_checksum(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    make_tree(lf, b"a\nb\n")
    make_tree(crlf, b"a\r\nb\r\n")
    assert compute_checksum(lf) == compute_checksum(crlf)


def test_write_and_read_expected(tmp_path):
    make_tree(tmp_path)
    assert expected_checksum(tmp_path) is None
    write_expected(tmp_path)
    assert expected_checksum(tmp_path) == compute_checksum(tmp_path)


def test_repo_checksum_matches_committed_file():
    from evals.integrity import REPO_ROOT

    assert expected_checksum(REPO_ROOT) == compute_checksum(REPO_ROOT)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_integrity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.integrity'`.

- [ ] **Step 3: Write `evals/integrity.py`**

```python
"""Checksum of eval files an experiment loop must never edit.

Update only by deliberate human commit:  python -m evals.integrity --update
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_FILE = "evals/PROTECTED.sha256"
PROTECTED = (
    "evals/cases.py",
    "evals/grading.py",
    "evals/metrics.py",
    "evals/layers/retrieval.py",
    "evals/layers/routing.py",
    "evals/worlds/model.py",
    "evals/worlds/generator.py",
    "evals/worlds/handwritten.py",
    "evals/worlds/handwritten_cases.yaml",
    "evals/suites/regression.txt",
)


def compute_checksum(root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    for rel in PROTECTED:
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update((root / rel).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def expected_checksum(root: Path = REPO_ROOT) -> str | None:
    path = root / EXPECTED_FILE
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def write_expected(root: Path = REPO_ROOT) -> str:
    value = compute_checksum(root)
    path = root / EXPECTED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the current checksum as expected")
    args = parser.parse_args(argv)
    if args.update:
        print(write_expected())
        return 0
    current, expected = compute_checksum(), expected_checksum()
    print(f"current  {current}\nexpected {expected}")
    return 0 if current == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate the committed checksum**

Run: `python -m evals.integrity --update`
Expected: prints a 64-character hex digest; `evals/PROTECTED.sha256` exists.

- [ ] **Step 5: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_integrity.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add evals/integrity.py evals/PROTECTED.sha256 evals/tests/test_integrity.py
git commit -m "Add integrity checksum for protected eval files"
```

---

### Task 11: Runner, leaderboard and transcript viewer

**Files:**
- Create: `evals/run.py`, `evals/leaderboard.py`, `evals/view.py`, `evals/experiments/smoke.yaml`, `evals/experiments/baseline.yaml`
- Test: `evals/tests/test_run.py`

- [ ] **Step 1: Write the failing test**

```python
import json

import pytest

from evals.llm import fake_llm
from evals.run import Config, load_config, main, run_experiment
from evals.view import render


async def test_smoke_run_writes_results_leaderboard_and_report(tmp_path):
    cfg = Config(name="smoke", n_agents=[10, 20], world_seeds=[1], k=2, cases_per_kind=1, budget_usd=1.0)
    path = await run_experiment(cfg, llm=fake_llm(), out_dir=tmp_path, split="dev", final=False)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert {"10", "20"} <= set(data["summary"]["routing"])
    assert {"10", "20"} <= set(data["summary"]["retrieval"])
    assert data["routing_rows"] and {r["sample"] for r in data["routing_rows"]} == {0, 1}
    assert data["meta"]["checksum"] and data["meta"]["valid"] is True
    assert path.stem in (tmp_path / "leaderboard.md").read_text(encoding="utf-8")

    html = render(data)
    assert "<details>" in html and data["routing_rows"][0]["case_id"] in html


def test_test_split_requires_final(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("name: x\n", encoding="utf-8")
    assert main(["--config", str(config), "--split", "test", "--fake-llm", "--out-dir", str(tmp_path)]) == 2


def test_unknown_config_key_is_rejected(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("name: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        load_config(config)


def test_cli_fake_run(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("name: cli\nn_agents: [10]\nk: 1\ncases_per_kind: 1\ninclude_handwritten: false\n", encoding="utf-8")
    assert main(["--config", str(config), "--fake-llm", "--out-dir", str(tmp_path)]) == 0
    assert list(tmp_path.glob("*-cli-dev.json"))
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest evals/tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.run'`.

- [ ] **Step 3: Write `evals/leaderboard.py`**

```python
"""Regenerate results/leaderboard.md from every result JSON in a directory."""

from __future__ import annotations

import json
from pathlib import Path


def _size_order(size: str) -> tuple[int, int, str]:
    return (0, int(size), "") if size.isdigit() else (1, 0, size)


def write_leaderboard(results_dir: Path) -> Path:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "meta" in data and "summary" in data:
            runs.append((path.stem, data))
    sizes = sorted({size for _, data in runs for size in data["summary"].get("routing", {})}, key=_size_order)
    lines = [
        "# Eval leaderboard",
        "",
        "Each cell: pass@1 / pass^k / duplicate rate. Generated by `evals/leaderboard.py`; do not edit.",
        "",
        "| Run | System | Split | Valid | " + " | ".join(f"n={size}" for size in sizes) + " | Cost |",
        "|" + "---|" * (5 + len(sizes)),
    ]
    for stem, data in runs:
        meta, routing = data["meta"], data["summary"].get("routing", {})
        cells = []
        for size in sizes:
            group = routing.get(size)
            if group is None:
                cells.append("–")
                continue
            dup = "–" if group["duplicate_rate"] is None else f"{group['duplicate_rate']:.2f}"
            cells.append(f"{group['pass_at_1']:.2f} / {group['pass_hat_k']:.2f} / {dup}")
        cost = sum(group["total_cost_usd"] for group in routing.values())
        valid = "yes" if meta["valid"] else "**no**"
        lines.append(f"| {stem} | {meta['system']} | {meta['split']} | {valid} | " + " | ".join(cells) + f" | ${cost:.2f} |")
    out = results_dir / "leaderboard.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
```

- [ ] **Step 4: Write `evals/view.py`**

```python
"""Render one result JSON as a static HTML report.

    python -m evals.view evals/results/<run>.json   ->  evals/results/<run>.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from .grading import CORRECT

_STYLE = """
body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b2430;background:#fff}
table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccd5dd;padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}details{border:1px solid #ccd5dd;border-radius:6px;margin:6px 0;padding:6px 10px}
summary{cursor:pointer}pre{white-space:pre-wrap;background:#f2f5f7;padding:8px;border-radius:4px}.bad{color:#b4400c}
"""


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def render(data: dict[str, Any]) -> str:
    meta, summary = data["meta"], data["summary"]
    routing, retrieval = summary.get("routing", {}), summary.get("retrieval", {})
    parts = [f"<!doctype html><meta charset='utf-8'><title>{escape(meta['name'])}</title><style>{_STYLE}</style>"]
    parts.append(f"<h1>{escape(meta['name'])} · {escape(meta['system'])} · {escape(meta['split'])}</h1>")
    validity = "valid" if meta["valid"] else "<span class='bad'>INVALID: protected eval files changed</span>"
    parts.append(f"<p>model {escape(meta['model'])} · k={meta['k']} · git {escape(str(meta['git_sha']))} · {validity} · spent ${meta['budget_spent_usd']:.2f}</p>")

    parts.append("<h2>Routing (L2)</h2>")
    parts.append(_table(
        ["n agents", "cases", "pass@1", "pass^k", "sample std", "duplicate rate", "prompt tokens", "cost $"],
        [[n, g["cases"], g["pass_at_1"], g["pass_hat_k"], g["sample_std"], g["duplicate_rate"], g["mean_prompt_tokens"], g["total_cost_usd"]]
         for n, g in routing.items()],
    ))
    kinds = sorted({kind for g in routing.values() for kind in g["by_kind"]})
    parts.append("<h3>pass@1 by kind</h3>")
    parts.append(_table(["kind", *routing.keys()], [[kind, *[g["by_kind"].get(kind, "–") for g in routing.values()]] for kind in kinds]))
    outcomes = sorted({o for g in routing.values() for o in g["outcome_rates"]})
    parts.append("<h3>Outcome rates</h3>")
    parts.append(_table(["outcome", *routing.keys()], [[o, *[g["outcome_rates"].get(o, 0) for g in routing.values()]] for o in outcomes]))

    parts.append("<h2>Retrieval (L1)</h2>")
    parts.append(_table(
        ["n agents", "cases", "recall", "MRR", "shown", "roster tokens"],
        [[n, g["cases"], g["recall"], g["mrr"], g["mean_shown"], g["mean_roster_tokens"]] for n, g in retrieval.items()],
    ))

    parts.append("<h2>Failing samples</h2>")
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data.get("routing_rows", []):
        if row["outcome"] != CORRECT:
            failures[str(row["n_agents"])].append(row)
    for n, rows in failures.items():
        parts.append(f"<h3>n = {escape(n)} ({len(rows)} failing)</h3>")
        for row in rows:
            detail = {
                "gold": row["gold"], "acceptable": row["acceptable"], "dispatched_ids": row["dispatched_ids"],
                "created_names": row["created_names"], "user_messages": row["user_messages"],
                "tool_calls": row["tool_calls"], "cache_keys": row["cache_keys"],
            }
            parts.append(
                f"<details><summary><b class='bad'>{escape(row['outcome'])}</b> · {escape(row['kind'])} · "
                f"{escape(row['case_id'])} · sample {row['sample']} — {escape(row['message'])}</summary>"
                f"<pre>{escape(json.dumps(detail, indent=2))}</pre></details>"
            )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.result.read_text(encoding="utf-8"))
    out = args.result.with_suffix(".html")
    out.write_text(render(data), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Write `evals/run.py`**

```python
"""Run an eval experiment.

    python -m evals.run --config evals/experiments/baseline.yaml            # dev split
    python -m evals.run --config evals/experiments/smoke.yaml --fake-llm    # offline
    python -m evals.run --config ... --split test --final                   # held-out, once per finalist
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .cases import Case, filter_cases
from .grading import ERROR, Decision, grade
from .integrity import REPO_ROOT, compute_checksum, expected_checksum
from .layers.retrieval import run_retrieval_case
from .layers.routing import run_routing_case
from .leaderboard import write_leaderboard
from .llm import Budget, BudgetExceeded, LLMClient, OpenRouterLLM, fake_llm
from .metrics import summarize_retrieval, summarize_routing
from .systems import get_system
from .worlds.generator import build_world, generate_cases
from .worlds.handwritten import load_handwritten
from .worlds.model import World

DEFAULT_OUT = REPO_ROOT / "evals" / "results"
CACHE_DIR = REPO_ROOT / "evals" / ".cache" / "llm"


@dataclass
class Config:
    name: str
    system: str = "baseline"
    model: str = "anthropic/claude-sonnet-4"
    layers: list[str] = field(default_factory=lambda: ["retrieval", "routing"])
    n_agents: list[int] = field(default_factory=lambda: [10, 50, 200, 1000])
    world_seeds: list[int] = field(default_factory=lambda: [1])
    k: int = 3
    cases_per_kind: int = 3
    kinds: list[str] | None = None
    suite: str = "all"
    budget_usd: float = 15.0
    concurrency: int = 8
    include_handwritten: bool = True


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)


Unit = tuple[Any, World, list[Case]]  # (roster-size label, world, cases)


def _build_units(cfg: Config, split: str) -> list[Unit]:
    units: list[Unit] = []
    for seed in cfg.world_seeds:
        full = build_world(seed)
        cases = filter_cases(generate_cases(full, cfg.cases_per_kind), split=split, suite=cfg.suite, kinds=cfg.kinds)
        for n in cfg.n_agents:
            units.append((n, full.truncated(n), cases))
    if cfg.include_handwritten:
        world, cases = load_handwritten()
        units.append(("handwritten", world, filter_cases(cases, split=split, suite=cfg.suite, kinds=cfg.kinds)))
    return units


def _routing_row(label: Any, world: World, case: Case, sample: int, d: Decision, outcome: str, system: str) -> dict[str, Any]:
    return {
        "system": system, "n_agents": label, "world_seed": world.seed, "case_id": case.id, "kind": case.kind,
        "sample": sample, "message": case.message, "gold": case.gold_targets, "acceptable": case.acceptable_ids,
        "expects_reuse": case.expects_reuse, "outcome": outcome, "dispatched_ids": d.dispatched_ids,
        "created_names": d.created_names, "clarified": d.clarified, "user_messages": d.user_messages,
        "tool_calls": d.tool_calls, "iterations": d.iterations, "cost": d.cost, "latency_s": d.latency_s,
        "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens, "cache_keys": d.cache_keys,
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def run_experiment(cfg: Config, *, llm: LLMClient, out_dir: Path, split: str, final: bool) -> Path:
    started = datetime.now(timezone.utc)
    checksum, expected = compute_checksum(), expected_checksum()
    system = get_system(cfg.system)
    units = _build_units(cfg, split)

    retrieval_rows: list[dict[str, Any]] = []
    if "retrieval" in cfg.layers:
        for label, world, cases in units:
            for case in cases:
                retrieval_rows.append({
                    "n_agents": label, "world_seed": world.seed, "case_id": case.id, "kind": case.kind,
                    **run_retrieval_case(case, world, system),
                })

    routing_rows: list[dict[str, Any]] = []
    budget_exhausted = False
    if "routing" in cfg.layers:

        async def one(label: Any, world: World, case: Case, sample: int) -> dict[str, Any] | None:
            nonlocal budget_exhausted
            if budget_exhausted:
                return None
            try:
                decision = await run_routing_case(case, world, system, llm, model=cfg.model, sample=sample)
                outcome = grade(case, decision)
            except BudgetExceeded:
                budget_exhausted = True
                return None
            except Exception as exc:  # recorded as a failing sample, never silently dropped
                decision, outcome = Decision(user_messages=[f"ERROR: {type(exc).__name__}: {exc}"]), ERROR
            return _routing_row(label, world, case, sample, decision, outcome, system.name)

        jobs = [one(label, world, case, s) for label, world, cases in units for case in cases for s in range(cfg.k)]
        routing_rows = [row for row in await asyncio.gather(*jobs) if row is not None]

    budget = getattr(llm, "budget", None)
    result = {
        "meta": {
            **asdict(cfg),
            "split": split,
            "final": final,
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "expected_checksum": expected,
            "valid": expected is not None and checksum == expected,
            "budget_spent_usd": round(budget.spent_usd, 4) if budget else 0.0,
            "budget_exhausted": budget_exhausted,
        },
        "summary": {"retrieval": summarize_retrieval(retrieval_rows), "routing": summarize_routing(routing_rows)},
        "retrieval_rows": retrieval_rows,
        "routing_rows": routing_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{started:%Y%m%d-%H%M%S}-{cfg.name}-{split}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_leaderboard(out_dir)
    return path


def _api_key() -> str | None:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    from server.config import get_settings  # loads the repo-root .env

    return get_settings().openrouter_api_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required to run the held-out test split")
    parser.add_argument("--fake-llm", action="store_true", help="offline scripted model; no API calls")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.split == "test" and not args.final:
        print("Refusing to run the test split without --final. Tune on dev; run test once per finalist.", file=sys.stderr)
        return 2
    cfg = load_config(args.config)

    llm: LLMClient
    if args.fake_llm:
        llm = fake_llm()
    else:
        key = _api_key()
        if not key:
            print("OPENROUTER_API_KEY is not set. Add it to .env at the repo root.", file=sys.stderr)
            return 2
        llm = OpenRouterLLM(api_key=key, cache_dir=CACHE_DIR, budget=Budget(cfg.budget_usd), concurrency=cfg.concurrency)

    async def go() -> Path:
        try:
            return await run_experiment(cfg, llm=llm, out_dir=args.out_dir, split=args.split, final=args.final)
        finally:
            await llm.aclose()

    path = asyncio.run(go())
    data = json.loads(path.read_text(encoding="utf-8"))
    for size, group in data["summary"]["routing"].items():
        print(f"n={size:>11}  pass@1={group['pass_at_1']:.2f}  pass^k={group['pass_hat_k']:.2f}  dup={group['duplicate_rate']}")
    print(f"spent ${data['meta']['budget_spent_usd']:.2f}  valid={data['meta']['valid']}  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Write the experiment configs**

`evals/experiments/smoke.yaml`: a cheap live check, about $0.30.

```yaml
name: smoke
system: baseline
model: anthropic/claude-sonnet-4
n_agents: [10, 200]
world_seeds: [1]
k: 1
cases_per_kind: 1
include_handwritten: false
budget_usd: 1.0
```

`evals/experiments/baseline.yaml`: the baseline. Roughly 270 samples, about $12 expected, capped at $20.

```yaml
name: baseline
system: baseline
model: anthropic/claude-sonnet-4
n_agents: [10, 50, 200, 1000]
world_seeds: [1]
k: 3
cases_per_kind: 3
include_handwritten: true
budget_usd: 20.0
concurrency: 8
```

- [ ] **Step 7: Run the tests to make sure they pass**

Run: `python -m pytest evals/tests/test_run.py -v`
Expected: 4 passed.

- [ ] **Step 8: Run the full suite and an offline CLI smoke run**

Run: `python -m pytest -q`
Expected: all tests pass (2 server + every evals test).

Run: `python -m evals.run --config evals/experiments/smoke.yaml --fake-llm --out-dir evals/.cache/fake-results`
Expected: one `n=…` line per roster size, `valid=True`, and a JSON path under `evals/.cache/fake-results/` (ignored by git, so fake runs never reach the leaderboard).

- [ ] **Step 9: Commit**

```bash
git add evals/run.py evals/leaderboard.py evals/view.py evals/experiments/smoke.yaml evals/experiments/baseline.yaml evals/tests/test_run.py
git commit -m "Add eval runner, leaderboard and HTML transcript report"
```

---

### Task 12: Experiment journal and harness README

**Files:**
- Create: `evals/experiments/JOURNAL.md`, `evals/README.md`

- [ ] **Step 1: Write `evals/experiments/JOURNAL.md`**

```markdown
# Experiment journal

One entry per experiment, written **before** the next experiment starts. Tune on `dev` only.
The `test` split runs once per finalist (`--split test --final`).

Template:

## <date> · <experiment name>
- **Hypothesis:** what should change, and why (cite the error-analysis category).
- **Change:** config / prompt / code diff, one variable.
- **Run:** result file name.
- **Dev result:** pass@1 / pass^k / duplicate rate per roster size, vs the previous best.
- **Regression suite:** pass / fail.
- **Decision:** keep or revert, and why.

---
```

- [ ] **Step 2: Write `evals/README.md`**

````markdown
# OpenPoke eval harness

Measures how well the Interaction Agent routes work to execution agents as the roster grows.

## Layers
- **L1 retrieval** (no LLM): is the right agent among what the system shows? recall, MRR, roster tokens.
- **L2 routing** (live): one real Interaction Agent turn with the server's own prompt; tools are
  intercepted and the decision is graded by outcome (reuse / create / clarify).
- L3 end-to-end scenarios arrive in Plan 4.

## Run
```bash
python -m pytest -q                                                    # unit tests, offline
python -m evals.run --config evals/experiments/smoke.yaml --fake-llm   # offline smoke
python -m evals.run --config evals/experiments/baseline.yaml           # live, needs OPENROUTER_API_KEY in .env
python -m evals.view evals/results/<run>.json                          # HTML report with failing transcripts
```

## Rules
- Tune on `dev`. `--split test` requires `--final` and runs once per finalist.
- Protected files (cases, generator, grading, metrics, layers, regression suite) are checksummed;
  a run with a mismatch is marked **invalid**. Update only by deliberate commit:
  `python -m evals.integrity --update`.
- Responses are cached in `evals/.cache/llm/`, keyed by request + sample index. Reruns are free.
- Every experiment gets a `experiments/JOURNAL.md` entry.
````

- [ ] **Step 3: Commit**

```bash
git add evals/experiments/JOURNAL.md evals/README.md
git commit -m "Add experiment journal and eval harness README"
```

---

### Task 13: OpenRouter key, live smoke run, baseline run

**This is the step where Samin sets up the OpenRouter key.**

**Files:**
- Create (not committed): `.env` at the repo root
- Create (committed): `evals/results/<timestamp>-baseline-dev.json`, `evals/results/leaderboard.md`

- [ ] **Step 1: Samin adds the key**

Create `.env` in the repo root (it's already in `.gitignore`) containing one line:

```text
OPENROUTER_API_KEY=sk-or-...
```

Check it's ignored: `git check-ignore .env` should print `.env`.

- [ ] **Step 2: Confirm the model id is served**

Run:

```bash
python -c "import httpx; ids={m['id'] for m in httpx.get('https://openrouter.ai/api/v1/models', timeout=30).json()['data']}; print('anthropic/claude-sonnet-4' in ids)"
```

Expected: `True`. If `False`, choose the closest Sonnet id from that list, set `model:` in both experiment configs and `FALLBACK_PRICING` in `evals/llm.py`, and note it in the journal.

- [ ] **Step 3: Live smoke run (about $0.30)**

Run: `python -m evals.run --config evals/experiments/smoke.yaml --out-dir evals/.cache/live-smoke`
Expected: `n=10` and `n=200` lines, spend under $1, `valid=True`. The output directory is git-ignored, so the smoke run stays off the leaderboard. Open the report: `python -m evals.view evals/.cache/live-smoke/<smoke file>.json`. Check that failing samples show real tool calls, not `ERROR:` messages. If errors appear, fix the cause before continuing.

- [ ] **Step 4: Baseline run (about $12, capped at $20)**

Run: `python -m evals.run --config evals/experiments/baseline.yaml`
Expected: four `n=` lines plus `n=handwritten`, `valid=True`, `budget_exhausted` false in the JSON. If the budget runs out, the result is partial: say so in the journal rather than rerunning blindly. Thanks to the cache, a rerun with a higher cap only pays for the missing samples.

- [ ] **Step 5: Render the report**

Run: `python -m evals.view evals/results/<baseline file>.json`

- [ ] **Step 6: Journal entry**

Append to `evals/experiments/JOURNAL.md`:

```markdown
## 2026-10-28 · baseline
- **Hypothesis:** routing accuracy falls and duplicate creation rises as the roster grows from 10 to 1,000 bare names (F1, F2).
- **Change:** none. Production behaviour (`system: baseline`).
- **Run:** <baseline file name>
- **Dev result:** <paste the four n= lines and n=handwritten>
- **Regression suite:** n/a (empty)
- **Decision:** reference point for Plan 2.
```

- [ ] **Step 7: Commit**

```bash
git add evals/results/*.json evals/results/leaderboard.md evals/experiments/JOURNAL.md
git commit -m "Record baseline eval results"
```

---

### Task 14: Error analysis and first regression suite

Read failures by hand before choosing metrics or fixes.

**Files:**
- Create: `evals/results/error_analysis.md`
- Modify: `evals/suites/regression.txt`, `evals/PROTECTED.sha256`

- [ ] **Step 1: Read about 60 failing samples**

Open `evals/results/<baseline file>.html`. Take failures from every roster size, weighted toward n=200 and n=1000. For each one, read the gold target, the tool calls and the user messages, and give it **one** category in your own words. Typical starting categories: *picked a same-person different-topic agent*, *created a near-duplicate name*, *reused the most recent agent by default*, *ignored context for vague references*, *guessed between two people instead of asking*, *dispatched on an agent-result turn*.

- [ ] **Step 2: Write `evals/results/error_analysis.md`**

```markdown
# Baseline error analysis

- **Run:** <baseline file name>
- **Sampled:** <how many> failing samples (<per-size counts>), read in full.

| Category | Definition | Count | Example case ids |
|---|---|---|---|
| <category> | <one sentence> | <n> | <ids> |

## Biggest categories → Plan 2 targets
1. <category>: <why it happens, from the transcripts>; the fix it suggests.
2. …
3. …

## Surprises
- <anything the spec did not anticipate>
```

- [ ] **Step 3: Seed the regression suite with cases the baseline gets right every time**

Run:

```bash
python - <<'EOF'
import json, pathlib
from collections import defaultdict
path = sorted(pathlib.Path("evals/results").glob("*-baseline-dev.json"))[-1]
rows = json.loads(path.read_text(encoding="utf-8"))["routing_rows"]
ok = defaultdict(list)
for r in rows:
    ok[r["case_id"]].append(r["outcome"] == "correct")
passing = sorted(case for case, results in ok.items() if all(results))
reg = pathlib.Path("evals/suites/regression.txt")
header = [l for l in reg.read_text(encoding="utf-8").splitlines() if l.startswith("#")]
reg.write_text("\n".join(header + passing) + "\n", encoding="utf-8")
print(len(passing), "regression cases")
EOF
```

Expected: a count of case ids that passed every sample at every roster size.

- [ ] **Step 4: Update the protected checksum (deliberate change)**

Run: `python -m evals.integrity --update` then `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add evals/results/error_analysis.md evals/suites/regression.txt evals/PROTECTED.sha256
git commit -m "Add baseline error analysis and seed regression suite"
```

---

## Done when

- `python -m pytest -q` passes on Windows (and Linux).
- `evals/results/leaderboard.md` has a valid `baseline` row with numbers for n = 10, 50, 200, 1000 and handwritten.
- `evals/results/error_analysis.md` names the largest failure categories, which become Plan 2's targets.
