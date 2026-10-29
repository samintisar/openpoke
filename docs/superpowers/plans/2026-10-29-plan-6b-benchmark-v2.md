# Plan 6b: Benchmark v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A routing benchmark that can be trusted. It has:
- two lives (dev on life 1, sealed test on life 2);
- the same mix of message kinds in both;
- two new kinds, `status_question` and `paraphrase`;
- answer keys that come from construction and are checked by three other model families;
- severity-graded scoring.

Every system is then re-measured on v2 dev.

**Architecture:**
- **What already exists.** The lived world already gives construction-based answer keys. Every plan **thread** (person + topic) records the agents that served it during growth (`Snapshot.thread_agents`), and probes take their gold from that.
- **What v2 adds:**
  - a second persona (`build_plan(2)`);
  - two probe families;
  - a builder (`bench_v2.py`) that produces balanced, ambiguity-aware probe sets for both lives;
  - three checker models that flag doubtful keys for human review;
  - a finalize step;
  - a loader that serves life 1 as dev and life 2 as test;
  - severity rates in the metrics.

**Tech Stack:** Python 3.10+, pytest (asyncio auto mode), the existing eval harness, OpenRouter (DeepSeek V4.1 Flash for growth and writing; GPT-6 Luna, Gemini 3.1 Flash Lite and Qwen3.8 Flash as checkers).

**Spec:** `docs/superpowers/specs/2026-10-29-agent-overload-v2-design.md` §3 and §7 steps 3–4.

**Ground rules for every task:**
- **Tests:** `.venv/Scripts/python -m pytest -q` must be green before each commit. The only exception is the integrity checks while a protected file is uncommitted, and that exception covers only `test_integrity.py`'s two repo checks and `test_run.py`'s two validity checks.
- **Checksum:** protected files need `.venv/Scripts/python -m evals.integrity --update` plus a commit before any live run.
- **Life 1 must not change.** `build_plan(1)` must stay byte-identical: the frozen world `world-s1.json` was grown from it. Task 1 adds a golden test for this before anything moves.
- **Commits** end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Do not touch** `.serena/`, `pyproject.toml`, `uv.lock` or `.env`.

---

## File map

| File | Responsibility |
|---|---|
| `evals/worlds/lived/personas.py` (new) | Persona dataclass and the two personas' name tables and text |
| `evals/worlds/lived/plan.py` | `build_plan(seed)` uses the seed's persona; topic text personalized with the persona's child; `service_role()` |
| `evals/worlds/lived/probes.py` | `status_question` and `paraphrase` families; service role lookup via `service_role()` |
| `evals/cases.py` | the two new kinds; `Case.split_override` |
| `evals/worlds/lived/bench_v2.py` (new) | build, balance, check and finalize v2 probe sets; load v2 units |
| `evals/worlds/lived/checkers.py` (new) | three-model key checks |
| `evals/labelling/probes_page.html` | shows checker answers on each probe |
| `evals/grading.py`, `evals/metrics.py` | severity per outcome; harmful and costly rates |
| `evals/run.py` | `world: lived_v2` with `life: 1 \| 2` |
| `evals/integrity.py` | protects the reviewed v2 files and `bench_v2.py` |
| `evals/experiments/v2-*.yaml` (new) | dev re-baseline configs |
| tests | `test_lived_plan.py`, `test_lived_probes.py`, `test_bench_v2.py` (new), `test_checkers.py` (new), `test_grading.py`, `test_metrics.py`, `test_run.py` |

---

### Task 1: Golden test that life 1's plan never changes

**Files:** Test: `evals/tests/test_lived_plan.py`

- [ ] **Step 1: Add the test**

```python
import hashlib
import json
from dataclasses import asdict

from evals.worlds.lived.plan import build_plan


def _plan_digest(seed: int) -> str:
    plan = build_plan(seed)
    blob = json.dumps(asdict(plan), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


LIFE1_PLAN_SHA256 = "<filled in Step 2>"


def test_life1_plan_is_frozen():
    assert _plan_digest(1) == LIFE1_PLAN_SHA256
```

- [ ] **Step 2: Record the digest from the current code**

```bash
.venv/Scripts/python -c "from evals.tests.test_lived_plan import _plan_digest; print(_plan_digest(1))"
```

Paste the printed hash into `LIFE1_PLAN_SHA256`.

- [ ] **Step 3:** Run `.venv/Scripts/python -m pytest -q evals/tests/test_lived_plan.py`. Expected: pass.
- [ ] **Step 4: Commit** with the message "Golden digest of life 1's plan, before personas move out of plan.py".

---

### Task 2: Personas, with life 1 unchanged

**Files:**
- Create: `evals/worlds/lived/personas.py`
- Modify: `evals/worlds/lived/plan.py`:
  - move `PERSONA`, `WORK_COMPANY`, `COLLIDING_FIRSTS`, `SHARED_SURNAME`, `WORK`, `CLIENTS`, `RECRUITERS`, `FAMILY`, `FRIENDS` and `SERVICES` out;
  - change `_contacts`, `_threads`, `build_plan` and the `TOPIC_KEYWORDS` build.
- Modify: `evals/worlds/lived/probes.py:57` (service role lookup)
- Test: `evals/tests/test_lived_plan.py`

- [ ] **Step 1: Write the failing tests** (append to `test_lived_plan.py`)

```python
from evals.worlds.lived.personas import PERSONAS, persona_for
from evals.worlds.lived.plan import service_role, topic_keyword


def test_seed_picks_persona():
    assert persona_for(1).name == "jordan" and persona_for(2).name == "riley"
    assert persona_for(3).name == "riley"  # any seed other than 1 uses the second life


def test_life2_shares_no_people_with_life1():
    one = {c.full for c in build_plan(1).contacts}
    two = build_plan(2)
    assert not one & {c.full for c in two.contacts}
    assert "Riley" in two.persona and "Mia" not in two.persona


def test_life2_has_namesakes_and_a_shared_surname():
    two = build_plan(2)
    firsts = [c.first for c in two.contacts]
    for first in persona_for(2).colliding_firsts:
        assert firsts.count(first) >= 2
    assert sum(c.last == persona_for(2).shared_surname for c in two.contacts) >= 2


def test_child_name_follows_the_persona():
    topics2 = {t.topic for t in build_plan(2).threads}
    assert not any("Mia" in topic for topic in topics2)
    for topic in topics2:
        assert topic_keyword(topic)  # every personalized topic still has its keyword


def test_service_roles_for_both_lives():
    assert service_role(build_plan(1).contact("Gary Whitman")) == "landlord"
    two = build_plan(2)
    landlord = next(c for c in two.contacts if c.relation == "service" and service_role(c) == "landlord")
    assert landlord.full == "Dennis Varga"
```

- [ ] **Step 2:** Run them. Expected: ImportError (`personas`).

- [ ] **Step 3: Create `evals/worlds/lived/personas.py`**

Move Jordan's tables **verbatim** from `plan.py`. Copy the existing tuples exactly as they are; their order matters for the seed-1 digest.

```python
"""The people of each simulated life. Life 1 (Jordan) is frozen: its world was grown from these tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    text: str  # the persona paragraph the writer and the IA see
    child: str  # replaces "Mia" in topic text
    work_company: str
    colliding_firsts: tuple[str, ...]  # first names shared by two contacts (same_name_people probes)
    shared_surname: str
    work: tuple[tuple[str, str], ...]
    clients: tuple[tuple[str, str, str], ...]
    recruiters: tuple[tuple[str, str, str], ...]
    family: tuple[tuple[str, str], ...]
    friends: tuple[tuple[str, str], ...]
    services: tuple[tuple[str, str, str, str], ...]  # (first, last, company, role)


JORDAN = Persona(
    name="jordan",
    text=(...),  # the exact PERSONA string from plan.py
    child="Mia",
    work_company="Northwind Labs",
    colliding_firsts=("Alice", "Ben"),
    shared_surname="Doyle",
    work=(...),  # the exact WORK tuple
    clients=(...),  # the exact CLIENTS tuple
    recruiters=(...),  # the exact RECRUITERS tuple
    family=(...),  # the exact FAMILY tuple
    friends=(...),  # the exact FRIENDS tuple
    services=(...),  # the exact SERVICES tuple
)
```

The `(...)` placeholders above mean "paste the current tuple from plan.py unchanged". They are not new content.

Then add Riley:

```python
RILEY = Persona(
    name="riley",
    text=(
        "Riley Nakamura, 41, operations director at Bluefin Logistics in Denver, Colorado. Partner Sam, "
        "two kids, Theo (9) and Lena (5). Sister Joanna and brother Kai; cousins on the Holm side (Mom's "
        "family) and the Sullivan side. Juggles a team, clients, recruiters, friends and family logistics "
        "through an AI assistant, mostly from their phone."
    ),
    child="Lena",
    work_company="Bluefin Logistics",
    colliding_firsts=("Nina", "Max"),
    shared_surname="Holm",  # colleague Ingrid Holm and cousins Erik and Ada Holm
    work=(
        ("Nina", "Castillo"), ("Victor", "Hayes"), ("Aisha", "Bello"), ("Bruno", "Silva"), ("Yuki", "Tanaka"),
        ("Farah", "Qureshi"), ("Lars", "Nilsson"), ("Mei", "Wong"), ("Adrian", "Popescu"), ("Ingrid", "Holm"),
        ("Samir", "Haddou"), ("Clara", "Dubois"), ("Rohan", "Mehta"), ("Elise", "Moreau"), ("Tomasz", "Nowak"),
        ("Imani", "Okeke"), ("Hector", "Ruiz"), ("Signe", "Dahl"),
    ),
    clients=(
        ("Max", "Weller", "Globex"), ("Odette", "Fournier", "Initech"), ("Raj", "Pillai", "Hooli"),
        ("Bianca", "Ferreira", "Umbrella"), ("Caleb", "Young", "Stark"), ("Delia", "Murray", "Wayne"),
        ("Emeka", "Obi", "Soylent"), ("Fiona", "Grant", "Acme"), ("Gustavo", "Lima", "Globex"),
        ("Heidi", "Kraus", "Initech"), ("Ivan", "Petrenko", "Hooli"), ("Jade", "Thornton", "Umbrella"),
        ("Kofi", "Asante", "Stark"), ("Lorna", "Price", "Wayne"),
    ),
    recruiters=(
        ("Mona", "Ellison", "Hirewell"), ("Nate", "Rivers", "Talentbridge"), ("Opal", "Grey", "Hirewell"),
        ("Pierce", "Lang", "Pinnacle Search"), ("Rosa", "Vidal", "Talentbridge"), ("Stefan", "Kroll", "Pinnacle Search"),
    ),
    family=(
        ("Sam", "Nakamura"), ("Joanna", "Nakamura"), ("Kai", "Nakamura"), ("Erik", "Holm"), ("Martha", "Nakamura"),
        ("Hiro", "Nakamura"), ("Ada", "Holm"), ("Bea", "Sullivan"), ("Frank", "Sullivan"), ("Tessa", "Nakamura"),
        ("Ollie", "Sullivan"), ("June", "Park"), ("Walter", "Nakamura"), ("Ruth", "Sullivan"),
    ),
    friends=(
        ("Nina", "Park"), ("Max", "Okonkwo"), ("Zara", "Ahmed"), ("Brendan", "Kelly"), ("Lucy", "Hammond"),
        ("Tariq", "Aziz"), ("Vera", "Kowal"), ("Will", "Danvers"), ("Ximena", "Rojas"), ("Yara", "Saleh"),
        ("Abby", "Wren"), ("Cole", "Mercer"), ("Dina", "Farouk"), ("Eli", "Strand"), ("Flora", "Bayer"),
        ("Gus", "Moran"), ("Holly", "Pratt"), ("Ian", "Doherty"),
    ),
    services=(
        ("Dennis", "Varga", "Varga Rentals", "landlord"), ("Paula", "Ng", "Bright Day Dental", "dentist"),
        ("Arthur", "Bell", "Bell Tax", "accountant"), ("Marisol", "Vega", "Fresh Nest", "cleaner"),
        ("Bob", "Kaminski", "Mile High Auto", "mechanic"), ("Nadia", "Brandt", "Cherry Creek Vets", "vet"),
        ("Ray", "Duffy", "Duffy Plumbing", "plumber"), ("Edith", "Lowe", "Lowe Music", "piano teacher"),
        ("Trent", "Hollis", "Hollis Build", "contractor"), ("Wanda", "Mills", "Summit Insurance", "insurance agent"),
    ),
)

PERSONAS = (JORDAN, RILEY)


def persona_for(seed: int) -> Persona:
    """Seed 1 is life 1 (Jordan, frozen); every other seed is life 2 (Riley)."""
    return JORDAN if seed == 1 else RILEY
```

- [ ] **Step 4: Rewire `plan.py`**

1. Delete the moved constants. Import with `from .personas import PERSONAS, Persona, persona_for`.
2. Add a helper and rebuild `TOPIC_KEYWORDS` over every persona's child name, so personalized topics resolve:

```python
def personalize(text: str, persona: Persona) -> str:
    """Topic and detail text with the persona's child in place of life 1's ("Mia")."""
    return text.replace("Mia", persona.child)


TOPIC_KEYWORDS: dict[str, str] = {
    personalize(topic, persona): keyword
    for persona in PERSONAS
    for pool in (*TOPICS.values(), *SERVICE_TOPICS.values(), ERRANDS)
    for topic, keyword in pool
}
```

3. Change `_contacts()` to `_contacts(persona: Persona)`. It loops over the `persona.*` tables instead of the module constants, and personalizes each topic pool:

```python
    def pool(topics):
        return tuple((personalize(topic, persona), keyword) for topic, keyword in topics)
```

   Wrap every `TOPICS[...]` and `SERVICE_TOPICS[role]` passed to `add(...)` in `pool(...)`.
4. Change `_threads(contacts, pools, rng)` to take a `persona` argument. Use `persona.colliding_firsts` in place of `COLLIDING_FIRSTS`. Errand topics become `personalize(topic, persona)`.
5. In `build_plan(seed)`:
   - start with `persona = persona_for(seed)`;
   - call `_contacts(persona)` and `_threads(contacts, pools, rng, persona)`;
   - pass `persona=persona.text` to `LifePlan`;
   - personalize the background phrases and the `_Mentions` detail strings only if they contain "Mia". Check with `grep -n "Mia" evals/worlds/lived/plan.py` after the move; every remaining hit must go through `personalize`.
6. Add:

```python
def service_role(contact: Contact) -> str | None:
    """A service contact's role (landlord, dentist, ...) from whichever persona lists them."""
    for persona in PERSONAS:
        for first, last, _, role in persona.services:
            if (first, last) == (contact.first, contact.last):
                return role
    return None
```

7. In `probes.py`:
   - replace the `SERVICES` lookup at line 57 with `service_role(contact)`;
   - drop `SERVICES` from its imports;
   - import `service_role` from `.plan`.

The order of `rng` calls in `build_plan` must not change for seed 1. Keep every call where it was.

- [ ] **Step 5:** Run `.venv/Scripts/python -m pytest -q evals/tests/test_lived_plan.py evals/tests/test_lived_probes.py evals/tests/test_lived_round2.py evals/tests/test_lived_grow.py`. Expected: all pass, **including `test_life1_plan_is_frozen`**. If the digest changed, the move altered life 1. Fix it; never update the digest.
- [ ] **Step 6:** Full suite (the integrity exception applies; `plan.py` and `probes.py` may be protected).
- [ ] **Step 7: Commit** with the message "Personas: life 1 (Jordan, frozen) and life 2 (Riley); plan built from the seed's persona".

---

### Task 3: Two new kinds and a split override on cases

**Files:**
- Modify: `evals/cases.py`
- Modify: `evals/worlds/lived/round2.py` (`TestCase` becomes a thin alias)
- Test: `evals/tests/test_cases.py`

- [ ] **Step 1: Failing tests** (append to `test_cases.py`)

```python
from evals.cases import ROUTING_KINDS, Case


def test_new_routing_kinds():
    assert "status_question" in ROUTING_KINDS and "paraphrase" in ROUTING_KINDS


def test_split_override_wins_over_the_hash():
    base = dict(id="x", kind="follow_up", world_seed=2, message="m", gold_targets=["a"])
    assert Case(**base, split_override="test").split == "test"
    assert Case(**base, split_override="dev").split == "dev"
    assert Case(**base).split in ("dev", "test")


def test_split_override_is_validated():
    import pytest
    with pytest.raises(ValueError):
        Case(id="x", kind="follow_up", world_seed=2, message="m", gold_targets=[], split_override="train")
```

- [ ] **Step 2:** Run. Expected: fail.

- [ ] **Step 3: Implement in `evals/cases.py`**
  - Append `"status_question", "paraphrase"` to `ROUTING_KINDS`.
  - Add a field after `dispatch_optional`: `split_override: str | None = None  # "dev"/"test": a whole life is one split (benchmark v2)`.
  - In `__post_init__`, add: `if self.split_override not in (None, "dev", "test"): raise ValueError(f"unknown split_override {self.split_override!r}")`.
  - In `split`, return `self.split_override` when it's set, before the hash.

- [ ] **Step 4:** In `round2.py`, replace the `TestCase` body. It still forces test, and the existing round-2 tests keep passing:

```python
class TestCase(Case):
    """A round-2 case: always in the test split, whatever its id hashes to."""

    __test__ = False  # not a pytest class

    def __post_init__(self) -> None:
        self.split_override = "test"
        super().__post_init__()
```

- [ ] **Step 5:** Some tests and code may enumerate `ROUTING_KINDS` and assume every kind has a probe builder. Check with `grep -rn "ROUTING_KINDS" evals`, run the full suite, and fix anything that breaks: give the new kinds builders in Task 4, or skip them where a v1 path iterates kinds.
- [ ] **Step 6: Commit** with the message "Cases: status_question and paraphrase kinds; split_override for whole-life splits".

---

### Task 4: `status_question` and `paraphrase` probe families

**Files:**
- Modify: `evals/worlds/lived/probes.py`
- Test: `evals/tests/test_lived_probes.py`

The new families need to be the following:

| Kind | Probe | Gold |
|---|---|---|
| `status_question` (known) | asks how a matter that has an agent is going; names the person or only the topic keyword | the thread's agents |
| `status_question` (unknown) | asks how something that was **never** started is going (a reserve errand) | `[]` with `clarify_ok=True`: the right move is to ask |
| `paraphrase` | refers to a matter whose contact is a service or a client with a unique company, by role or company only, never using the first name, last name or topic keyword | the thread's agents; `clarify_ok=True` when that contact has more than one agent-backed thread |

About 2 in 3 status probes are known and 1 in 3 unknown. Unknown ones are still made when no reserve topics remain.

- [ ] **Step 1: Failing tests** (append to `test_lived_probes.py`; follow how existing tests build a snapshot and a fake writer, and reuse those fixtures)

```python
async def test_status_question_probes(lived_snapshot_and_plan, fake_writer_llm):
    snapshot, plan = lived_snapshot_and_plan
    cases = await build_probes(snapshot, plan, llm=fake_writer_llm, model="m", per_kind=6, rng_seed=7)
    status = [c for c in cases if c.kind == "status_question"]
    assert status
    known = [c for c in status if c.gold_targets]
    unknown = [c for c in status if not c.gold_targets]
    assert known and unknown
    assert all(c.clarify_ok for c in unknown)
    served = {a for ids in snapshot.thread_agents.values() for a in ids}
    assert all(set(c.gold_targets) <= served for c in known)


async def test_paraphrase_probes_avoid_names_and_keywords(lived_snapshot_and_plan, fake_writer_llm):
    snapshot, plan = lived_snapshot_and_plan
    cases = await build_probes(snapshot, plan, llm=fake_writer_llm, model="m", per_kind=6, rng_seed=7)
    para = [c for c in cases if c.kind == "paraphrase"]
    assert para
    for case in para:
        assert case.gold_targets and case.gold_targets[0] != "NEW"
```

If the existing test module has no `lived_snapshot_and_plan` or `fake_writer_llm` fixture, build them the way its other tests do, for example with `round2.rebuild_snapshot` on the saved world plus `round2.fake_writer()`. A fake writer must satisfy each intent's mention and forbid rules. Read `round2.fake_writer` and make sure it echoes the mentions; extend it if not.

- [ ] **Step 2:** Run. Expected: fail.

- [ ] **Step 3: Implement** in `probes.py`, next to the other families:

```python
STATUS_DETAILS = (
    "ask how the {topic} is going",
    "ask if there's any word back on the {topic}",
)
STATUS_PERSON_DETAIL = "ask whether {first} got back to you about the {topic}"
UNKNOWN_STATUS_DETAIL = "ask how the {topic} is going, as if you had asked the assistant about it before"
PARAPHRASE_DETAIL = (
    "ask for an update on it without using the words {forbidden}: refer to the person only as "
    "{handle} and describe the topic in other words"
)


async def _status_question(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    unknown_want = want // 3
    for thread in p.shuffled(p.backed):
        if len(out) == want - unknown_want:
            break
        contact = p.contact(thread)
        topic = topic_keyword(thread.topic)
        if contact and p.rng.random() < 0.5:
            detail = STATUS_PERSON_DETAIL.format(first=contact.first, topic=topic)
            intent = p.thread_intent(thread, "status_question", detail)
        else:
            if not p.keyword_unique(thread):
                continue  # topic alone must point at one thread
            detail = p.rng.choice(STATUS_DETAILS).format(topic=topic)
            intent = p.thread_intent(thread, "status_question", detail, contacts=(), mention=(topic,), forbid=())
        text = await p.write(intent)
        if text:
            gold, acceptable = p.gold(thread)
            out.append((text, gold, {"acceptable_ids": acceptable}))
    while p.reserve and len(out) < want:
        topic = p.reserve.pop()
        keyword = topic_keyword(topic)
        intent = Intent(kind="status_question", thread_ids=(), topic=topic,
                        detail=UNKNOWN_STATUS_DETAIL.format(topic=keyword), mention=(keyword,))
        text = await p.write(intent)
        if text:
            out.append((text, [], {"clarify_ok": True}))
    return out


def _paraphrase_handle(p: _Probes, contact: Contact) -> str | None:
    """How a paraphrase names the person: 'my <role>' for a service, the company for a unique client."""
    role = service_role(contact)
    if contact.relation == "service" and role:
        return f"my {role}"
    if contact.relation == "client" and contact.company:
        if sum(1 for c in p.plan.contacts if c.company == contact.company) == 1:
            return f"the person at {contact.company}"
    return None


async def _paraphrase(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    for thread in p.shuffled(p.backed):
        if len(out) == want:
            break
        contact = p.contact(thread)
        handle = _paraphrase_handle(p, contact) if contact else None
        if not handle:
            continue
        keyword = topic_keyword(thread.topic)
        forbidden = (contact.first, contact.last, keyword)
        detail = PARAPHRASE_DETAIL.format(forbidden=", ".join(forbidden), handle=handle)
        mention = (handle.split()[-1],)  # the role word or the company name
        intent = p.thread_intent(thread, "paraphrase", detail, mention=mention, forbid=forbidden)
        text = await p.write(intent)
        if text:
            gold, acceptable = p.gold(thread)
            siblings = [t for t in p.backed if t.contact == thread.contact and t.id != thread.id]
            out.append((text, gold, {"acceptable_ids": acceptable, "clarify_ok": bool(siblings)}))
    return out
```

Also:
- Register both in `_BUILDERS` **after** `"agent_result"`, so v1 rng order for the existing kinds is unchanged.
- Add `service_role` to the `.plan` import.
- Check that `p.reserve` holds topic strings that `topic_keyword` resolves. It does today; keep it that way.

- [ ] **Step 4:** Run the probe and round-2 tests. If an existing test pins exact v1 probe output, it must still pass: new builders run last, so earlier kinds see the same rng draws.
- [ ] **Step 5:** Full suite.
- [ ] **Step 6: Commit** with the message "Probe families: status_question (known and never-started) and paraphrase (role or company, no names or keywords)".

---

### Task 5: The v2 builder (balanced, ambiguity-aware probe sets)

**Files:**
- Create: `evals/worlds/lived/bench_v2.py`
- Test: `evals/tests/test_bench_v2.py`

**What it does.** For one life (seed):
1. Rebuild each snapshot from the saved world: `world-s{seed}.json`, using `round2.rebuild_snapshot`.
2. Build probes with `build_probes(per_kind=QUOTA[size])`, where `QUOTA = {10: 3, 50: 5, 200: 10}`.
3. Mark ambiguity.
4. Write `bench-v2-s{seed}.json`:
   - `meta`: seed, world sha256, model, quotas;
   - `rosters`: label → agents with id, name and purpose;
   - `probes`: each case dict plus `size`, `labels` and `gold_agents`, in the same shape as round 2.

The 1,000 label reuses the 200 probes, as in round 2.

**Ambiguity.** A probe whose thread's contact has **another** agent-backed thread with overlapping topic words (`plan.topic_words`) gets `clarify_ok = True`.

**Balancing.** `balance(a, b)` trims both lives' probe lists so each (size, kind) has `min(count_a, count_b)` probes, keeping the lowest-numbered ones.

**Ids.** `v2-s{seed}-{size}-{kind}-{i:02d}`, renumbered after balancing.

- [ ] **Step 1: Failing tests** (`evals/tests/test_bench_v2.py`)

```python
from evals.worlds.lived.bench_v2 import QUOTA, balance, mark_ambiguous


def _probe(pid, size, kind):
    return {"id": pid, "size": size, "kind": kind}


def test_balance_trims_to_the_smaller_count_per_size_and_kind():
    a = [_probe(f"a{i}", 200, "typo") for i in range(5)] + [_probe("a9", 50, "typo")]
    b = [_probe(f"b{i}", 200, "typo") for i in range(3)]
    a2, b2 = balance(a, b)
    assert [p["id"] for p in a2] == ["v2-sx-200-typo-00", "v2-sx-200-typo-01", "v2-sx-200-typo-02"]
    assert len(b2) == 3 and not [p for p in a2 if p["size"] == 50]  # life b has no 50-agent typo probes


def test_quota():
    assert QUOTA == {10: 3, 50: 5, 200: 10}


def test_mark_ambiguous_flags_same_person_overlapping_topics():
    from evals.worlds.lived.model import Contact, LifePlan, Thread
    plan = LifePlan(seed=9, persona="p", contacts=[Contact("Ann", "Lee", "friend")], events=[], threads=[
        Thread("t1", "Ann Lee", "ski trip", "friend"), Thread("t2", "Ann Lee", "ski trip photos", "friend"),
        Thread("t3", "Ann Lee", "book club", "friend"),
    ])
    agents = {"t1": ["a1"], "t2": ["a2"], "t3": ["a3"]}
    assert mark_ambiguous(plan, agents, "t1") is True
    assert mark_ambiguous(plan, agents, "t3") is False
```

- [ ] **Step 2:** Run. Expected: ImportError.

- [ ] **Step 3: Implement `bench_v2.py`**

```python
"""Benchmark v2 probe sets: one per life, balanced across lives, ambiguity marked (spec v2 §3).

    python -m evals.worlds.lived.bench_v2 build --seed 1 --seed 2     # writes bench-v2-s{1,2}.json, balanced
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...cases import NEW
from ...llm import Budget, OpenRouterLLM
from ...run import CACHE_DIR, _api_key
from .model import LifePlan
from .plan import build_plan, topic_words
from .probes import build_probes
from .round2 import rebuild_snapshot

HERE = Path(__file__).resolve().parent
QUOTA = {10: 3, 50: 5, 200: 10}
FULL_SIZE = 1000
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"


def world_path(seed: int) -> Path:
    return HERE / f"world-s{seed}.json"


def bench_path(seed: int, *, reviewed: bool = False) -> Path:
    return HERE / f"bench-v2-s{seed}{'.reviewed' if reviewed else ''}.json"


def mark_ambiguous(plan: LifePlan, thread_agents: dict[str, list[str]], thread_id: str) -> bool:
    """Whether another agent-backed thread of the same person shares a topic word with this one."""
    thread = plan.thread(thread_id)
    if not thread.contact:
        return False
    words = topic_words(thread.topic)
    return any(
        other.id != thread_id and other.contact == thread.contact and thread_agents.get(other.id)
        and words & topic_words(other.topic)
        for other in plan.threads
    )


def balance(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Trim both lives to equal counts per (size, kind), keeping each life's first probes; renumber ids."""
    def groups(probes):
        out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for probe in probes:
            out[(probe["size"], probe["kind"])].append(probe)
        return out

    ga, gb = groups(a), groups(b)
    keep_a, keep_b = [], []
    for key in sorted(set(ga) | set(gb)):
        n = min(len(ga.get(key, [])), len(gb.get(key, [])))
        keep_a += _renumber(ga.get(key, [])[:n])
        keep_b += _renumber(gb.get(key, [])[:n])
    return keep_a, keep_b


def _renumber(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, probe in enumerate(probes):
        seed = probe.get("world_seed", "x")
        out.append({**probe, "id": f"v2-s{seed}-{probe['size']}-{probe['kind']}-{i:02d}"})
    return out
```

Test probes have no `world_seed`, so the seed in renumbered ids prints as `x`.

Then add the build pipeline. It reuses round 2's record shape, so the review page and finalize can share code:

```python
async def build_life(seed: int, *, llm, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    path = world_path(seed)
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = build_plan(seed)
    entries = {entry["size"]: entry for entry in data["sizes"]}
    rosters = {f"lived-{s}": [{"id": a["id"], "name": a["name"], "purpose": a.get("purpose", "")}
                              for a in e["agents"]] for s, e in entries.items()}
    probes: list[dict[str, Any]] = []
    for size, quota in QUOTA.items():
        snapshot = rebuild_snapshot(entries[size], plan, skipped=data["meta"]["skipped"])
        cases = await build_probes(snapshot, plan, llm=llm, model=model, per_kind=quota,
                                   rng_seed=seed * 7000 + size)
        labels = [f"lived-{size}"] + ([f"lived-{FULL_SIZE}"] if size == 200 and FULL_SIZE in entries else [])
        by_id = {a["id"]: a for a in rosters[f"lived-{size}"]}
        for case in cases:
            record = {**asdict(case), "size": size, "labels": labels,
                      "gold_agents": [by_id[g] for g in case.gold_targets + case.acceptable_ids if g != NEW]}
            thread = _thread_of(case, snapshot)
            if thread and mark_ambiguous(plan, snapshot.thread_agents, thread):
                record["clarify_ok"] = True
            probes.append(record)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"meta": {"seed": seed, "world_sha256": sha, "model": model, "quota": QUOTA},
            "rosters": rosters, "probes": probes}


def _thread_of(case, snapshot) -> str | None:
    """The thread whose first agent is the case's best answer (None for new-agent or never-started probes)."""
    best = next((g for g in case.gold_targets if g != NEW), None)
    return next((t for t, ids in snapshot.thread_agents.items() if ids and ids[0] == best), None)
```

**CLI.** `build` takes `--seed` twice, `--model`, and `--budget` (default 2.0):
- it calls `build_life` for each seed;
- then `balance(...)` on the two `probes` lists;
- then writes `bench_path(seed)` for each.

Follow `round2.main`'s structure: api key check, `OpenRouterLLM(... budget=Budget(args.budget))`, and `asyncio.run` with `aclose` in a `finally`. Add a `--fake-llm` flag that uses `round2.fake_writer()` so the pipeline can run offline.

- [ ] **Step 4: Offline check**

Life 2's world doesn't exist yet (Task 6). Run life 1 alone once, with balancing skipped, through a test that monkeypatches `world_path` to `world-s1.json` for both seeds:

```python
async def test_build_life_offline_on_life1(monkeypatch):
    from evals.worlds.lived import bench_v2
    from evals.worlds.lived.round2 import fake_writer
    body = await bench_v2.build_life(1, llm=fake_writer(), model="m")
    kinds = {p["kind"] for p in body["probes"]}
    assert {"status_question", "paraphrase", "follow_up"} <= kinds
    assert all(len([p for p in body["probes"] if (p["size"], p["kind"]) == key]) <= bench_v2.QUOTA[key[0]]
               for key in {(p["size"], p["kind"]) for p in body["probes"]})
```

- [ ] **Step 5:** Full suite.
- [ ] **Step 6: Commit** with the message "Benchmark v2 builder: per-size quotas, both lives balanced per kind, ambiguity marked".

---

### Task 6: Grow life 2 (live, about $2–3)

**Files:** Create `evals/worlds/lived/world-s2.json`.

- [ ] **Step 1:** Make sure the tree is clean and the checksum is committed: `.venv/Scripts/python -m evals.integrity` must show matching hashes.
- [ ] **Step 2: Grow**

```bash
.venv/Scripts/python -m evals.worlds.lived.generate --seed 2 --per-kind 0 --budget 4 --out evals/worlds/lived/world-s2.json
```

`--per-kind 0` makes no v1 probes; v2 probes come from Task 7. It runs in the background and takes tens of minutes.

- [ ] **Step 3: Check the result**

```bash
.venv/Scripts/python -c "import json; d=json.load(open('evals/worlds/lived/world-s2.json',encoding='utf-8')); print(d['meta'], [(e['size'], len(e['agents'])) for e in d['sizes']])"
```

Expected:
- `meta.skipped == 0` (the snapshot rebuild requires it);
- sizes 10, 50, 200 and 1000 with those agent counts.

If skipped is above 0, stop and report. Don't build probes on a world that can't be rebuilt.

- [ ] **Step 4: Rebuild check** (free)

```bash
.venv/Scripts/python -c "
import json; from evals.worlds.lived.plan import build_plan; from evals.worlds.lived.round2 import rebuild_snapshot
d=json.load(open('evals/worlds/lived/world-s2.json',encoding='utf-8'))
for e in d['sizes']:
    if e['size']!=1000: s=rebuild_snapshot(e, build_plan(2), skipped=0); assert [a.id for a in s.agents]==[a['id'] for a in e['agents']]; print(e['size'],'ok')"
```

- [ ] **Step 5: Commit** `world-s2.json` with the message "Life 2 (Riley) grown through the real IA on DeepSeek V4.1 Flash: <n> events, 0 skipped, $<cost>".

**Do not** open or summarize life 2's rosters beyond these checks. The test life stays unseen.

---

### Task 7: Build both probe sets (live, about $0.50)

- [ ] **Step 1: Build**

```bash
.venv/Scripts/python -m evals.worlds.lived.bench_v2 build --seed 1 --seed 2 --budget 2
```

- [ ] **Step 2: Check the counts.** Per (size, kind), both files must have equal counts, and the totals should be roughly 150–180 probes per life:

```bash
.venv/Scripts/python -c "
import json, collections
for s in (1, 2):
    p = json.load(open(f'evals/worlds/lived/bench-v2-s{s}.json', encoding='utf-8'))['probes']
    print(s, len(p), sorted(collections.Counter((x['size'], x['kind']) for x in p).items()))"
```

If any kind at 200 has fewer than 5 probes, report it before going on.

- [ ] **Step 3: Commit** both files with the message "Benchmark v2 probe sets for both lives, balanced: <n> probes each".

---

### Task 8: Three-model checks (live, about $1.50)

**Files:**
- Create: `evals/worlds/lived/checkers.py`
- Test: `evals/tests/test_checkers.py`

**Design.**
- **Candidate cards** for each probe:
  - the gold and acceptable agents;
  - every agent sharing a contact name with them (first + last name appearing in the agent name or purpose);
  - the top retrieval matches from production `select`, up to 10 total, shuffled with a seeded rng.

  Use `evals/systems/cards_from_world.py` to turn the probe's roster into `AgentCard`s, then `server.services.agents.retrieval.select(message, cards, now)`. Read `cards_from_world.py` for the exact constructor.
- **Prompt:** system plus user. It contains the last 6 context entries, the message and the numbered candidate cards (name and purpose). It asks for JSON `{"answer": "<card number>" | "NEW" | "ASK"}`. A parse failure counts as disagreement.
- **Agreement:** a checker's answer agrees when it is:
  - a card whose agent id is in gold or acceptable;
  - `NEW` when gold contains `NEW`;
  - `ASK` when `clarify_ok`;
  - `NEW` or `ASK` when gold is `[]` and `clarify_ok`.
- **Flagging:** a probe is flagged when any checker disagrees. Otherwise a seeded 10% sample (`random.Random(seed).random() < 0.10`, one draw per probe in id order) is flagged as `audit`.
- **Output:** each probe gets `"check": {"answers": {model: answer}, "agree": bool, "flag": null | "disagree" | "audit"}`.

```python
CHECKERS = ("openai/gpt-6-luna", "google/gemini-3.1-flash-lite", "qwen/qwen3.8-flash")
AUDIT_SHARE = 0.10
```

- [ ] **Step 1: Failing tests** (`evals/tests/test_checkers.py`). Use `ScriptedLLM` (from `evals.llm`) returning fixed JSON answers. Test:
  1. all three agree with gold → not flagged (unless audit);
  2. one says another card → flagged `disagree`;
  3. `ASK` on a `clarify_ok` probe agrees;
  4. unparseable output → `disagree`;
  5. the audit draw is deterministic for a seed.

  Write each test against a hand-built two-card roster and a single probe dict in the bench record shape (`id`, `message`, `context`, `gold_targets`, `acceptable_ids`, `clarify_ok`, `size`, `labels`).

- [ ] **Step 2:** Run. Expected: fail.

- [ ] **Step 3: Implement** `checkers.py` with:
  - `candidates(probe, roster, now, rng) -> list[dict]`;
  - `prompt(probe, cards) -> tuple[str, str]`;
  - `parse(text, cards) -> str` (an agent id, `NEW`, `ASK` or `INVALID`);
  - `agrees(answer, probe) -> bool`;
  - `async check_bench(bench, *, llm, seed) -> dict`, which returns the bench with `check` fields;
  - a CLI: `python -m evals.worlds.lived.checkers --seed 1 --seed 2 --budget 3`. It rewrites `bench-v2-s{seed}.json` in place and prints the counts per flag.

  Use `llm.complete(model=..., system=..., messages=[...], tools=[], sample=0)`, and read the text from `choices[0].message.content`.

- [ ] **Step 4:** Tests pass, then the full suite.
- [ ] **Step 5:** Commit the code with the message "Three-model key checks: disagreements and a 10% audit sample go to human review".
- [ ] **Step 6: Run live**

```bash
.venv/Scripts/python -m evals.worlds.lived.checkers --seed 1 --seed 2 --budget 3
```

Expected: about 40–60 flagged probes in total. Commit both bench files with the message "Checker results: <d> disagreements, <a> audit samples".

---

### Task 9: Human review page and finalize

**Files:**
- Modify: `evals/labelling/probes_page.html`. Show `check.answers` under the proposed answer as a small "Checkers said: …" line naming each answer's card, `NEW` or `ASK`.
- Modify: `evals/worlds/lived/bench_v2.py`. Add `review-data` and `finalize` subcommands.
- Test: `evals/tests/test_bench_v2.py`

**`review-data`**
- Writes the page's embedded JSON: `{source_sha256, rosters, probes}`, with only flagged probes from both lives.
- Rosters are keyed `s{seed}:lived-{size}` so both lives fit. Each probe carries `roster_key`.
- Update the page's `rosterOf(p)` to use `DATA.rosters[p.roster_key]` when it's present, falling back to the current key.
- Build the page exactly as before: replace `__PROBES__` with the JSON, `</` escaped.

**`finalize`**
- Takes `--decisions decisions.json`: a list of `{id, decision, note, gold_targets?, acceptable_ids?, clarify_ok?}`, the same shape `round2.apply_decisions` accepts.
- Writes `bench-v2-s{seed}.reviewed.json` per life with:
  - unflagged probes, kept;
  - `accept`, kept;
  - `fixed`, kept with the new key;
  - `reject`, dropped.
- Every flagged probe must have a decision. Otherwise it raises `ValueError("undecided: ...")`.
- `meta` records the counts and `label_error_rate` = audit probes fixed or rejected ÷ audit probes.

- [ ] **Step 1: Failing tests** for `finalize`:
  1. undecided raises;
  2. reject drops;
  3. fixed replaces gold;
  4. unflagged probes are kept untouched;
  5. `label_error_rate` is computed over audit probes only.
- [ ] **Step 2:** Implement, then run the tests and the full suite.
- [ ] **Step 3: Build and publish the page**

Publish it as a **new** artifact (the old probe-review artifact holds v1 decisions), with the `db` capability and a `reviews` collection, as before. **Then stop and ask the user to review**: expected 40–60 probes, about 30–45 minutes.

- [ ] **Step 4: Apply the review**

Once the user says the review is done:
1. Read the `reviews` collection.
2. Convert it to `decisions.json` (one entry per doc, using its `probe` field as id).
3. Run `finalize`.

Then:
1. Add `evals/worlds/lived/bench-v2-s1.reviewed.json`, `bench-v2-s2.reviewed.json` and `evals/worlds/lived/bench_v2.py` to `PROTECTED` in `evals/integrity.py`.
2. Run `--update` and the full suite.
3. Commit with the message "Benchmark v2 reviewed: <kept> probes per life, label error rate <r> (audit)".

---

### Task 10: Severity rates

**Files:**
- Modify: `evals/grading.py`
- Modify: `evals/metrics.py` (`_routing_group`)
- Test: `evals/tests/test_grading.py`, `evals/tests/test_metrics.py`

- [ ] **Step 1: Failing tests**

```python
# test_grading.py
from evals.grading import SEVERITY, CORRECT, WRONG_REUSE, MISSED_CREATE, DUPLICATE_CREATE, EXTRA_CREATE, PARTIAL, \
    SHOULD_CLARIFY, UNNECESSARY_CLARIFY, NO_ACTION


def test_every_outcome_has_a_severity():
    assert SEVERITY[CORRECT] is None
    assert SEVERITY[WRONG_REUSE] == SEVERITY[MISSED_CREATE] == "harmful"
    assert SEVERITY[DUPLICATE_CREATE] == SEVERITY[EXTRA_CREATE] == "costly"
    assert {SEVERITY[o] for o in (PARTIAL, SHOULD_CLARIFY, UNNECESSARY_CLARIFY, NO_ACTION)} == {"minor"}
```

```python
# test_metrics.py: build rows the way the module's existing tests do, then:
def test_harmful_and_costly_rates():
    rows = [_row("correct"), _row("wrong_reuse"), _row("duplicate_create"), _row("unnecessary_clarify")]
    group = summarize_routing(rows)["lived-200"]
    assert group["harmful_rate"] == 0.25 and group["costly_rate"] == 0.25 and group["minor_rate"] == 0.25
```

Reuse or write a `_row(outcome)` helper with every key `_routing_group` reads: `outcome`, `world_seed`, `case_id` (unique per row), `sample`, `kind`, `iterations`, `hit_iteration_cap`, `expects_reuse`, `duplicate_possible`, `prompt_tokens`, `cost`, `fresh_cost`, `cached_calls` and `n_agents="lived-200"`.

- [ ] **Step 2: Implement**

In `grading.py`, after the outcome constants:

```python
# How much a wrong outcome would hurt the user (spec v2 §3.5). harmful: work reaches the wrong agent or
# matter; costly: a duplicate agent that will drift; minor: an unneeded question, a half-done or no action.
SEVERITY: dict[str, str | None] = {
    CORRECT: None,
    WRONG_REUSE: "harmful",
    MISSED_CREATE: "harmful",
    DUPLICATE_CREATE: "costly",
    EXTRA_CREATE: "costly",
    PARTIAL: "minor",
    SHOULD_CLARIFY: "minor",
    UNNECESSARY_CLARIFY: "minor",
    NO_ACTION: "minor",
}
```

In `metrics._routing_group`, add `harmful_rate`, `costly_rate` and `minor_rate`: the share of `valid` rows whose outcome's severity is each level, using `_rate`. Import `SEVERITY`.

- [ ] **Step 3:** Run the tests and the full suite.
- [ ] **Step 4: Commit** with the message "Severity per outcome; harmful, costly and minor rates per size".

---

### Task 11: Load benchmark v2 in the runner

**Files:**
- Modify: `evals/worlds/lived/bench_v2.py`. Add `load_v2(seed) -> list[tuple[str, World, list[Case]]]`.
- Modify: `evals/run.py`:
  - `Config.world` accepts `lived_v2`;
  - new `Config.life: int = 1`;
  - validation;
  - unit loading.
- Create: `evals/experiments/v2-baseline.yaml`, `v2-retrieved.yaml` and `v2-full-cards.yaml`.
- Test: `evals/tests/test_bench_v2.py`, `evals/tests/test_run.py`

**`load_v2(seed)`**
- Reads `world_path(seed)` for rosters and `now`, the same way `store.load_lived` builds `World`s.
- Reads `bench_path(seed, reviewed=True)` for the cases.
- Builds `Case(**fields, split_override="dev" if seed == 1 else "test")` with only `Case` fields. Context entries become tuples.
- Each case joins every label in its `labels`.
- Returns units smallest first.

**Validation in `run.py`**
- `world: lived_v2` needs `life` to be 1 or 2.
- Life 2 runs only with `--split test --final`.
- Life 1 runs only on dev.
- Messages: "life 2 is the sealed test life: run it with --split test --final" and "life 1 is the dev life".

- [ ] **Step 1: Failing tests**
  - `load_v2(1)` gives only dev cases with labels `lived-10` … `lived-1000`. This test runs after Task 9 commits the reviewed files. Until then, mark it `pytest.mark.skipif(not bench_path(1, reviewed=True).exists(), ...)`.
  - The config validation messages above.
- [ ] **Step 2:** Implement.

- [ ] **Step 3: Configs**, all with `model: deepseek/deepseek-v4.1-flash`, `world: lived_v2`, `life: 1`, `n_agents: [10, 50, 200, 1000]`, `k: 3`, `include_handwritten: false` and `concurrency: 4`:

| Config | `system` | `budget_usd` | Other |
|---|---|---|---|
| `v2-baseline.yaml` | `baseline` | 3.0 | |
| `v2-retrieved.yaml` | `retrieved` | 3.0 | |
| `v2-full-cards.yaml` | `full_cards` | 4.0 | `n_agents: [10, 50, 200]` |

- [ ] **Step 4:** Offline check: `--fake-llm` runs of all three configs exit 0. Then the full suite, `integrity --update` (`run.py` is protected) and a commit with the message "Runner loads benchmark v2: life 1 as dev, life 2 sealed as test; v2 configs".

---

### Task 12: Re-baseline on v2 dev (live, about $3–5)

- [ ] **Step 1:** Confirm `.venv/Scripts/python -m evals.integrity` shows matching hashes.
- [ ] **Step 2: Run** (in the background, one after another)

```bash
.venv/Scripts/python -m evals.run --config evals/experiments/v2-baseline.yaml
.venv/Scripts/python -m evals.run --config evals/experiments/v2-retrieved.yaml
.venv/Scripts/python -m evals.run --config evals/experiments/v2-full-cards.yaml
```

- [ ] **Step 3: Intervals.** Add the three run files to a `V2_DEV` dict in `evals/charts.py`, next to `FINAL`. Print per-size correct-or-acceptable (pass@1) with 95% intervals, and the paired retrieved − baseline difference. Reuse `summary()` and `paired_difference()` by passing the runs dict. Add a `--runs v2-dev` flag to `main`.
- [ ] **Step 4: Journal entry** (`evals/experiments/JOURNAL.md`):
  - per size and system: pass@1 / pass^k / harmful / costly / duplicate rates with intervals;
  - prompt tokens per decision;
  - per-kind pass for `status_question`, `paraphrase` and `vague_reference`;
  - the 3–5 worst failure clusters, with one example transcript each.

  These are the starting point for Plan 6c.
- [ ] **Step 5: Commit** the results and journal with the message "v2 dev baseline: original vs retrieval vs full cards on life 1".
- [ ] **Step 6: Stop and report to the user.** Give the numbers against the spec §6 targets and the failure clusters, plus the spend so far. Plan 6c (the system changes) is written from these clusters.

---

## Self-review notes

- **Spec coverage, §3:**

  | Spec item | Task |
  |---|---|
  | 3.1 worlds | 2, 6 |
  | 3.2 stratified probes and new kinds | 3, 4, 5, 7 |
  | 3.3 keys from construction, including ambiguity | 4, 5 |
  | 3.4 verification and review | 8, 9 |
  | 3.5 scoring | 10, plus intervals in 12 |
  | 3.6 L1 recall | via the existing retrieval layer in the Task 12 runs |
  | 3.8 v1 retirement | the v1 world and files untouched; v2 uses its own files and loader |
  | §7 step 4 re-baseline | 12 |

- **Deferred to Plan 6d:** 3.7 (IRC check, optional).
- **Life 1 protection:** the Task 1 golden digest guards it, and Task 4 appends the new builders last so v1 probe rng order is unchanged.
