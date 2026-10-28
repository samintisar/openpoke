"""Seeded synthetic worlds whose correct routing answers are known by construction.

A world has a fixed core (the agents cases are about) plus filler agents that act as
distractors. Filler never gives a core person an agent on a core topic, so gold labels stay
correct at every roster size, and smaller worlds are prefixes of the full one.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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
    "find a dog walker for next week",
    "order more printer ink",
)
NEW_PERSON_TASKS = (
    "email {first} {last} about the {phrase}",
    "can you reach out to {first} {last} about the {phrase}",
    "draft a note to {first} {last} about the {phrase}",
    "start an email thread with {first} {last} on the {phrase}",
)
FOLLOW_UPS = (
    "add a P.S. to the {phrase} email to {name} saying I'm free Thursday",
    "can you make the {phrase} email to {name} shorter",
    "tell {name} I need one more day on the {phrase}",
    "update the {phrase} draft for {name} to mention the budget",
    "remind {name} that the {phrase} paperwork is due Friday",
)
SAME_PERSON_NEW_TOPIC = (
    "email {name} about the {phrase}",
    "can you reach out to {name} about the {phrase}",
    "start a new email to {name} about the {phrase}",
    "write to {name} and ask about the {phrase}",
)
VAGUE = (
    "any reply yet?",
    "did they get back to us?",
    "anything back on that?",
    "heard anything?",
    "any response on that one?",
    "did that get a reply?",
)
CROSS = (
    "let {main} know {colleague} signed off on the pricing",
    "tell {main} that {colleague} approved the pricing",
    "can you let {main} know the pricing has {colleague}'s approval",
    "pass along to {main} that {colleague} okayed the pricing",
    "update {main}: {colleague} gave the green light on pricing",
)
DORMANT = (
    "any update on the {phrase} with {name}?",
    "whatever happened with {name}'s {phrase}?",
    "can you check where the {phrase} with {name} ended up?",
    "let's pick the {phrase} with {name} back up",
    "is the {phrase} with {name} still open?",
)
TYPO = (
    "can you nudge {name} on the {phrase}?",
    "did {name} answer about the {phrase}?",
    "ping {name} again about the {phrase}",
    "any word from {name} on the {phrase}?",
    "follow up with {name} on the {phrase} please",
)
TWIN_TOPIC = (
    "follow up with {first} about the {phrase}",
    "did {first} reply about the {phrase}?",
    "ping {first} on the {phrase}",
)
TWIN_VAGUE = (
    "did {first} get back to me?",
    "what's the latest with {first}?",
    "any news from {first}?",
    "has {first} replied yet?",
)
AGENT_RESULTS = (
    "{first} replied on the {phrase} thread asking to move the call to Thursday.",
    "{first} confirmed the {phrase} and asked for the final numbers.",
    "{first} sent a question about the {phrase} timeline.",
    "{first} wrote back on the {phrase}: they need a signature by Monday.",
)


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
        purpose: str | None = None,
    ) -> AgentSpec | None:
        """Add an agent for person on topic; purpose overrides the default description."""
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
                purpose=purpose or f"{phrase.capitalize()} with {person.full} at {person.company}{suffix}",
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
    """The seeded world; build_world(s, n) is the first n agents of build_world(s)."""
    rng = random.Random(seed)
    # Drawn first so it never depends on how many filler agents follow.
    reserved_topic = rng.choice(RESERVED_TOPICS)
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

    def core(
        role: str, person: Person, topic: tuple[str, str], low: float, high: float, purpose: str | None = None
    ) -> None:
        agent = builder.add_person_agent(person, topic, days=rng.uniform(low, high), core=True, purpose=purpose)
        assert agent is not None
        roles[role] = agent.id

    core("main", main, t_main, 0.5, 4)
    core("main_neg", main, t_main_neg, 5, 20)
    core("twin", twin, t_twin, 1, 10)
    core("colleague", colleague, CROSS_TOPIC, 1, 6)
    # The main person's side of the colleague's deal: where "let main know colleague approved" goes.
    core(
        "main_cross", main, CROSS_TOPIC, 1, 6,
        purpose=f"{CROSS_TOPIC[0].capitalize()} for {colleague.full}'s deal with {main.full} at {main.company}",
    )
    core("dormant", dormant, t_dormant, 35, 60)
    core("typo", typo, t_typo, 1, 8)
    core("other", other, t_other, 0.5, 6)
    # Not a gold target of any case kind: a core distractor every roster size keeps.
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
            "main": main, "main_neg": main, "main_cross": main, "twin": twin, "colleague": colleague,
            "dormant": dormant, "typo": typo, "other": other,
        },
        reserved_topic=reserved_topic,
    )

@dataclass
class _Draft:
    """A candidate case before it gets an id: message, gold and extra Case fields."""

    message: str
    gold: list[str]
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        return self.message, tuple(self.extra.get("context", ()))


class _Kit:
    """Looks up a world's role agents for the case builders."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.agents = world.by_id()

    def target(self, role: str) -> tuple[AgentSpec, Person]:
        return self.agents[self.world.roles[role]], self.world.role_people[role]

    def follow_up(self, role: str, rng: random.Random) -> tuple[AgentSpec, str]:
        """A follow-up line for role's agent; main and twin go by full name (they share a first)."""
        agent, person = self.target(role)
        name = person.full if role in ("main", "main_neg", "twin") else person.first
        return agent, rng.choice(FOLLOW_UPS).format(phrase=agent.topic, name=name)


def _date(moment: datetime) -> str:
    """Short date such as "Oct 26"."""
    return f"{moment:%b} {moment.day}"


def _follow_up(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    role = ("main", "colleague", "typo", "other", "twin", "main_neg")[i % 6]
    agent, message = kit.follow_up(role, rng)
    return _Draft(message, [agent.id])


def _new_task(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    if i % 2 == 0:
        first, last = NEW_PEOPLE[(i // 2) % len(NEW_PEOPLE)]
        template = rng.choice(NEW_PERSON_TASKS)
        return _Draft(template.format(first=first, last=last, phrase=rng.choice(TOPICS)[0]), [NEW])
    return _Draft(NEW_TASKS[(i // 2) % len(NEW_TASKS)], [NEW])


def _same_person_new_topic(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    main = kit.world.role_people["main"]
    assert kit.world.reserved_topic is not None
    template = SAME_PERSON_NEW_TOPIC[(i // 2) % len(SAME_PERSON_NEW_TOPIC)]
    phrase = kit.world.reserved_topic[0]
    if i % 2 == 0:
        return _Draft(template.format(name=main.full, phrase=phrase), [NEW])
    # The first name alone also fits the twin: asking which one is fine, and a new agent is right either way.
    return _Draft(template.format(name=main.first, phrase=phrase), [NEW], {"clarify_ok": True})


def _vague_reference(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    agent, person = kit.target(rng.choice(("main", "other", "colleague", "typo")))
    if rng.random() < 0.5:
        context = [
            ("user_message", f"email {person.full} about the {agent.topic}"),
            ("poke_reply", f"drafted it for {person.first}, want me to send it?"),
            ("user_message", "yes send it"),
            ("poke_reply", "sent"),
        ]
    else:
        sent_on = _date(agent.last_active_at(kit.world.now))
        context = [
            ("conversation_summary",
             f"Earlier the user asked to email {person.full} about the {agent.topic}; the email was sent on {sent_on}."),
            ("user_message", "thanks"),
            ("poke_reply", "anytime"),
        ]
    return _Draft(VAGUE[i % len(VAGUE)], [agent.id], {"context": context})


def _multi_task(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    agent, follow = kit.follow_up(("main", "other", "colleague", "typo")[i % 4], rng)
    return _Draft(f"{follow}, and also {rng.choice(NEW_TASKS)}", [agent.id, NEW])


def _same_name_people(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    first = kit.world.role_people["main"].first
    if i % 2 == 1:
        return _Draft(TWIN_VAGUE[(i // 2) % len(TWIN_VAGUE)].format(first=first), [], {"clarify_ok": True})
    # Main and twin share a first name; each topic belongs to only one of them, so it decides.
    agent, _ = kit.target(("twin", "main", "twin", "main_neg")[(i // 2) % 4])
    return _Draft(rng.choice(TWIN_TOPIC).format(first=first, phrase=agent.topic), [agent.id])


def _cross_agent(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    cross, main = kit.target("main_cross")
    colleague_agent, colleague = kit.target("colleague")
    message = CROSS[i % len(CROSS)].format(main=main.full, colleague=rng.choice((colleague.first, colleague.full)))
    return _Draft(message, [cross.id], {"acceptable_ids": [colleague_agent.id]})


def _dormant_revival(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    agent, person = kit.target("dormant")
    name = rng.choice((person.first, person.full))
    return _Draft(DORMANT[i % len(DORMANT)].format(phrase=agent.topic, name=name), [agent.id])


def _typo_case(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    agent, person = kit.target("typo")
    message = TYPO[i % len(TYPO)].format(name=_typo(person.first, rng), phrase=agent.topic)
    return _Draft(message, [agent.id])


def _agent_result(kit: _Kit, i: int, rng: random.Random) -> _Draft:
    roles = ("main", "colleague", "other", "dormant", "main_cross", "typo", "twin", "main_neg")
    agent, person = kit.target(roles[i % len(roles)])
    body = rng.choice(AGENT_RESULTS).format(first=person.first, phrase=agent.topic)
    return _Draft(f"[SUCCESS] {agent.name}: {body}", [agent.id], {"message_type": "agent", "dispatch_optional": True})


_BUILDERS: dict[str, Callable[[_Kit, int, random.Random], _Draft]] = {
    "follow_up": _follow_up,
    "new_task": _new_task,
    "same_person_new_topic": _same_person_new_topic,
    "vague_reference": _vague_reference,
    "multi_task": _multi_task,
    "same_name_people": _same_name_people,
    "cross_agent": _cross_agent,
    "dormant_revival": _dormant_revival,
    "typo": _typo_case,
    "agent_result": _agent_result,
}


def generate_cases(world: World, per_kind: int) -> list[Case]:
    """Up to per_kind unique cases of every routing kind, all about the world's core agents.

    Each kind draws candidates with a running index and keeps those whose (message, context)
    it has not produced yet, stopping at per_kind kept or after per_kind * 20 candidates. A kind
    whose template and role combinations run out therefore yields fewer than per_kind cases
    (cross_agent, for one, has at most ten distinct messages per world). Ids number the kept
    cases: f"{kind}-s{seed}-{i:02d}".
    """
    kit = _Kit(world)
    cases: list[Case] = []
    for offset, (kind, builder) in enumerate(_BUILDERS.items()):
        rng = random.Random(world.seed * 7919 + 17 + offset)
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for index in range(per_kind * 20):
            if len(seen) == per_kind:
                break
            draft = builder(kit, index, rng)
            if draft.key() in seen:
                continue
            cases.append(Case(
                id=f"{kind}-s{world.seed}-{len(seen):02d}", kind=kind, world_seed=world.seed,
                message=draft.message, gold_targets=draft.gold, **draft.extra,
            ))
            seen.add(draft.key())
    return cases
