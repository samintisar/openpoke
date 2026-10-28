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

        cross_agent, main_person = target("main_cross")
        colleague_agent, colleague_person = target("colleague")
        add(
            "cross_agent", i,
            rng.choice(CROSS).format(main=main_person.full, colleague=colleague_person.first),
            [cross_agent.id], acceptable_ids=[colleague_agent.id],
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
