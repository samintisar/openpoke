"""Held-out probe cases for one grown snapshot, one family per routing kind.

Probes are never played into growth. Gold comes from the snapshot's thread map: the agents that
served a thread, first-served first, so `gold_targets=[first]` and `acceptable_ids=rest`. Each probe
is an Intent the writer turns into text; a probe whose writer returns None is skipped.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from evals.cases import NEW, Case
from evals.llm import LLMClient
from evals.worlds.lived.model import Contact, Intent, LifePlan, Snapshot, Thread
from evals.worlds.lived.plan import (
    ERRAND_RESULT_DETAILS, ERRANDS, FOLLOW_UP_DETAILS, RESULT_DETAILS, REVIVAL_DETAILS, SERVICE_TOPICS, SERVICES,
    START_DETAILS, TOPICS, topic_keyword,
)
from evals.worlds.lived.writer import write

CONTEXT = 20  # transcript entries a probe's context holds
TAIL = 6  # transcript entries the writer sees
ACTIVE_DAYS = 14.0  # follow_up / typo: last event at most this many days ago
DORMANT_DAYS = 21.0  # dormant_revival: idle at least this long

VAGUE_DETAIL = (
    "ask if there's any update on it, referring to it only as 'that' or 'it'; "
    "the conversation above makes clear what you mean"
)
TOPIC_ONLY_DETAIL = "ask how the {topic} is going, without saying who it's with"
SAME_NAME_DETAILS = (
    "ask if there's any update from {first}",
    "ask whether {first} ever got back to you",
    "say you need to check in with {first} and ask where things stand",
)
CROSS_VERBS = ("approved", "signed off on")
CROSS_DETAIL = "ask the assistant to let {a} know that {b} {verb} the {topic}"
NEW_TOPIC_DETAIL = "a new request about the {topic}: {detail}"
MULTI_DETAIL = "two requests in one message: {a} (about: {a_topic}); and {b} (about: {b_topic})"
NEW_ERRAND_DETAIL = "a brand-new request to {topic}: {detail}"


def _group(domain: str) -> str:
    """The detail pool a thread's domain draws from (mirrors plan.py)."""
    if domain in ("work", "client", "recruiter"):
        return "pro"
    if domain in ("family", "friend"):
        return "personal"
    return domain


def _relation_pool(contact: Contact) -> tuple[str, ...]:
    """Topics a contact's relation could have; for services, the role's own pool."""
    if contact.relation == "service":
        role = next((r for first, last, _, r in SERVICES if (first, last) == (contact.first, contact.last)), None)
        return tuple(topic for topic, _ in SERVICE_TOPICS.get(role, ()))
    return tuple(topic for topic, _ in TOPICS.get(contact.relation, ()))


def typo_name(name: str) -> str:
    """A deterministic misspelling: swap the first differing pair of inner letters, else double the last."""
    if len(name) >= 4:
        for i in range(1, len(name) - 2):
            if name[i].lower() != name[i + 1].lower():
                return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    return name + name[-1]


def corrupt_name(text: str, first: str) -> str:
    """`text` with every occurrence of `first` misspelled, keeping each occurrence's own casing."""
    def swap(match: re.Match[str]) -> str:
        return typo_name(match.group(0))

    out, count = re.subn(rf"\b{re.escape(first)}\b", swap, text, flags=re.IGNORECASE)
    if not count:
        out = re.sub(re.escape(first), swap, text, flags=re.IGNORECASE)
    return out


class _Probes:
    """Snapshot state, name rules and the writer call shared by every probe family."""

    def __init__(self, snapshot: Snapshot, plan: LifePlan, llm: LLMClient, model: str, rng: random.Random) -> None:
        self.snapshot, self.plan, self.llm, self.model, self.rng = snapshot, plan, llm, model, rng
        self.context = [tuple(entry) for entry in snapshot.transcript[-CONTEXT:]]
        self.context_text = " ".join(entry[1] for entry in self.context).lower()
        self.tail = list(snapshot.transcript[-TAIL:])
        self.agents = {agent.id: agent for agent in snapshot.agents}
        self.by_full = {c.full: c for c in plan.contacts}
        self.namesakes: dict[str, list[Contact]] = {}
        for contact in plan.contacts:
            self.namesakes.setdefault(contact.first, []).append(contact)
        self.backed = [t for t in plan.threads if snapshot.thread_agents.get(t.id)]
        self.backed_ids = {t.id for t in self.backed}
        used = {t.topic for t in plan.threads}
        self.reserve = [topic for topic, _ in ERRANDS if topic not in used and not self._matches_thread(topic)]
        self.rng.shuffle(self.reserve)

    # --- names and gold ---------------------------------------------------------------------

    def _matches_thread(self, errand: str) -> bool:
        """Whether a reserve errand's keyword appears in any plan thread's topic or keyword."""
        keyword = topic_keyword(errand).lower()
        return any(keyword in t.topic.lower() or keyword in topic_keyword(t.topic).lower() for t in self.plan.threads)

    def contact(self, thread: Thread) -> Contact | None:
        return self.by_full[thread.contact] if thread.contact else None

    def names(self, contact: Contact) -> tuple[str, ...]:
        """First name, plus the last name when another contact shares the first."""
        if len(self.namesakes[contact.first]) > 1:
            return contact.first, contact.last
        return (contact.first,)

    def namesake_lasts(self, contact: Contact) -> tuple[str, ...]:
        return tuple(c.last for c in self.namesakes[contact.first] if c.full != contact.full)

    def thread_mention(self, thread: Thread) -> tuple[str, ...]:
        contact = self.contact(thread)
        return (self.names(contact) if contact else ()) + (topic_keyword(thread.topic),)

    def thread_forbid(self, thread: Thread) -> tuple[str, ...]:
        contact = self.contact(thread)
        return self.namesake_lasts(contact) if contact else ()

    def gold(self, thread: Thread) -> tuple[list[str], list[str]]:
        ids = self.snapshot.thread_agents[thread.id]
        return [ids[0]], list(ids[1:])

    def idle(self, thread: Thread) -> float | None:
        last = self.snapshot.thread_last_day.get(thread.id)
        return None if last is None else self.snapshot.day - last

    def shuffled(self, items: Sequence) -> list:
        out = list(items)
        self.rng.shuffle(out)
        return out

    def keyword_unique(self, thread: Thread) -> bool:
        """Whether no other plan thread's topic or keyword contains this thread's keyword."""
        keyword = topic_keyword(thread.topic).lower()
        return not any(
            keyword in t.topic.lower() or keyword in topic_keyword(t.topic).lower()
            for t in self.plan.threads if t.id != thread.id
        )

    # --- writing ----------------------------------------------------------------------------

    async def write(self, intent: Intent) -> str | None:
        return await write(
            intent, transcript_tail=self.tail, persona=self.plan.persona, llm=self.llm, model=self.model
        )

    def thread_intent(self, thread: Thread, kind: str, detail: str, **overrides) -> Intent:
        fields = {
            "kind": kind, "thread_ids": (thread.id,), "contacts": (thread.contact,) if thread.contact else (),
            "topic": thread.topic, "detail": detail, "mention": self.thread_mention(thread),
            "forbid": self.thread_forbid(thread),
        }
        return Intent(**{**fields, **overrides})


_Draft = tuple[str, list[str], dict]  # (message, gold_targets, extra Case fields)


async def _follow_up(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    for thread in p.shuffled(t for t in p.backed if (p.idle(t) is not None and p.idle(t) <= ACTIVE_DAYS)):
        if len(out) == want:
            break
        detail = p.rng.choice(FOLLOW_UP_DETAILS[_group(thread.domain)])
        text = await p.write(p.thread_intent(thread, "follow_up", detail))
        if text:
            gold, acceptable = p.gold(thread)
            out.append((text, gold, {"acceptable_ids": acceptable}))
    return out


async def _dormant_revival(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    for thread in p.shuffled(t for t in p.backed if (p.idle(t) is not None and p.idle(t) >= DORMANT_DAYS)):
        if len(out) == want:
            break
        text = await p.write(p.thread_intent(thread, "dormant_revival", p.rng.choice(REVIVAL_DETAILS)))
        if text:
            gold, acceptable = p.gold(thread)
            out.append((text, gold, {"acceptable_ids": acceptable}))
    return out


async def _vague_reference(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    last = p.snapshot.last_user_threads
    used: set[str] = set()
    if want and len(last) == 1 and last[0] in p.backed_ids:
        # (a) the thread the user just talked about, by pronoun only.
        thread = p.plan.thread(last[0])
        contact = p.contact(thread)
        names = (contact.first, contact.last) if contact else ()
        intent = Intent(
            kind="vague_reference", thread_ids=(thread.id,), detail=VAGUE_DETAIL,
            forbid=names + (topic_keyword(thread.topic),),
        )
        used.add(thread.id)
        text = await p.write(intent)
        if text:
            gold, acceptable = p.gold(thread)
            out.append((text, gold, {"acceptable_ids": acceptable}))
    # (b) a topic keyword no other thread shares, without the person.
    candidates = [t for t in p.backed if t.contact and t.id not in used and p.keyword_unique(t)]
    for thread in p.shuffled(candidates):
        if len(out) == want:
            break
        contact = p.contact(thread)
        intent = Intent(
            kind="vague_reference", thread_ids=(thread.id,), topic=thread.topic,
            detail=TOPIC_ONLY_DETAIL.format(topic=thread.topic), mention=(topic_keyword(thread.topic),),
            forbid=(contact.first, contact.last),
        )
        text = await p.write(intent)
        if text:
            gold, acceptable = p.gold(thread)
            out.append((text, gold, {"acceptable_ids": acceptable}))
    return out


async def _new_task(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    while p.reserve and len(out) < want:
        errand = p.reserve.pop(0)
        intent = Intent(
            kind="new_task", thread_ids=(), topic=errand,
            detail=NEW_ERRAND_DETAIL.format(topic=errand, detail=p.rng.choice(START_DETAILS["errand"])),
            mention=(topic_keyword(errand),),
        )
        text = await p.write(intent)
        if text:
            out.append((text, [NEW], {}))
    return out


async def _same_person_new_topic(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    had: dict[str, set[str]] = {}
    for t in p.plan.threads:
        if t.contact:
            had.setdefault(t.contact, set()).add(t.topic)
    people = sorted({t.contact for t in p.backed if t.contact})
    for full in p.shuffled(people):
        if len(out) == want:
            break
        contact = p.by_full[full]
        fresh = [topic for topic in _relation_pool(contact) if topic not in had.get(full, set())]
        if not fresh:
            continue
        topic = p.rng.choice(fresh)
        detail = p.rng.choice(START_DETAILS[_group(contact.relation)])
        intent = Intent(
            kind="same_person_new_topic", thread_ids=(), contacts=(full,), topic=topic,
            detail=NEW_TOPIC_DETAIL.format(topic=topic, detail=detail),
            mention=p.names(contact) + (topic_keyword(topic),), forbid=p.namesake_lasts(contact),
        )
        text = await p.write(intent)
        if text:
            out.append((text, [NEW], {}))
    return out


async def _multi_task(p: _Probes, want: int) -> list[_Draft]:
    """Even probes pair two agent-backed threads; odd ones pair a thread with a reserve errand."""
    out: list[_Draft] = []
    order = p.shuffled(p.backed)
    used: set[str] = set()
    for thread in order:
        if len(out) == want:
            break
        if thread.id in used:
            continue
        agents_a = set(p.snapshot.thread_agents[thread.id])
        detail_a = p.rng.choice(FOLLOW_UP_DETAILS[_group(thread.domain)])
        gold_a, acceptable_a = p.gold(thread)
        if len(out) % 2 == 0:
            other = next((
                t for t in order
                if t.id not in used and t.id != thread.id
                and (t.contact is None or t.contact != thread.contact)
                and not agents_a & set(p.snapshot.thread_agents[t.id])
            ), None)
            if other is None:
                continue
            detail_b = p.rng.choice(FOLLOW_UP_DETAILS[_group(other.domain)])
            gold_b, acceptable_b = p.gold(other)
            second, mention_b, forbid_b = other.id, p.thread_mention(other), p.thread_forbid(other)
            topic_b, contacts_b = other.topic, ((other.contact,) if other.contact else ())
        else:
            if not p.reserve:
                continue
            topic_b = p.reserve.pop(0)
            detail_b = "ask the assistant to " + topic_b + ", a new request"
            gold_b, acceptable_b = [NEW], []
            second, mention_b, forbid_b, contacts_b = None, (topic_keyword(topic_b),), (), ()
        intent = Intent(
            kind="multi_task", thread_ids=(thread.id,) + ((second,) if second else ()),
            contacts=((thread.contact,) if thread.contact else ()) + contacts_b,
            topic=f"{thread.topic}; {topic_b}",
            detail=MULTI_DETAIL.format(a=detail_a, a_topic=thread.topic, b=detail_b, b_topic=topic_b),
            mention=tuple(dict.fromkeys(p.thread_mention(thread) + mention_b)),
            forbid=tuple(dict.fromkeys(p.thread_forbid(thread) + forbid_b)),
        )
        text = await p.write(intent)
        if text:
            used.update({thread.id} | ({second} if second else set()))
            out.append((text, gold_a + gold_b, {"acceptable_ids": acceptable_a + acceptable_b}))
    return out


async def _same_name_people(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    for first in sorted(name for name, group in p.namesakes.items() if len(group) > 1):
        if len(out) == want:
            break
        group = p.namesakes[first]
        threads = {c.full: [t for t in p.backed if t.contact == c.full] for c in group}
        if not all(threads.values()):
            continue
        if any(c.last.lower() in p.context_text for c in group):
            continue
        keywords = [topic_keyword(t.topic) for c in group for t in p.plan.threads if t.contact == c.full]
        intent = Intent(
            kind="same_name_people", thread_ids=tuple(t.id for ts in threads.values() for t in ts),
            detail=p.rng.choice(SAME_NAME_DETAILS).format(first=first), mention=(first,),
            forbid=tuple(dict.fromkeys([c.last for c in group] + keywords)),
        )
        text = await p.write(intent)
        if text:
            out.append((text, [], {"clarify_ok": True}))
    return out


async def _cross_agent(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    backed_people = sorted({t.contact for t in p.backed if t.contact})
    for thread in p.shuffled(t for t in p.backed if t.contact):
        if len(out) == want:
            break
        pa = p.contact(thread)
        others = [f for f in backed_people if p.by_full[f].first != pa.first]
        if not others:
            others = sorted(c.full for c in p.plan.contacts if c.first != pa.first)
        if not others:
            continue
        pb = p.by_full[p.rng.choice(others)]
        detail = CROSS_DETAIL.format(a=pa.first, b=pb.first, verb=p.rng.choice(CROSS_VERBS), topic=thread.topic)
        intent = p.thread_intent(
            thread, "cross_agent", detail, contacts=(pa.full, pb.full),
            mention=tuple(dict.fromkeys(p.names(pa) + p.names(pb) + (topic_keyword(thread.topic),))),
            forbid=tuple(dict.fromkeys(p.namesake_lasts(pa) + p.namesake_lasts(pb))),
        )
        text = await p.write(intent)
        if text:
            gold, acceptable = p.gold(thread)
            out.append((text, gold, {"acceptable_ids": acceptable}))
    return out


async def _typo(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    firsts = {c.first.lower() for c in p.plan.contacts}
    candidates = [t for t in p.backed if t.contact and p.idle(t) is not None and p.idle(t) <= ACTIVE_DAYS]
    for thread in p.shuffled(candidates):
        if len(out) == want:
            break
        first = p.contact(thread).first
        if typo_name(first).lower() in firsts:
            continue  # the misspelling would name someone else
        detail = p.rng.choice(FOLLOW_UP_DETAILS[_group(thread.domain)])
        text = await p.write(p.thread_intent(thread, "typo", detail))
        if not text:
            continue
        corrupted = corrupt_name(text, first)
        if corrupted == text:
            continue
        gold, acceptable = p.gold(thread)
        out.append((corrupted, gold, {"acceptable_ids": acceptable}))
    return out


async def _agent_result(p: _Probes, want: int) -> list[_Draft]:
    out: list[_Draft] = []
    for thread in p.shuffled(p.backed):
        if len(out) == want:
            break
        contact = p.contact(thread)
        if contact:
            detail = p.rng.choice(RESULT_DETAILS).format(first=contact.first)
            mention = (contact.first,)
        else:
            detail = p.rng.choice(ERRAND_RESULT_DETAILS)
            mention = (topic_keyword(thread.topic),)
        intent = p.thread_intent(thread, "agent_result", detail, mention=mention, message_type="agent")
        body = await p.write(intent)
        if not body:
            continue
        gold, acceptable = p.gold(thread)
        name = p.agents[gold[0]].name
        out.append((
            f"[SUCCESS] {name}: {body}", gold,
            {"acceptable_ids": acceptable, "message_type": "agent", "dispatch_optional": True, "clarify_ok": True},
        ))
    return out


_BUILDERS = {
    "follow_up": _follow_up,
    "new_task": _new_task,
    "same_person_new_topic": _same_person_new_topic,
    "vague_reference": _vague_reference,
    "multi_task": _multi_task,
    "same_name_people": _same_name_people,
    "cross_agent": _cross_agent,
    "dormant_revival": _dormant_revival,
    "typo": _typo,
    "agent_result": _agent_result,
}


async def build_probes(
    snapshot: Snapshot, plan: LifePlan, *, llm: LLMClient, model: str, per_kind: int = 4, rng_seed: int
) -> list[Case]:
    """Up to per_kind probes of every routing kind for one snapshot; deterministic for a seed.

    Kinds run in ROUTING_KINDS order off one random.Random(rng_seed). Ids number the kept probes:
    f"lived-s{plan.seed}-{snapshot.size}-{kind}-{i:02d}".
    """
    probes = _Probes(snapshot, plan, llm, model, random.Random(rng_seed))
    cases: list[Case] = []
    for kind, builder in _BUILDERS.items():
        for i, (message, gold, extra) in enumerate(await builder(probes, per_kind)):
            cases.append(Case(
                id=f"lived-s{plan.seed}-{snapshot.size}-{kind}-{i:02d}", kind=kind, world_seed=plan.seed,
                message=message, gold_targets=gold, context=list(probes.context), **extra,
            ))
    return cases
