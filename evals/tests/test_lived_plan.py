import hashlib
from collections import Counter

from evals.worlds.lived.model import LIFE_DAYS
from evals.worlds.lived.plan import (
    BACKGROUND_SIZE, COLLIDING_FIRSTS, ERRANDS, SHARED_SURNAME, TOPIC_KEYWORDS, build_plan, topic_keyword, topic_words,
)

PLAN = build_plan(1)
USER_KINDS = ("start", "follow_up", "revival", "multi")


def _thread_events(plan):
    """thread id -> its events in day order."""
    events: dict[str, list] = {}
    for event in plan.events:
        for thread_id in event.intent.thread_ids:
            events.setdefault(thread_id, []).append(event)
    return events


def test_plan_is_deterministic():
    assert build_plan(1) == PLAN
    assert build_plan(2).events != PLAN.events


def test_shape():
    kinds = Counter(e.intent.kind for e in PLAN.events)
    assert len(PLAN.events) >= 600
    assert kinds["start"] >= 250 and kinds["start"] == len(PLAN.threads)
    assert 70 <= len(PLAN.contacts) <= 90
    with_contact = sum(t.contact is not None for t in PLAN.threads)
    assert 0.6 < with_contact / len(PLAN.threads) < 0.8
    assert {c.relation for c in PLAN.contacts} == {"work", "client", "recruiter", "family", "friend", "service"}
    users = sum(kinds[k] for k in USER_KINDS)
    assert 0.05 < kinds["multi"] / users < 0.11
    assert kinds["revival"] > 0 and kinds["result"] > 0


def test_events_are_sorted_with_sequential_ids():
    days = [e.day for e in PLAN.events]
    assert days == sorted(days)
    assert 0 <= days[0] and days[-1] <= LIFE_DAYS
    assert [e.id for e in PLAN.events] == [f"e{i:04d}" for i in range(len(PLAN.events))]


def test_every_continuation_comes_after_its_threads_start():
    threads = {t.id: t for t in PLAN.threads}
    for thread_id, events in _thread_events(PLAN).items():
        assert thread_id in threads
        assert events[0].intent.kind == "start", thread_id
        assert sum(e.intent.kind == "start" for e in events) == 1, thread_id
        assert all(e.day > events[0].day for e in events[1:]), thread_id


def test_counts_per_thread_and_revival_gap():
    for thread_id, events in _thread_events(PLAN).items():
        kinds = Counter(e.intent.kind for e in events if e.intent.thread_ids == (thread_id,))
        assert kinds["result"] <= 2 and kinds["revival"] <= 1, thread_id
        assert kinds["follow_up"] + sum(e.intent.thread_ids[0] == thread_id and e.intent.kind == "multi" for e in events) <= 4, thread_id
        for i, event in enumerate(events):
            if event.intent.kind == "revival":
                assert event.day - events[i - 1].day >= 21, thread_id


def test_contacts_never_repeat_a_topic():
    pairs = [(t.contact, t.topic) for t in PLAN.threads if t.contact]
    assert len(pairs) == len(set(pairs))
    errands = [t.topic for t in PLAN.threads if t.contact is None]
    assert len(errands) == len(set(errands))
    assert all(t.domain == "errand" for t in PLAN.threads if t.contact is None)


def test_collisions_exist():
    firsts = Counter(c.first for c in PLAN.contacts)
    for first in COLLIDING_FIRSTS:
        assert firsts[first] == 2, first
    assert [n for n, count in firsts.items() if count > 1] == list(COLLIDING_FIRSTS)
    shared = {c.relation for c in PLAN.contacts if c.last == SHARED_SURNAME}
    assert shared == {"work", "family"}
    # Probes need both namesakes to have threads.
    with_threads = {t.contact for t in PLAN.threads}
    assert all(c.full in with_threads for c in PLAN.contacts if c.first in COLLIDING_FIRSTS)


def test_names_are_not_hidden_inside_other_words():
    """A first name must not match inside another name or a topic, or mention checks pass by accident."""
    texts = [c.full.lower() for c in PLAN.contacts] + [t.lower() for t in TOPIC_KEYWORDS]
    for contact in PLAN.contacts:
        first = contact.first.lower()
        clashes = [t for t in texts if first in t and not t.startswith(f"{first} ")]
        assert not clashes, (contact.first, clashes)


def test_mentions_make_the_thread_unambiguous():
    contacts = {c.full: c for c in PLAN.contacts}
    threads = {t.id: t for t in PLAN.threads}
    per_contact = Counter(t.contact for t in PLAN.threads if t.contact)
    for topic, keyword in TOPIC_KEYWORDS.items():
        assert keyword.lower() in topic.lower(), topic
    for event in PLAN.events:
        intent = event.intent
        assert intent.message_type == ("agent" if intent.kind == "result" else "user"), event.id
        for thread_id in intent.thread_ids:
            thread = threads[thread_id]
            keyword = topic_keyword(thread.topic)
            if thread.contact is None:
                assert keyword in intent.mention, event.id
                continue
            contact = contacts[thread.contact]
            assert contact.first in intent.mention, event.id
            if intent.kind == "result":
                continue
            if contact.first in COLLIDING_FIRSTS:
                assert contact.last in intent.mention, event.id
                assert intent.forbid, event.id
            if intent.kind in ("start", "revival") or per_contact[thread.contact] > 1:
                assert keyword in intent.mention, event.id
        assert intent.detail, event.id


def test_multi_events_combine_two_threads():
    multis = [e for e in PLAN.events if e.intent.kind == "multi"]
    assert multis
    threads = {t.id: t for t in PLAN.threads}
    for event in multis:
        a, b = (threads[i] for i in event.intent.thread_ids)
        assert a.id != b.id
        assert a.contact is None or a.contact != b.contact
        assert event.intent.contacts == tuple(t.contact for t in (a, b) if t.contact)


def test_background_is_unique_and_disjoint_from_threads():
    background = PLAN.background
    assert len(background) == BACKGROUND_SIZE
    assert len({b.topic for b in background}) == BACKGROUND_SIZE
    thread_words = set()
    for topic, keyword in TOPIC_KEYWORDS.items():
        thread_words |= topic_words(topic) | topic_words(keyword)
    names = {c.first.lower() for c in PLAN.contacts} | {c.last.lower() for c in PLAN.contacts}
    for intent in background:
        assert (intent.kind, intent.thread_ids, intent.contacts) == ("start", (), ())
        words = topic_words(intent.topic) | {w for m in intent.mention for w in topic_words(m)}
        assert not words & thread_words, intent.topic
        assert not words & names, intent.topic
        assert all(m in intent.topic for m in intent.mention)


def test_unused_errands_remain_as_a_reserve():
    used = {t.topic for t in PLAN.threads if t.contact is None}
    assert len({topic for topic, _ in ERRANDS} - used) >= 10


def test_persona_matches_the_contacts():
    contacts = {c.full for c in PLAN.contacts}
    assert "Jamie Reyes" in contacts and "Elena Reyes" in contacts
    for word in ("Jamie", "Elena", "Mateo", "Doyle", "Northwind Labs", "Portland"):
        assert word in PLAN.persona


# Pins the exact plan. Any change to plan.py changes it: update deliberately after checking the plan.
GOLDEN_SHA256 = "9227f764f83144a2041271cfe4fb4273b0b586d493d8fe2896196a699adefc2a"


def test_plan_matches_the_golden_hash():
    text = "\n".join(f"{e.id}|{e.day}|{e.intent!r}" for e in PLAN.events) + "\n" + "\n".join(
        repr(b) for b in PLAN.background
    )
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == GOLDEN_SHA256
