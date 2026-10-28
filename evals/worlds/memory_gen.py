"""Seeded execution-agent histories with planted facts, for the memory eval (L2 memory).

No LLM is involved. Each agent owns one matter for the user, and its log follows production order, four entries
per run: agent_request, agent_action ("Calling <tool> with: <args[:200]>"), tool_response ("<tool>: <result[:500]>")
and agent_response, exactly as `ExecutionAgent.record_tool_execution` and the log store write them.

Six facts per agent are planted in the first third of the runs, so at the production 4,000-token budget they fall
in the compacted prefix, and none of them reappears in the raw tail:
- thread_id: the Gmail thread id of the matter's main thread;
- draft_id: the draft awaiting the user's confirmation;
- email: a third party's address (not guessable from the name);
- preference: how the user wants this matter handled;
- decision: an agreed amount, first stated and then superseded by an update (gold is the update);
- commitment: what the user promised, by which date.
Distractors (other threads, drafts, addresses, amounts and dates) appear throughout, prefix and tail.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from server.services.agents.memory import Entry

FACT_KINDS = ("thread_id", "draft_id", "email", "preference", "decision", "commitment")
USER = "Jordan Reyes"
USER_EMAIL = "jordan.reyes@gmail.com"
START = datetime(2026, 9, 1, 9, 0)
MIN_RUNS = 24  # the seven planting runs must fit in the first third, after the kickoff run


@dataclass(frozen=True)
class Matter:
    slug: str
    name: str
    purpose: str
    first: str
    last: str
    company: str
    domain: str
    subject: str
    item: str  # what the decided amount is for
    amount_range: tuple[int, int]
    promise: str  # what the user committed to do (the commitment)
    role: str  # the third party whose address is planted


MATTERS = (
    Matter("lease-gary", "Lease renewal with Gary", "Negotiate the apartment lease renewal with Gary Whitman, the landlord",
           "Gary", "Whitman", "Whitman Properties", "whitmanproperties.com", "Lease renewal for unit 4B",
           "monthly rent on the renewed lease", (2800, 4800), "send the signed lease", "office manager"),
    Matter("globex-renewal", "Globex contract renewal", "Close the 2027 contract renewal with Ben Carter at Globex",
           "Ben", "Carter", "Globex", "globex.com", "Globex renewal: 2027 terms",
           "annual license fee", (18000, 60000), "return the countersigned order form", "procurement lead"),
    Matter("tax-paul", "Tax return with Paul", "Get the 2025 tax return filed with Paul Fenwick, the accountant",
           "Paul", "Fenwick", "Fenwick Tax", "fenwicktax.com", "2025 return: missing documents",
           "preparation fee", (600, 1800), "upload the brokerage 1099s", "bookkeeper"),
    Matter("bathroom-remodel", "Bathroom remodel", "Agree scope, price and schedule for the bathroom remodel with Carl Jensen",
           "Carl", "Jensen", "Jensen Builders", "jensenbuilders.com", "Bathroom remodel quote",
           "fixed price for the remodel", (9000, 25000), "pick the tile samples", "site supervisor"),
    Matter("director-offer", "Director role offer", "Negotiate the Director of Product offer with Anjali Rao at Talentbridge",
           "Anjali", "Rao", "Talentbridge", "talentbridge.io", "Director of Product offer",
           "signing bonus", (10000, 40000), "give a final answer on the offer", "HR coordinator"),
    Matter("crown-dental", "Crown treatment", "Sort out the crown treatment and its cost with Brightsmile Dental",
           "Laura", "Kemp", "Brightsmile Dental", "brightsmiledental.com", "Crown fitting estimate",
           "out-of-pocket cost for the crown", (700, 2200), "send the insurance pre-authorization form",
           "billing coordinator"),
    Matter("brake-repair", "Brake repair", "Get the car's brakes repaired at Eastside Auto",
           "Steve", "Kowalski", "Eastside Auto", "eastsideauto.net", "Brake job estimate",
           "price for the brake job", (400, 1600), "drop off the car", "service writer"),
    Matter("initech-sow", "Initech statement of work", "Agree the phase 2 statement of work with Hugo Lambert at Initech",
           "Hugo", "Lambert", "Initech", "initech.com", "SOW for phase 2",
           "phase 2 budget", (25000, 90000), "send the revised statement of work", "project coordinator"),
    Matter("biscuit-surgery", "Biscuit's surgery", "Arrange Biscuit's dental surgery at Paws Clinic",
           "Monica", "Reed", "Paws Clinic", "pawsclinic.vet", "Biscuit dental surgery",
           "surgery estimate", (900, 3000), "confirm the pre-op bloodwork appointment", "practice manager"),
    Matter("tahoe-cabin", "Tahoe cabin weekend", "Plan the Tahoe cabin weekend with Rowan Brooks and friends",
           "Rowan", "Brooks", "Brooks family", "brooksmail.com", "Tahoe cabin weekend",
           "user's share of the cabin rental", (300, 1200), "Venmo the deposit", "rental agent"),
    Matter("water-heater", "Water heater replacement", "Replace the water heater with Tran Plumbing",
           "Henry", "Tran", "Tran Plumbing", "tranplumbing.com", "Water heater replacement quote",
           "installed price for the water heater", (1200, 3800), "clear access to the garage", "dispatcher"),
    Matter("home-policy", "Home policy renewal", "Renew the home insurance policy with Amy Fox at Cascade Insurance",
           "Amy", "Fox", "Cascade Insurance", "cascadeins.com", "Home policy renewal 2027",
           "annual premium", (1100, 3500), "send photos of the new roof", "underwriting assistant"),
)

THIRD_PARTIES = (
    ("Dana", "Okoye"), ("Marisol", "Quint"), ("Theo", "Vasquez"), ("Ingrid", "Halvorsen"), ("Raj", "Mehta"),
    ("Colette", "Dubois"), ("Femi", "Adeyemi"), ("Yara", "Haddad"), ("Bruno", "Sauer"), ("Keiko", "Tanaka"),
    ("Lars", "Nyberg"), ("Noemi", "Ferro"), ("Arjun", "Pillai"), ("Wren", "Calloway"),
)
OTHER_PEOPLE = (
    ("Priya", "Shah", "northwindlabs.com"), ("Marcus", "Lee", "northwindlabs.com"), ("Elena", "Reyes", "gmail.com"),
    ("Kwame", "Mensah", "outlook.com"), ("Freya", "Lindqvist", "gmail.com"), ("Omar", "Haddad", "northwindlabs.com"),
    ("Sara", "Lund", "fastmail.com"), ("Derek", "Shaw", "yahoo.com"), ("Grace", "Liu", "northwindlabs.com"),
)
SIDE_SUBJECTS = (
    "Quick question about scheduling", "Your receipt from {company}", "{company} newsletter: October updates",
    "Re: paperwork follow-up", "Updated contact details", "Holiday hours at {company}", "Survey: how did we do?",
)
SIDE_FEES = ("late fee", "admin fee", "rush fee", "cancellation fee", "document fee", "parking charge", "deposit refund")
SIDE_MEETINGS = ("a quick call", "the walkthrough", "a check-in", "the paperwork review")
SIDE_TIMES = ("1pm", "2:30pm", "3pm", "4pm", "4:30pm")
FILLER_BODIES = (
    "Thanks for the update. I have passed this along to the rest of the team and will come back to you once I hear "
    "more from them. In the meantime there is nothing you need to do on your side.",
    "Just following up on the earlier note so it doesn't get lost. No action is needed on your side for now; we will "
    "reach out again if anything changes or if we need a signature.",
    "Attached is the information you asked for last week. Let me know if anything is missing or unclear and I will "
    "send a corrected copy the same day.",
    "We are running a little behind this week because of staff holidays but expect to catch up shortly. Apologies for "
    "the delay and thank you for your patience.",
    "Confirming receipt of your message. Our team will review it and reply within two business days. Please keep this "
    "email for your records.",
)
# Appended to routine responses: production responses restate where things stand.
STATUS_NOTES = (
    "Nothing else changed on the matter; the open items are the same as in the last update and no reply is overdue.",
    "No other messages need attention right now. I checked the inbox for related mail and found nothing new to act on.",
    "This was routine housekeeping; the main thread is quiet and nothing is waiting on the user from this step.",
    "Logged for the record. No follow-up is scheduled from this step and nothing on the main matter moved today.",
)
# (statement, question, gold answers, needle): the preference planted for a matter.
PREFERENCES = (
    ("keep every reply to {first} formal: full names, no exclamation marks",
     "What tone did the user ask for in replies to {first}?", ("formal",), "formal"),
    ("sign every email in this matter with 'Warm regards, Jordan'",
     "What sign-off did the user ask for on emails in this matter?", ("warm regards",), "Warm regards"),
    ("never propose a call or meeting before 10am",
     "What is the earliest time the user is willing to take a call or meeting for this matter?",
     ("10am", "10:00", "10 o'clock"), "10am"),
    ("always cc Jamie on anything that mentions money",
     "Who did the user ask to be cc'd on anything that mentions money?", ("jamie",), "Jamie"),
    ("write to {first} in Spanish from now on",
     "In what language did the user ask for emails to {first} to be written?", ("spanish",), "Spanish"),
    ("keep every email in this matter under five sentences",
     "What length limit did the user set for emails in this matter?", ("five sentences", "5 sentences"), "five sentences"),
)
MONTH_ABBREVIATIONS = {"September": ("Sep", "Sept"), "October": ("Oct",), "November": ("Nov",)}


@dataclass(frozen=True)
class Fact:
    kind: str
    question: str
    gold: tuple[str, ...]  # any of these, normalized, in the answer counts as recalled
    needle: str  # the raw planted text; it never appears after the planting runs
    run: int  # the last run that mentions the fact
    distractors: tuple[str, ...] = ()  # wrong values a stale or confused memory would give


@dataclass(frozen=True)
class AgentHistory:
    agent_id: str
    name: str
    purpose: str
    entries: tuple[Entry, ...]
    facts: tuple[Fact, ...]


@dataclass(frozen=True)
class _Run:
    request: str
    tool: str
    args: dict[str, Any]
    result: Any
    response: str


def _money(amount: int) -> str:
    return f"${amount:,}"


def _date(day: datetime) -> str:
    return f"{day:%A}, {day:%B} {day.day}"


def _date_gold(day: datetime) -> tuple[str, ...]:
    month = f"{day:%B}"
    names = (month, *MONTH_ABBREVIATIONS.get(month, (month[:3],)))
    return (*(f"{name} {day.day}" for name in names), f"{day.day} {month}", f"{day:%Y-%m-%d}")


class _Ids:
    """Unique Gmail-shaped ids: 16-hex message/thread ids and `r-` + 19-digit draft ids."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def _fresh(self, make) -> str:
        while True:
            value = make()
            if value not in self.used:
                self.used.add(value)
                return value

    def hex(self) -> str:
        return self._fresh(lambda: "1" + "".join(self.rng.choice("0123456789abcdef") for _ in range(15)))

    def draft(self) -> str:
        return self._fresh(lambda: f"r-{self.rng.randint(10**18, 10**19 - 1)}")


def _record(run: _Run, when: datetime) -> list[Entry]:
    """The four entries production writes for one run, with its truncation."""
    stamps = [(when + timedelta(seconds=offset)).strftime("%Y-%m-%d %H:%M:%S") for offset in (0, 4, 9, 15)]
    return [
        Entry("agent_request", run.request, stamps[0]),
        Entry("agent_action", f"Calling {run.tool} with: {json.dumps(run.args, default=str)[:200]}", stamps[1]),
        Entry("tool_response", f"{run.tool}: {json.dumps(run.result, default=str)[:500]}", stamps[2]),
        Entry("agent_response", run.response, stamps[3]),
    ]


def _search_hit(msg: str, thread: str, query: str, subject: str, sender: str, when: datetime, text: str) -> dict:
    return {
        "id": msg, "thread_id": thread, "query": query, "subject": subject, "sender": sender, "recipient": USER_EMAIL,
        "timestamp": when.strftime("%Y-%m-%dT%H:%M:%S-07:00"), "label_ids": ["INBOX", "UNREAD"],
        "clean_text": text, "has_attachments": False, "attachment_count": 0, "attachment_filenames": [],
    }


def _sent(msg: str, thread: str) -> dict:
    return {"data": {"id": msg, "threadId": thread, "labelIds": ["SENT"]}, "error": None, "successful": True}


def _person(name: str, email: str, ids: _Ids) -> dict:
    """A gmail_search_people hit (the resource name is seeded)."""
    return {
        "data": {"results": [{"person": {"resourceName": f"people/c{ids.rng.randint(10**11, 10**12 - 1)}",
                                          "names": [{"displayName": name}], "emailAddresses": [{"value": email}]}}]},
        "error": None, "successful": True,
    }


def _build_agent(seed: int, index: int, runs: int) -> AgentHistory:
    rng = random.Random(f"memory-{seed}-{index}")
    matter = MATTERS[index % len(MATTERS)]
    cycle = index // len(MATTERS)
    agent_id = matter.slug if cycle == 0 else f"{matter.slug}-{cycle + 1}"
    ids = _Ids(rng)
    first = matter.first
    cp = f"{matter.first} {matter.last}"
    cp_email = f"{matter.first.lower()}@{matter.domain}"
    thread, draft = ids.hex(), ids.draft()
    tp_first, tp_last = rng.choice(THIRD_PARTIES)
    third = f"{tp_first} {tp_last}"
    third_email = f"{tp_first[0].lower()}{tp_last.lower()}{rng.randint(10, 99)}@{matter.domain}"

    low, high = matter.amount_range
    step = 50 if high < 10000 else 500
    agreed_first = rng.randrange(low, high, step)
    while True:
        agreed = rng.randrange(low, high, step)
        a, b = str(agreed_first), str(agreed)
        if a not in b and b not in a:
            break

    statement, pref_question, pref_gold, pref_needle = rng.choice(PREFERENCES)
    statement, pref_question = statement.format(first=first), pref_question.format(first=first)

    # Planting runs: the thread is found first; the superseded amount precedes its update.
    window = list(range(1, runs // 3 + 1))
    positions = sorted(rng.sample(window, 7))
    kinds = ["draft", "email", "preference", "decision_first", "decision_update", "commitment"]
    rng.shuffle(kinds)
    plan = {"thread": positions[0], **dict(zip(kinds, positions[1:], strict=True))}
    if plan["decision_first"] > plan["decision_update"]:
        plan["decision_first"], plan["decision_update"] = plan["decision_update"], plan["decision_first"]
    by_run = {run: kind for kind, run in plan.items()}

    when = [START + timedelta(days=run, hours=rng.randint(0, 9), minutes=rng.randint(0, 59)) for run in range(runs)]
    # Day of month >= 10, so a gold "September 1" can never be a prefix of a wrong "September 14".
    promised_on = when[plan["commitment"]]
    options = [promised_on + timedelta(days=d) for d in range(4, 14) if (promised_on + timedelta(days=d)).day >= 10]
    deadline = rng.choice(options).replace(hour=0, minute=0)
    deadline_text = _date(deadline)

    # Distractor material: side threads, other people, other drafts, fees and dates.
    side_threads = [(ids.hex(), subject.format(company=matter.company))
                    for subject in rng.sample(SIDE_SUBJECTS, 3)]
    others = rng.sample(OTHER_PEOPLE, 4)
    pending_other_drafts: list[tuple[str, str]] = []

    def side_amount() -> int:
        while True:
            value = rng.randrange(25, 950, 5)
            text = str(value)
            if all(text not in str(x) and str(x) not in text for x in (agreed, agreed_first)):
                return value

    def side_date(run: int) -> datetime:
        while True:
            day = when[run] + timedelta(days=rng.randint(1, 12))
            if day.date() != deadline.date():
                return day

    def planted(kind: str, run: int) -> _Run:
        if kind == "thread":
            query = f"from:{cp_email} {matter.subject.split(':')[0].lower()}"
            text = f"Hi Jordan, following up on {matter.subject.lower()}. Can we settle the open points this week?"
            hit = _search_hit(ids.hex(), thread, query, matter.subject, f"{cp} <{cp_email}>",
                              when[run] - timedelta(hours=5), text)
            return _Run(
                f"Find the email thread with {cp} about '{matter.subject}' and tell me where it stands.",
                "task_email_search", {"search_query": query}, [hit],
                f"Found the '{matter.subject}' thread with {cp} (thread_id {thread}). {first}'s latest message asks to "
                f"settle the open points this week; nothing has been agreed yet.",
            )
        if kind == "draft":
            body = f"Hi {first}, thanks for your patience. I'd like to move forward on the open points; details below."
            return _Run(
                f"Draft a reply to {first} on the '{matter.subject}' thread saying we want to move forward. Don't send it "
                f"until the user confirms.",
                "gmail_create_draft",
                {"recipient_email": cp_email, "subject": f"Re: {matter.subject}", "body": body, "thread_id": thread},
                {"data": {"response_data": {"id": draft, "message": {"id": ids.hex(), "threadId": thread,
                                                                     "labelIds": ["DRAFT"]}}},
                 "error": None, "successful": True},
                f"Created draft {draft} to {cp_email}, subject 'Re: {matter.subject}', body: \"{body}\" It is waiting "
                f"for the user's confirmation before sending.",
            )
        if kind == "email":
            return _Run(
                f"{first} says paperwork for this should go to their {matter.role}, {third}. Find their email address "
                f"and keep it on file.",
                "gmail_search_people", {"query": third, "other_contacts": True},
                _person(third, third_email, ids),
                f"{third} ({first}'s {matter.role}) is at {third_email}. Saved for future paperwork.",
            )
        if kind == "preference":
            return _Run(
                f"Note from the user for this matter: {statement}. Check whether any pending drafts need changing.",
                "gmail_list_drafts", {"max_results": 10, "verbose": False},
                {"data": {"drafts": []}, "error": None, "successful": True},
                f"Noted the user's preference: {statement}. No pending drafts needed changes.",
            )
        if kind == "decision_first":
            return _Run(
                f"{first} replied: we agreed on {_money(agreed_first)} for the {matter.item}. Reply on the thread to "
                f"confirm it.",
                "gmail_reply_to_thread",
                {"thread_id": thread, "recipient_email": cp_email,
                 "message_body": f"Hi {first}, confirming {_money(agreed_first)} for the {matter.item}."},
                _sent(ids.hex(), thread),
                f"Replied on the '{matter.subject}' thread confirming {_money(agreed_first)} for the {matter.item}.",
            )
        if kind == "decision_update":
            return _Run(
                f"Updated: {first} came back and the final agreed {matter.item} is {_money(agreed)}, replacing the "
                f"earlier figure. Acknowledge it on the thread.",
                "gmail_reply_to_thread",
                {"thread_id": thread, "recipient_email": cp_email,
                 "message_body": f"Hi {first}, thanks. Acknowledging the updated figure of {_money(agreed)}."},
                _sent(ids.hex(), thread),
                f"Acknowledged the update on the thread: the {matter.item} is now {_money(agreed)}, superseding the "
                f"earlier figure.",
            )
        if kind == "commitment":
            return _Run(
                f"The user promised {first} they'll {matter.promise} by {deadline_text}. Let {first} know on the thread.",
                "gmail_reply_to_thread",
                {"thread_id": thread, "recipient_email": cp_email,
                 "message_body": f"Hi {first}, Jordan will {matter.promise} by {deadline_text}."},
                _sent(ids.hex(), thread),
                f"Told {first} on the thread that the user will {matter.promise} by {deadline_text}.",
            )
        raise ValueError(kind)

    def kickoff(run: int) -> _Run:
        return _Run(
            f"New task: {matter.purpose}. Start by finding {first}'s contact details.",
            "gmail_search_people", {"query": cp, "other_contacts": True},
            _person(cp, cp_email, ids),
            f"{cp} ({matter.company}) is at {cp_email}. Ready for next steps on: {matter.purpose}.",
        )

    def filler(run: int) -> _Run:
        choices = ["replies", "side_reply", "other_draft", "lookup", "forward_fee", "schedule"]
        if pending_other_drafts:
            choices.append("delete_draft")
        choice = rng.choice(choices)
        side_thread, side_subject = rng.choice(side_threads)
        other_first, other_last, other_domain = rng.choice(others)
        other = f"{other_first} {other_last}"
        other_email = f"{other_first.lower()}.{other_last.lower()}@{other_domain}"
        body = rng.choice(FILLER_BODIES)
        if choice == "replies":
            query = f"from:{cp_email} newer_than:3d"
            hit = _search_hit(ids.hex(), side_thread, query, side_subject, f"{matter.company} <noreply@{matter.domain}>",
                              when[run] - timedelta(hours=3), body)
            return _Run(
                f"Check whether anything new came in from {first} or {matter.company}.",
                "task_email_search", {"search_query": query}, [hit],
                f"Nothing new from {first} personally. One automated message on '{side_subject}' (thread "
                f"{side_thread}): \"{body}\" No action needed.",
            )
        if choice == "side_reply":
            reply = f"Thanks, received. {rng.choice(FILLER_BODIES)}"
            return _Run(
                f"Reply briefly on the '{side_subject}' thread to say thanks, received.",
                "gmail_reply_to_thread",
                {"thread_id": side_thread, "recipient_email": f"noreply@{matter.domain}", "message_body": reply},
                _sent(ids.hex(), side_thread),
                f"Replied on '{side_subject}' (thread {side_thread}) with a short acknowledgement.",
            )
        if choice == "other_draft":
            other_draft = ids.draft()
            pending_other_drafts.append((other_draft, other))
            note = f"Hi {other_first}, quick update: things are moving along. {body}"
            return _Run(
                f"Draft a short note to {other} with a quick status update. Leave it as a draft.",
                "gmail_create_draft", {"recipient_email": other_email, "subject": "Quick update", "body": note},
                {"data": {"response_data": {"id": other_draft, "message": {"id": ids.hex(), "threadId": ids.hex(),
                                                                           "labelIds": ["DRAFT"]}}},
                 "error": None, "successful": True},
                f"Created draft {other_draft} to {other_email} (subject 'Quick update'). Left unsent.",
            )
        if choice == "delete_draft":
            other_draft, owner = pending_other_drafts.pop(0)
            return _Run(
                f"The note to {owner} isn't needed any more; delete draft {other_draft}.",
                "gmail_delete_draft", {"draft_id": other_draft},
                {"data": {}, "error": None, "successful": True},
                f"Deleted draft {other_draft} (the note to {owner}).",
            )
        if choice == "lookup":
            return _Run(
                f"Find the email address for {other}; the user may loop them in later.",
                "gmail_search_people", {"query": other, "other_contacts": True},
                _person(other, other_email, ids),
                f"{other} is at {other_email}.",
            )
        if choice == "forward_fee":
            fee, amount = rng.choice(SIDE_FEES), side_amount()
            return _Run(
                f"Forward {matter.company}'s note about the {fee} ({_money(amount)}) to {other} for their records.",
                "gmail_forward_email",
                {"message_id": ids.hex(), "recipient_email": other_email,
                 "additional_text": f"FYI: {matter.company} {fee}, {_money(amount)}."},
                _sent(ids.hex(), side_thread),
                f"Forwarded the {fee} note ({_money(amount)}) to {other_email}.",
            )
        day, meeting, time = side_date(run), rng.choice(SIDE_MEETINGS), rng.choice(SIDE_TIMES)
        return _Run(
            f"Let {matter.company} know the user can do {meeting} on {_date(day)} at {time}.",
            "gmail_reply_to_thread",
            {"thread_id": side_thread, "recipient_email": f"noreply@{matter.domain}",
             "message_body": f"Hello, Jordan is available for {meeting} on {_date(day)} at {time}."},
            _sent(ids.hex(), side_thread),
            f"Offered {meeting} on {_date(day)} at {time} on the '{side_subject}' thread.",
        )

    entries: list[Entry] = []
    for run in range(runs):
        if run == 0:
            step = kickoff(run)
        elif run in by_run:
            step = planted(by_run[run], run)
        else:
            step = filler(run)
            step = _Run(step.request, step.tool, step.args, step.result, f"{step.response} {rng.choice(STATUS_NOTES)}")
        entries.extend(_record(step, when[run]))

    facts = (
        Fact("thread_id", f"What is the Gmail thread id of the '{matter.subject}' thread with {first}?",
             (thread,), thread, max(plan.values()), tuple(t for t, _ in side_threads)),
        Fact("draft_id", f"What is the id of the draft reply to {first} on the '{matter.subject}' thread that is "
                         f"waiting for the user's confirmation?", (draft,), draft, plan["draft"]),
        Fact("email", f"What is the email address of {third}, {first}'s {matter.role}?",
             (third_email,), third_email, plan["email"], (cp_email,)),
        Fact("preference", pref_question, pref_gold, pref_needle, plan["preference"]),
        Fact("decision", f"What is the currently agreed {matter.item}?", (_money(agreed),), _money(agreed),
             plan["decision_update"], (_money(agreed_first),)),
        Fact("commitment", f"By what date did the user promise {first} they would {matter.promise}?",
             _date_gold(deadline), deadline_text, plan["commitment"]),
    )
    name = matter.name if cycle == 0 else f"{matter.name} ({cycle + 1})"
    return AgentHistory(agent_id, name, matter.purpose, tuple(entries), facts)


def build_histories(seed: int = 1, n_agents: int = 12, runs: int = 40) -> list[AgentHistory]:
    """`n_agents` seeded agent histories of `runs` runs each (agents past the twelfth reuse matters, renamed)."""
    if runs < MIN_RUNS:
        raise ValueError(f"runs must be at least {MIN_RUNS} so the planted facts fit in the first third")
    return [_build_agent(seed, index, runs) for index in range(n_agents)]


__all__ = ["FACT_KINDS", "AgentHistory", "Fact", "build_histories"]
