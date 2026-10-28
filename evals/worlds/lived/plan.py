"""Seeded life plan for the lived-in world: persona, contacts, threads, events and background errands.

No LLM is involved. Every event knows its threads (the gold label), and its Intent lists what the
written message must mention, so growth never has to guess which thread a message continues: a
contact's first name (plus the last name for a shared first name) and the topic keyword where the
person alone is not enough.
"""

from __future__ import annotations

import random
import re

from .model import LIFE_DAYS, Contact, Event, Intent, LifePlan, Thread

PERSONA = (
    "Jordan Reyes, 34, senior product manager at Northwind Labs in Portland, Oregon. Married to Jamie, "
    "with a six-year-old daughter, Mia. Sister Elena and brother Mateo; cousins on the Doyle side (Mom's "
    "family) and the Alvarez side. Juggles a team, clients, recruiters, friends and family logistics "
    "through an AI assistant, mostly from their phone."
)

WORK_COMPANY = "Northwind Labs"
COLLIDING_FIRSTS = ("Alice", "Ben")
SHARED_SURNAME = "Doyle"  # colleague Kevin Doyle and cousin Nora Doyle

WORK = (
    ("Alice", "Chen"), ("Kevin", "Doyle"), ("Priya", "Shah"), ("Marcus", "Lee"), ("Hana", "Sato"),
    ("Omar", "Haddad"), ("Greta", "Fischer"), ("Ravi", "Iyer"), ("Chloe", "Martin"), ("Kenji", "Mori"),
    ("Zoe", "Turner"), ("Dmitri", "Petrov"), ("Maya", "Cohen"), ("Tobias", "Becker"), ("Sofia", "Rossi"),
    ("Dana", "Kim"), ("Felix", "Wagner"), ("Grace", "Liu"),
)
CLIENTS = (
    ("Ben", "Carter", "Globex"), ("Hugo", "Lambert", "Initech"), ("Iris", "Novak", "Hooli"),
    ("Jonas", "Berg", "Umbrella"), ("Kira", "Nair", "Stark"), ("Leo", "Costa", "Wayne"),
    ("Petra", "Horvat", "Soylent"), ("Noah", "Fraser", "Acme"), ("Olga", "Ivanova", "Globex"),
    ("Pablo", "Ortega", "Initech"), ("Quinn", "Harper", "Hooli"), ("Seth", "Cole", "Umbrella"),
    ("Tara", "Singh", "Stark"), ("Yusuf", "Aydin", "Wayne"),
)
RECRUITERS = (
    ("Alice", "Brennan", "Hirewell"), ("Anjali", "Rao", "Talentbridge"), ("Gavin", "Hale", "Hirewell"),
    ("Wendy", "Zhou", "Pinnacle Search"), ("Leila", "Rahman", "Talentbridge"), ("Simon", "Abbott", "Pinnacle Search"),
)
FAMILY = (
    ("Jamie", "Reyes"), ("Elena", "Reyes"), ("Mateo", "Reyes"), ("Nora", "Doyle"), ("Teresa", "Alvarez"),
    ("Luis", "Alvarez"), ("Diego", "Reyes"), ("Isabel", "Moreno"), ("Rafael", "Moreno"), ("Helen", "Walsh"),
    ("George", "Walsh"), ("Gloria", "Santos"), ("Pedro", "Alvarez"), ("Lucia", "Reyes"),
)
FRIENDS = (
    ("Ben", "Okafor"), ("Rowan", "Brooks"), ("Beatriz", "Duarte"), ("Kwame", "Mensah"), ("Jake", "Morrison"),
    ("Freya", "Lindqvist"), ("Owen", "Ellis"), ("Fatima", "Malik"), ("Liam", "Byrne"), ("Callum", "Dunn"),
    ("Megan", "Hart"), ("Nikhil", "Bose"), ("Sara", "Lund"), ("Ethan", "Ward"), ("Julia", "Weber"),
    ("Marco", "Bianchi"), ("Anya", "Volkova"), ("Derek", "Shaw"),
)
# (first, last, company, role): the role picks the service's own topic pool.
SERVICES = (
    ("Gary", "Whitman", "Whitman Properties", "landlord"), ("Laura", "Kemp", "Brightsmile Dental", "dentist"),
    ("Paul", "Fenwick", "Fenwick Tax", "accountant"), ("Rita", "Gomez", "Sparkle Home", "cleaner"),
    ("Steve", "Kowalski", "Eastside Auto", "mechanic"), ("Monica", "Reed", "Paws Clinic", "vet"),
    ("Henry", "Tran", "Tran Plumbing", "plumber"), ("Harriet", "Lawson", "Lawson Music", "piano teacher"),
    ("Carl", "Jensen", "Jensen Builders", "contractor"), ("Amy", "Fox", "Cascade Insurance", "insurance agent"),
)

# (topic, keyword): the keyword is the distinctive word a message must use to name the topic.
TOPICS = {
    "work": (
        ("Q3 roadmap review", "roadmap"), ("sprint planning", "sprint"), ("launch checklist", "checklist"),
        ("hiring loop feedback", "hiring"), ("performance review", "performance"), ("team offsite", "offsite"),
        ("budget forecast", "forecast"), ("design spec review", "spec"), ("on-call rotation", "on-call"),
        ("landing page copy", "landing page"), ("API migration", "migration"), ("quarterly OKRs", "OKRs"),
        ("vendor security review", "security"), ("demo day slides", "demo"), ("analytics dashboard", "dashboard"),
        ("incident postmortem", "postmortem"), ("conference talk proposal", "conference"),
    ),
    "client": (
        ("contract renewal", "renewal"), ("invoice dispute", "invoice"), ("statement of work", "statement of work"),
        ("pilot kickoff", "pilot"), ("discount request", "discount"), ("integration timeline", "integration"),
        ("support escalation", "escalation"), ("quarterly business review", "business review"), ("case study approval", "case study"),
        ("procurement paperwork", "procurement"), ("onboarding call", "onboarding"), ("NDA signature", "NDA"),
        ("data export issue", "export"),
    ),
    "recruiter": (
        ("offer negotiation", "negotiation"), ("director of product role", "director"), ("phone screen", "phone screen"),
        ("onsite interview", "onsite"), ("reference check", "reference"), ("salary expectations", "salary"),
        ("startup opportunity", "startup"), ("relocation package", "relocation"), ("signing bonus", "bonus"),
    ),
    "family": (
        ("Thanksgiving plans", "Thanksgiving"), ("Mom's birthday", "birthday"), ("Christmas gifts", "Christmas"),
        ("family reunion", "reunion"), ("Mia's school pickup", "pickup"), ("holiday travel home", "holiday"),
        ("Dad's retirement party", "retirement"), ("baby shower", "baby shower"), ("summer cabin week", "cabin"),
        ("Grandma's recipes", "recipes"), ("anniversary dinner", "anniversary"), ("babysitting schedule", "babysitting"),
        ("moving help", "moving"),
    ),
    "friend": (
        ("ski trip", "ski trip"), ("wedding RSVP", "RSVP"), ("bachelor party", "bachelor"), ("book club", "book club"),
        ("fantasy football league", "fantasy"), ("climbing gym meetup", "climbing"), ("camping weekend", "camping"),
        ("housewarming party", "housewarming"), ("trivia night", "trivia"), ("half marathon training", "marathon"),
        ("game night", "game night"), ("road trip to Big Sur", "Big Sur"), ("borrowed camera", "camera"),
        ("podcast guest spot", "podcast"),
    ),
}
SERVICE_TOPICS = {
    "landlord": (("lease renewal", "lease"), ("broken heater", "heater"), ("parking spot", "parking")),
    "dentist": (("cleaning appointment", "cleaning"), ("crown fitting", "crown"), ("dental x-rays", "x-rays")),
    "accountant": (("tax return", "tax return"), ("quarterly estimates", "estimates"), ("receipts for deductions", "deductions")),
    "cleaner": (("deep clean", "deep clean"), ("biweekly schedule", "biweekly"), ("key handoff", "key handoff")),
    "mechanic": (("brake repair", "brake"), ("oil change", "oil change"), ("tire rotation", "tire")),
    "vet": (("vaccination booster", "vaccination"), ("checkup for Biscuit", "checkup"), ("flea medication", "flea")),
    "plumber": (("leaky faucet", "faucet"), ("clogged drain", "drain"), ("garbage disposal", "disposal")),
    "piano teacher": (("piano recital", "recital"), ("lesson schedule", "lesson"), ("sheet music", "sheet music")),
    "contractor": (("bathroom remodel", "remodel"), ("permit paperwork", "permit"), ("deck repair", "deck")),
    "insurance agent": (("policy premium", "premium"), ("accident claim", "claim"), ("home coverage", "coverage")),
}

ERRAND_TEMPLATES = (
    ("flight to {}", ("Lisbon", "Denver", "Tokyo", "Oslo", "Montreal", "Nashville", "Seattle", "Chicago", "Boston",
                      "Honolulu", "Reykjavik", "Athens", "Dublin", "Lima", "Austin", "Toronto", "Vancouver", "Cancun")),
    ("hotel in {}", ("Kyoto", "Barcelona", "Vienna", "Prague", "Savannah", "Tucson", "Lyon", "Florence", "Edinburgh", "Porto")),
    ("table at {}", ("Nopa", "Lucca", "Zuni", "Tartine", "Septime", "Delfina", "Kokkari", "Rintaro", "Nightbird")),
    ("tickets for {}", ("Hamilton", "Coldplay", "the Lakers game", "Wicked", "Adele", "the Dodgers game")),
    ("rental car in {}", ("Phoenix", "Banff", "Sedona", "Maui")),
)
ERRAND_EXTRAS = (
    ("renew my passport", "passport"), ("renew the car registration", "registration"),
    ("pick up suits from the tailor", "tailor"), ("book a haircut", "haircut"), ("order Mia's school uniform", "uniform"),
    ("schedule a flu shot", "flu shot"), ("cancel the meal kit subscription", "meal kit"),
    ("dispute a credit card charge", "credit card"), ("find a dog sitter", "dog sitter"),
    ("fix the cracked laptop screen", "laptop"), ("book a massage", "massage"), ("order new contact lenses", "contact lenses"),
    ("get a quote for solar panels", "solar"), ("sign Mia up for swim class", "swim"), ("renew the library books", "library"),
    ("order a cake from the bakery", "bakery"), ("reserve a campsite at Yosemite", "Yosemite"),
    ("buy a gift card for the teacher", "gift card"), ("donate old clothes", "donate"),
    ("schedule a window washer", "window washer"), ("refill the prescription", "prescription"),
    ("book a photographer for family photos", "photographer"), ("renew the gym membership", "gym"),
    ("order a new mattress", "mattress"), ("frame the vacation prints", "frame"), ("book a tennis court", "tennis"),
    ("find a Spanish tutor", "Spanish"), ("pay the speeding ticket", "speeding"),
    ("cancel the newspaper delivery", "newspaper"), ("book a wine tasting in Napa", "Napa"),
    ("schedule a chimney sweep", "chimney"), ("buy a costume for Halloween", "Halloween"),
    ("get the rug repaired", "rug"), ("find a notary", "notary"), ("reserve a cooking class", "cooking"),
    ("book a spa day", "spa day"), ("renew the museum membership", "museum"), ("sell the old couch", "couch"),
    ("book a pumpkin patch visit", "pumpkin"), ("order new running shoes", "running shoes"),
    ("replace the smoke detector", "smoke detector"), ("book a ferry to Victoria", "ferry"),
    ("get the gutter repaired", "gutter"), ("order flowers for the office", "flowers"),
    ("book a driving test", "driving test"), ("renew the fishing license", "fishing"),
    ("schedule an eye exam", "eye exam"), ("book a boat tour", "boat tour"), ("sign up for a pottery class", "pottery"),
    ("get the watch battery replaced", "watch battery"), ("book a karaoke room", "karaoke"),
    ("renew the zoo pass", "zoo pass"), ("find a locksmith", "locksmith"),
)
ERRAND_THREADS = 80  # errands drawn for threads; the rest of the pool stays unused (a reserve for probes)

# Background errands share no word with any thread topic: action x object x qualifier.
BACKGROUND_ACTIONS = (
    "compare prices for", "read reviews of", "dig up a coupon for", "see shipping times for", "look up the warranty on",
    "track down a refurbished", "set a price alert for", "download the manual for", "see who stocks", "look for a used",
)
BACKGROUND_OBJECTS = (
    "standing desk", "air purifier", "espresso grinder", "robot vacuum", "electric toothbrush", "mesh router",
    "label printer", "cordless drill", "rowing machine", "dehumidifier", "bread maker", "e-reader", "monitor arm",
    "leaf blower", "waffle iron", "bike rack", "smart thermostat", "white noise machine", "yoga mat",
    "carry-on suitcase",
)
BACKGROUND_QUALIFIERS = ("under three hundred dollars", "in matte black", "with free returns", "that ships by Saturday")
BACKGROUND_SIZE = 800

THREAD_COUNTS = ((1, 2, 3, 4), (0.15, 0.3, 0.35, 0.2))  # threads per contact (values, weights)
RESULT_COUNTS = ((0, 1, 2), (0.5, 0.35, 0.15))
FOLLOW_UP_CONTINUE = 0.45  # geometric follow-up count, capped at MAX_FOLLOW_UPS
MAX_FOLLOW_UPS = 4
REVIVAL_P = 0.25
REVIVAL_MIN_GAP = 21.0
MULTI_SHARE = 0.08

_PRO = ("work", "client", "recruiter")
START_DETAILS = {
    "pro": (
        "ask them to send the latest draft by Friday", "set up 30 minutes this week to go over it",
        "ask what they need from me to get it moving", "share my notes and ask for comments by Wednesday",
        "say I'd like to wrap it up before the end of the month",
    ),
    "personal": (
        "ask what dates work for everyone", "ask if they're still up for it", "say I can help out on the weekend",
        "ask what they think we should do", "say I'm excited and ask how I can help",
    ),
    "service": (
        "ask for the earliest available slot", "ask how much it will cost", "ask if next Tuesday morning works",
        "say it's getting urgent and ask what they can do",
    ),
    "errand": (
        "find a few options and send them to me", "book it if it's under budget", "check what's available next weekend",
        "look into it and tell me the best choice",
    ),
}
FOLLOW_UP_DETAILS = {
    "pro": (
        "ask to move the call to Thursday", "say the draft looks good but cut the second paragraph",
        "tell them I need one more day", "add that the budget is capped at 20k", "ask them to loop in their manager",
        "say Friday works better for me", "ask if there's any update",
    ),
    "personal": (
        "say Saturday works better for us", "ask if they need me to bring anything", "say Jamie can come too",
        "ask if we can push it a week", "say I'll call them tonight about it", "ask if there's any update",
    ),
    "service": (
        "confirm the appointment time", "ask if they can come earlier", "say the problem is back again",
        "ask them to send the invoice", "ask if there's any update",
    ),
    "errand": (
        "go with the cheaper option", "change it to the following weekend", "make it for two people instead of one",
        "check if there's anything closer to home", "ask if there's any update",
    ),
}
RESULT_DETAILS = (
    "say {first} replied that Thursday at 3 works", "say {first} wrote back asking for the final numbers",
    "say {first} confirmed and said thanks", "say {first} asked if we can push it to next week",
    "say {first} replied with a question about the timeline", "say {first} needs an answer by Monday",
)
ERRAND_RESULT_DETAILS = (
    "say you found three options and need the user to pick one", "say it's booked and the confirmation is in the inbox",
    "say the price went up and ask whether to go ahead", "say the first choice is full and suggest an alternative",
)
REVIVAL_DETAILS = (
    "ask whatever happened with it", "say we should pick it back up", "ask if it's still on",
    "check where it ended up", "say sorry for dropping it and ask where things stand",
)

_STOPWORDS = frozenset({"a", "an", "the", "to", "for", "of", "in", "on", "at", "my", "and", "with", "by", "up", "s"})


def _group(domain: str) -> str:
    return "pro" if domain in _PRO else "personal" if domain in ("family", "friend") else domain


def _errand_pool() -> tuple[tuple[str, str], ...]:
    templated = tuple(
        (template.format(filler), filler.removeprefix("the ").removesuffix(" game"))
        for template, fillers in ERRAND_TEMPLATES for filler in fillers
    )
    return templated + ERRAND_EXTRAS


ERRANDS = _errand_pool()
TOPIC_KEYWORDS: dict[str, str] = {
    topic: keyword
    for pool in (*TOPICS.values(), *SERVICE_TOPICS.values(), ERRANDS)
    for topic, keyword in pool
}


def topic_keyword(topic: str) -> str:
    """The keyword a message uses to name a thread topic."""
    return TOPIC_KEYWORDS[topic]


def topic_words(text: str) -> set[str]:
    """Lowercase content words of a topic, for vocabulary-overlap checks."""
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in _STOPWORDS}


def _contacts() -> tuple[list[Contact], dict[str, tuple[tuple[str, str], ...]]]:
    """Every contact, plus the topic pool each contact draws from (keyed by Contact.full)."""
    contacts: list[Contact] = []
    pools: dict[str, tuple[tuple[str, str], ...]] = {}

    def add(contact: Contact, pool: tuple[tuple[str, str], ...]) -> None:
        contacts.append(contact)
        pools[contact.full] = pool

    for first, last in WORK:
        add(Contact(first, last, "work", WORK_COMPANY), TOPICS["work"])
    for first, last, company in CLIENTS:
        add(Contact(first, last, "client", company), TOPICS["client"])
    for first, last, company in RECRUITERS:
        add(Contact(first, last, "recruiter", company), TOPICS["recruiter"])
    for first, last in FAMILY:
        add(Contact(first, last, "family"), TOPICS["family"])
    for first, last in FRIENDS:
        add(Contact(first, last, "friend"), TOPICS["friend"])
    for first, last, company, role in SERVICES:
        add(Contact(first, last, "service", company), SERVICE_TOPICS[role])
    return contacts, pools


def _threads(contacts: list[Contact], pools: dict[str, tuple[tuple[str, str], ...]], rng: random.Random) -> list[Thread]:
    drafts: list[tuple[str | None, str, str]] = []
    for contact in contacts:
        count = rng.choices(*THREAD_COUNTS)[0]
        if contact.first in COLLIDING_FIRSTS:
            count = max(count, 2)  # probes need both namesakes to own agent-backed threads
        pool = pools[contact.full]
        for topic, _ in rng.sample(pool, min(count, len(pool))):
            drafts.append((contact.full, topic, contact.relation))
    for topic, _ in rng.sample(ERRANDS, ERRAND_THREADS):
        drafts.append((None, topic, "errand"))
    rng.shuffle(drafts)
    return [Thread(f"t{i:03d}", contact, topic, domain) for i, (contact, topic, domain) in enumerate(drafts)]


def _geometric(rng: random.Random) -> int:
    count = 0
    while count < MAX_FOLLOW_UPS and rng.random() < FOLLOW_UP_CONTINUE:
        count += 1
    return count


class _Mentions:
    """Builds the mention, forbid and detail fields of a thread's intents."""

    def __init__(self, contacts: list[Contact], threads: list[Thread], rng: random.Random) -> None:
        self.rng = rng
        self.by_full = {c.full: c for c in contacts}
        self.thread_count: dict[str, int] = {}
        for thread in threads:
            if thread.contact:
                self.thread_count[thread.contact] = self.thread_count.get(thread.contact, 0) + 1
        self.namesakes: dict[str, list[Contact]] = {}
        for contact in contacts:
            self.namesakes.setdefault(contact.first, []).append(contact)

    def names(self, thread: Thread) -> tuple[str, ...]:
        if thread.contact is None:
            return ()
        contact = self.by_full[thread.contact]
        if len(self.namesakes[contact.first]) > 1:
            return contact.first, contact.last
        return (contact.first,)

    def forbid(self, thread: Thread) -> tuple[str, ...]:
        """The namesakes' last names, so a message about one Alice never names the other."""
        if thread.contact is None:
            return ()
        contact = self.by_full[thread.contact]
        return tuple(c.last for c in self.namesakes[contact.first] if c.full != contact.full)

    def mention(self, thread: Thread, kind: str) -> tuple[str, ...]:
        keyword = topic_keyword(thread.topic)
        if thread.contact is None:
            return (keyword,)
        first = self.by_full[thread.contact].first
        if kind == "result":
            return (first,)
        needs_keyword = kind in ("start", "revival") or self.thread_count[thread.contact] > 1
        return self.names(thread) + ((keyword,) if needs_keyword else ())

    def detail(self, thread: Thread, kind: str) -> str:
        group = _group(thread.domain)
        if kind == "start":
            return self.rng.choice(START_DETAILS[group])
        if kind == "follow_up":
            return self.rng.choice(FOLLOW_UP_DETAILS[group])
        if kind == "revival":
            return self.rng.choice(REVIVAL_DETAILS)
        if thread.contact is None:
            return self.rng.choice(ERRAND_RESULT_DETAILS)
        return self.rng.choice(RESULT_DETAILS).format(first=self.by_full[thread.contact].first)

    def intent(self, thread: Thread, kind: str) -> Intent:
        return Intent(
            kind=kind,
            thread_ids=(thread.id,),
            contacts=(thread.contact,) if thread.contact else (),
            topic=thread.topic,
            detail=self.detail(thread, kind),
            mention=self.mention(thread, kind),
            forbid=self.forbid(thread),
            message_type="agent" if kind == "result" else "user",
        )

    def multi(self, first: Thread, second: Thread, detail_a: str) -> Intent:
        mention: list[str] = []
        for item in self.mention(first, "follow_up") + self.mention(second, "follow_up"):
            if item not in mention:
                mention.append(item)
        detail_b = self.detail(second, "follow_up")
        return Intent(
            kind="multi",
            thread_ids=(first.id, second.id),
            contacts=tuple(t.contact for t in (first, second) if t.contact),
            topic=f"{first.topic}; {second.topic}",
            detail=f"two requests in one message: {detail_a} (about: {first.topic}); and {detail_b} (about: {second.topic})",
            mention=tuple(mention),
            forbid=tuple(dict.fromkeys(self.forbid(first) + self.forbid(second))),
        )


def _timeline(rng: random.Random) -> list[tuple[float, str]]:
    """(day, kind) for one thread: start, follow-ups, results and maybe a revival."""
    start = round(rng.uniform(0, LIFE_DAYS - 15), 3)
    user_days = [start]
    day = start
    for _ in range(_geometric(rng)):
        day += rng.expovariate(1 / 4) + 0.1
        if day > LIFE_DAYS:
            break
        user_days.append(round(day, 3))
    results = []
    for _ in range(rng.choices(*RESULT_COUNTS)[0]):
        when = round(rng.choice(user_days) + rng.uniform(0.05, 1.5), 3)
        if when <= LIFE_DAYS:
            results.append(when)
    timeline = [(start, "start")] + [(d, "follow_up") for d in user_days[1:]] + [(d, "result") for d in sorted(results)]
    revive, gap = rng.random() < REVIVAL_P, rng.uniform(REVIVAL_MIN_GAP + 0.5, 45)
    revival_day = round(max(d for d, _ in timeline) + gap, 3)
    if revive and revival_day <= LIFE_DAYS:
        timeline.append((revival_day, "revival"))
    return timeline


def build_plan(seed: int = 1) -> LifePlan:
    """The seeded life plan; the same seed always gives the same plan."""
    rng = random.Random(seed)
    contacts, pools = _contacts()
    threads = _threads(contacts, pools, rng)
    mentions = _Mentions(contacts, threads, rng)

    # (day, sequence, intent): the sequence keeps same-day ordering stable.
    raw: list[tuple[float, int, Intent]] = []
    starts: dict[str, float] = {}
    revived: set[str] = set()
    for thread in threads:
        for day, kind in _timeline(rng):
            if kind == "start":
                starts[thread.id] = day
            if kind == "revival":
                revived.add(thread.id)
            raw.append((day, len(raw), mentions.intent(thread, kind)))

    # Fold a share of follow-ups into multi messages that also continue a second, unrevived thread.
    users = sum(1 for _, _, intent in raw if intent.message_type == "user")
    follow_ups = [i for i, (_, _, intent) in enumerate(raw) if intent.kind == "follow_up"]
    by_id = {t.id: t for t in threads}
    for index in sorted(rng.sample(follow_ups, min(len(follow_ups), round(users * MULTI_SHARE)))):
        day, seq, intent = raw[index]
        first = by_id[intent.thread_ids[0]]
        candidates = [
            t for t in threads
            if t.id != first.id and t.id not in revived and starts[t.id] < day
            and (t.contact is None or t.contact != first.contact)
        ]
        if candidates:
            raw[index] = (day, seq, mentions.multi(first, rng.choice(candidates), intent.detail))

    raw.sort(key=lambda item: (item[0], item[1]))
    events = [Event(f"e{i:04d}", day, intent) for i, (day, _, intent) in enumerate(raw)]
    return LifePlan(seed=seed, persona=PERSONA, contacts=contacts, threads=threads, events=events,
                    background=_background(rng))


def _background(rng: random.Random) -> list[Intent]:
    combos = [
        (action, obj, qualifier)
        for action in BACKGROUND_ACTIONS for obj in BACKGROUND_OBJECTS for qualifier in BACKGROUND_QUALIFIERS
    ]
    rng.shuffle(combos)
    return [
        Intent(kind="start", thread_ids=(), topic=phrase, detail=f"ask the assistant to {phrase}", mention=(obj,))
        for phrase, obj in (_background_phrase(*combo) for combo in combos[:BACKGROUND_SIZE])
    ]


def _background_phrase(action: str, obj: str, qualifier: str) -> tuple[str, str]:
    """("compare prices for an air purifier in matte black", "air purifier")."""
    if action.endswith(("refurbished", "used")):
        return f"{action} {obj} {qualifier}", obj
    article = "an" if obj[0] in "aeiou" else "a"
    return f"{action} {article} {obj} {qualifier}", obj
