# Plan 1b: Lived-in eval world

**Goal:** a realistic benchmark world (spec §8.1). A seeded life plan, with messages written by an LLM, played
through the real Interaction Agent turn. The result is a roster the IA named itself, plus held-out probe
cases with real conversation history. It's frozen to `evals/worlds/lived/world-s1.json` and run like the synthetic world.

**Conventions (same as Plan 1):** Python 3.14 in `.venv` (`.venv/Scripts/python`), pytest with `asyncio_mode = auto`,
offline tests only (use `ScriptedLLM`), match the existing code style in `evals/` (dataclasses, type hints, short
docstrings, sparse comments). Never print or commit the API key. Don't touch `.serena/`, `pyproject.toml`, `uv.lock`.
Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
After editing a protected file, run `python -m evals.integrity --update` and commit `evals/PROTECTED.sha256` with it.

## Package layout

```
evals/worlds/lived/
  __init__.py
  model.py      shared types (Task 0)
  plan.py       seeded persona, contacts, threads, events, background errands (Task 1)
  writer.py     LLM writes an Intent as a user message; mention checks (Task 2)
  grow.py       plays events through the real IA turn; snapshots; background naming (Task 3)
  probes.py     held-out probe cases per routing kind at each snapshot (Task 4)
  store.py      JSON save/load -> run units (Task 5)
  generate.py   CLI: plan -> grow -> probes -> world-s1.json (Task 5)
```

## Task 0: shared types (done by the controller)

`evals/worlds/lived/model.py`:

- `Contact(first, last, relation, company)` with `.full`, `.email` (like `Person`).
- `Thread(id, contact: str | None, topic: str, domain: str)`, where `contact` is `Contact.full` or `None` for errands.
- `Intent(kind, thread_ids, contacts, topic, detail, mention, forbid, message_type)`: what one message must say.
  `kind` is an event or probe kind. `mention` lists strings the message must contain (case-insensitive).
  `forbid` lists strings it must not contain. `detail` is a free-text instruction for the writer, such as
  "say you can sign on Friday".
- `Event(id, day, intent)`: one step of the life plan, where `day` is a float, days since the life started.
- `LifePlan(seed, persona, contacts, threads, events, background)`, where `background` is a list of `Intent`.
- `AgentSpec` gains `threads: list[str]` (gold only; systems must never read it) in `evals/worlds/model.py`.

## Task 1: life plan (`plan.py`)

`build_plan(seed: int = 1) -> LifePlan`, deterministic, with no LLM calls.

- **Persona:** a fixed short description (name, job, city, family), used by the writer.
- **About 80 contacts** across relations: work (colleagues, clients, recruiters), family, friends, services
  (landlord, dentist, accountant). Collisions are deliberate: two contacts named Alice, two named Ben, and a
  colleague and a cousin sharing a surname.
- **Topic pools per relation,** for example work: contract renewal, Q3 roadmap review, offer negotiation; family:
  Thanksgiving plans, Mom's birthday; friends: ski trip, wedding RSVP; services: lease renewal, cleaning
  appointment. **Errand threads** have no contact: book a restaurant, flight to Lisbon, and so on.
- **About 300 threads,** roughly 70% with a contact. A contact may have several threads on different topics
  (that's what same_person_new_topic tests). No contact has two threads on the same topic.
- **Events over 180 days:** each thread has a `start`, then 0–4 `follow_up`s (geometric distribution), 0–2
  `result`s (agent results, `message_type="agent"`), and with probability 0.25 a `revival` at least 21
  days after its last event. About 8% of user events are `multi` (two threads in one message). Sort events
  by day. Event ids are `e0000`…
- **Mentions:** a contact-thread `start`/`follow_up`/`revival` mentions the contact's first name; `start`
  also mentions a topic keyword. For a colliding first name, the last name is mentioned too, so growth
  never has to guess. `result` intents mention the contact's first name and say what they replied.
- **`background`:** 800 errand intents (kind `start`, no contact). They're built from a separate vocabulary
  (action × object × qualifier, for example "cancel my gym membership", "compare prices for a standing desk") that
  shares no topic keyword with any thread. They must be unique.
- **Tests (`evals/tests/test_lived_plan.py`):**
  - determinism: same seed gives the same plan;
  - a golden sha256 of the event ids and intents;
  - events are sorted;
  - every follow-up, result and revival comes after its thread's start;
  - the collisions exist;
  - background shares no topic keyword with any thread;
  - at least 600 events.

## Task 2: message writer (`writer.py`)

`async def write(intent, *, transcript_tail, persona, llm, model, sample=0) -> str | None`

- **Prompt:**
  - system: you write the exact chat message a busy person sends to their AI assistant. Be short and
    casual, often lowercase, occasionally a typo. Output only the message.
  - user: the persona, the last 6 transcript entries rendered as `tag: text`, and the intent in plain words
    built from `kind`, `contacts`, `topic` and `detail`. It also states the hard rules: must mention X,
    must not mention Y.
- **Agent results** (`message_type == "agent"`): the writer produces only the body. Example: "Alice replied: she
  can do Thursday at 3". `grow.py` adds the `[SUCCESS] <agent name>: ` prefix.
- **Validation:** strip quotes and whitespace, then check `mention`/`forbid`. On failure, retry with
  `sample + 1`, then `+ 2`. After three failures, return `None`.
- **Client call:** `llm.complete(model=..., system=..., messages=[...], tools=[], sample=...)`. Change
  `evals/llm.py:build_payload` to omit the `tools` key when `tools` is empty. Existing cache keys don't
  change, because every existing call has tools.
- **Tests:** with `ScriptedLLM`, check that a valid message is returned, that a forbidden mention retries with a new
  sample, that three failures return None, and that the payload has no `tools` key when empty.

## Task 3: growth (`grow.py`)

`async def grow(plan, *, llm, model, sizes=(10, 50, 200), max_agents=200) -> GrowthResult`

- **State:**
  - agents: a list of `AgentSpec` in creation order, with ids slugified from the name and made unique;
  - transcript: a list of `(tag, text, timestamp_iso)`;
  - thread to agent ids;
  - agent id to last active day;
  - the current day.
- **For each event in order:**
  - **Agent results:** skip a `result` whose thread has no agent yet. Otherwise, the message is
    `"[SUCCESS] {agent.name}: {body}"`.
  - **Writing:** write the message with the writer, using the last 6 transcript entries. Skip the
    event if the writer returns None.
  - **The IA turn:** build a `Case` with kind `follow_up`, which only satisfies the schema, the message,
    the message type, and context set to the last 20 transcript entries. Build a `World` from the current
    agents, with `now` set to the event's time and `created_days`/`last_active_days` computed from it.
    Run `run_routing_case(case, world, BaselineSystem(), llm, model=model, sample=0)`.
  - **Created names:**
    - A created name becomes a new `AgentSpec`. Its `purpose` is the instructions of that
      `send_message_to_agent` call, its `threads` the event's thread ids, its history
      `[(instructions, "")]`, and `entities=[]`.
    - If a turn creates several agents for a multi event, assign threads in order. If counts differ,
      give every created agent all of the event's threads.
  - **Dispatched ids:** add the event's threads to the agent's `threads` if missing, update its last active
    day, and append to its history.
  - **Transcript:** append `user_message` or `agent_message`, then a `poke_reply` for each user-facing
    message in `decision.user_messages`.
  - **Snapshots:** take one the first time `len(agents) >= size`, holding a deep copy of the agents, the
    transcript, the thread map and the day. Stop once `max_agents` is reached. Raise if the events run out
    before the largest size.
  - **Errors:** a turn that raises `RoutingCaseError` is logged and the event is skipped. A `RunAborted`
    propagates.
- **Background:**

  `async def name_background(plan, *, llm, model, concurrency=8) -> list[AgentSpec]`

  - Each background intent is written by the writer with an empty transcript.
  - It's then run through one IA turn with an empty roster. The first created name becomes the agent:
    `purpose` is its instructions, `threads=[]`, and the created day is spread uniformly over the 180
    days.
  - Turns that create nothing are dropped.
- **Merging at 1,000:** `merge_background(snapshot_200, background, target=1000)` interleaves background
  agents by created day. A background name equal to an existing name is dropped (production would have
  reused that agent). It stops at `target` agents.
- **Progress:** print a line every 25 events, for example `event 150/640 agents 62 spent $0.41`.
- **Tests:** a `ScriptedLLM` stands in for both the writer and the IA. Its rule: for a `start` event, create an
  agent named after the topic; otherwise reuse the agent whose name contains the topic. Check that:
  - snapshots are taken at the right sizes;
  - threads are recorded;
  - a merged thread keeps one agent;
  - the transcript holds user, reply and agent entries;
  - results are skipped before an agent exists;
  - background names merge on equal names.

## Task 4: probes (`probes.py`)

`async def build_probes(snapshot, plan, *, llm, model, per_kind=4, rng_seed) -> list[Case]`

- Probes are held out and never played into growth. Each probe's context is the snapshot transcript's last
  20 entries.
- **Ids:** `lived-s{seed}-{size}-{kind}-{i:02d}`, with `world_seed` set to the plan seed.
- **Gold for a thread:** its agents, as `gold_targets=[first]` and `acceptable_ids=rest`.
- **Probe types:**

| Kind | Picked from the snapshot state | Intent | Gold |
|---|---|---|---|
| follow_up | thread with an agent, active in the last 14 days | mention first name + topic keyword | thread agents |
| vague_reference | (a) the thread of the last user event, with an agent: pronoun only, forbid names and topic; (b) threads whose topic keyword is unique among threads: topic keyword only, forbid the name | as described | thread agents |
| new_task | an unused errand from a reserve pool, disjoint from the plan | describe it | `NEW` |
| same_person_new_topic | a contact with ≥1 agent-backed thread, plus a topic from their relation pool they've never had | mention name + topic | `NEW` |
| multi_task | two agent-backed threads, or one plus a reserve errand | mention both | both (agent ids and/or `NEW`) |
| same_name_people | a colliding first name where both contacts have agent-backed threads and neither last name is in the context | first name only, forbid both last names and topics | `[]`, `clarify_ok=True` |
| cross_agent | agent-backed thread A, plus another contact B: "let A know B approved the {A topic}" | mention both first names + A's topic | A's agents |
| dormant_revival | agent-backed thread idle ≥ 21 days | mention name + topic | thread agents |
| typo | as follow_up, then code swaps two inner letters of the first name in the message (deterministic) | | thread agents |
| agent_result | agent-backed thread; body from the writer; `message_type="agent"` | | thread agents, `dispatch_optional=True`, `clarify_ok=True` |

- **Selection:** use `random.Random(rng_seed)`. Skip a probe whose writer returns None. Produce up to
  `per_kind` per kind.
- **Tests:** with a hand-built snapshot and `ScriptedLLM`, check that:
  - gold follows the thread map, with multiple agents giving gold plus acceptable ids;
  - same_name is skipped when a last name appears in the context;
  - the typo differs from the name and is deterministic;
  - ids are unique;
  - new-task probes never match an existing thread.

## Task 5: store, generate CLI, runner integration

- **`store.py`:**
  - `save(path, plan_seed, snapshots, probes_by_size, merged_1000)` writes JSON. It holds the agents
    (with `threads`), the transcript tail and the probes for each of the sizes 10, 50, 200 and 1000; the
    1,000 size uses the 200 probes.
  - `load_lived(path) -> list[tuple[str, World, list[Case]]]` uses the labels `lived-10`, `lived-50`,
    `lived-200` and `lived-1000`.
  - Context entries become `(tag, text)` tuples. The probe's real timestamps are kept by extending
    `render_context` in `evals/layers/routing.py` to accept a 3-tuple `(tag, text, timestamp)`; 2-tuples
    behave as today.
- **`generate.py`:**
  - Run it as `python -m evals.worlds.lived.generate --seed 1 --model anthropic/claude-haiku-5.5 --budget 6`.
  - It builds the plan, grows the roster, names the background agents, merges, builds the probes, and
    saves `evals/worlds/lived/world-s1.json`.
  - It uses `OpenRouterLLM` with the shared cache, `Budget(budget)`, and `concurrency=8` for background
    naming. It prints the total spend.
- **`run.py`:**
  - New config key `world: synthetic | lived` (default synthetic), plus `lived_path` (default the s1 file).
    Validate both.
  - For `lived`, the units come from `load_lived`, filtered to the sizes in `n_agents`; handwritten cases
    work as today.
- **Integrity:** add `evals/worlds/lived/*.py` and `world-s1.json` to `PROTECTED` in `evals/integrity.py`.
  The JSON file may be absent until Task 6; integrity must tolerate a missing protected file by hashing
  its absence.
- **Experiment config:** `evals/experiments/baseline-lived.yaml` (baseline system, Haiku 5.5, k=3, sizes 10, 50,
  200 and 1000, budget $5, concurrency 8).
- **Tests:**
  - save/load round trip;
  - `load_lived` labels and case counts;
  - `render_context` with 3-tuples;
  - config validation for `world`;
  - a fake-LLM run on a tiny saved lived world.

## Task 6: generate, check, baseline (controller)

1. Run the generator live, in the background; expect about 60–75 minutes and about $4.
2. Spot-check 30 probes by hand across kinds: does the message say what the gold assumes? Drop or fix
   bad intents in code, regenerate (cached, cheap), and record the findings.
3. Commit `world-s1.json` and update the checksum.
4. Run `baseline-lived.yaml`. Add a journal entry and an error-analysis section on the lived world. Seed
   the lived cases that pass every sample into the regression suite. Commit.
