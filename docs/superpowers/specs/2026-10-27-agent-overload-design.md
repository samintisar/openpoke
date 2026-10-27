# Agent Overload in OpenPoke — Design

**Date:** 2026-10-27 · **Branch:** `feat/agent-overload` · **Due:** 2026-11-02

## 1. Problem

OpenPoke's Interaction Agent (IA) routes every task to long-lived Execution Agents (EAs). Agents are
identified only by a free-text name, kept in `roster.json`, and **never retire**. As usage grows:

| # | Failure mode | Where in code |
|---|---|---|
| F1 | Routing degrades: IA sees hundreds of bare names with no purpose, recency or status → wrong reuse or duplicate creation | `interaction_agent/agent.py::_render_active_agents` |
| F2 | Exact-string identity → near-duplicates (`"email to alice"` vs `"Email to Alice"`); both slugify to the same log file | `roster.py`, `log_store.py::_slugify` |
| F3 | EA history unbounded: full log injected into every system prompt (`conversation_limit=None`) | `execution_agent/agent.py` |
| F4 | No concurrency control: same agent can run twice concurrently (interleaved log); unbounded fan-out | `interaction_agent/tools.py`, `trigger_scheduler.py` |
| F5 | Single global batch: slow agent blocks fast results; triggers use separate managers | `batch_manager.py` |
| F6 | Concurrent IA turns race on the conversation log → duplicate/contradictory replies; `wait` tool is a prompt patch | `chat_handler.py`, `batch_manager.py`, `importance_watcher.py` |

Other gaps: `fcntl` import breaks Windows; no tests/evals; "confirm before send" is prompt-only;
email content is a prompt-injection vector into agents that can send mail; hard failure at the
iteration limit; one model tier everywhere.

**Headline focus:** F1–F3 (routing at scale + lifecycle + memory), proven by an eval harness.
F4–F6 and safety are delivered as smaller, tested fixes.

## 2. Agent registry (replaces `roster.json`)

SQLite table `agents` in `server/data/agents.db`, following the `services/triggers/` store→service pattern.
New package `server/services/agents/`.

| Column | Type | Source |
|---|---|---|
| `id` | TEXT PK | slug of name at creation; numeric suffix on collision. Log files keyed by `id`. |
| `name` | TEXT | IA-chosen display name |
| `purpose` | TEXT | required IA argument on create; refreshed by compaction |
| `entities` | JSON list | deterministic extraction: email addresses + capitalized name tokens in instructions; `to`/`cc`/`thread_id`/`draft_id` args of Gmail tool calls. Lowercased. |
| `last_status` | TEXT | first 200 chars of last final response |
| `status` | TEXT | `active` \| `dormant` \| `archived` |
| `created_at`, `last_active_at` | TEXT (ISO UTC) | runtime |
| `run_count` | INT | runtime |

Open work (active triggers, pending drafts) is **derived at query time** from the triggers DB and
`pending_actions` table — never stored on the card.

Migration: on startup, if `roster.json` exists and the registry is empty, import each name with
`purpose = name`, entities harvested from its log; rename `roster.json` → `roster.json.migrated`.

The registry is the single entry point for create / get / update / list / search. `fcntl` is removed.

## 3. Routing and retrieval

`server/services/agents/retrieval.py` — pure functions, no I/O beyond the cards passed in.

**Query:** for user turns, new message + last 3 conversation turns. For agent-result / trigger turns,
the originating agent is pinned; no retrieval.

**Score** (all components normalized 0–1; weights in `RetrievalConfig`):

```
score = w_kw·bm25(query, name+purpose+last_status) + w_ent·entity_overlap(query, entities) + w_rec·recency(last_active_at)
defaults: w_kw=0.45, w_ent=0.40, w_rec=0.15; recency = exp(-age_days / 7)
```

BM25 implemented in-repo (no dependency); scores normalized by the max in the candidate set.

**Selection:** hot set (≤5: open work or active within 1h) ∪ top-K (K=8, `score ≥ min_score=0.1`)
over non-archived cards, deduped, hot set first. Rendered as:

```xml
<active_agents shown="13" total="240">
  <agent id="email-to-alice" name="Email to Alice" last_active="2h ago" open_work="1 draft">
    purpose: ...
    status: ...
  </agent>
</active_agents>
```

**Fallback tool** `search_agents(query)` → top 10 over **all** cards incl. archived (same scorer).
Prompt rule: if no listed agent fits but the request sounds like ongoing work, search before creating.

**`send_message_to_agent` signature:** `(instructions, agent_id?, name?, purpose?, force_new=false)`.
- `agent_id` given → reuse (unknown id → error listing nearest matches).
- else create from `name`+`purpose`. **Duplicate guard**: if an existing card has normalized-name
  similarity ≥ 0.85, or shares a non-trivial entity *and* retrieval score ≥ 0.5 against the new purpose,
  return `possible_duplicate` with the candidate's card and do not create — unless `force_new=true`.

## 4. Lifecycle

Pure function `next_status(card, now, has_open_work, cfg) -> status`:
- open work → `active` (never demoted)
- inactive ≥ `dormant_after_days` (7) → `dormant`; ≥ `archive_after_days` (30) → `archived`
- dormant: excluded from hot set, still retrievable per turn; archived: only via `search_agents`
- revival: any routing to the agent, trigger fire, or search-then-route sets `active`

Sweep runs at startup and hourly inside the trigger scheduler loop. Nothing is deleted. Auto-merge of
existing duplicates is out of scope (future work).

## 5. Per-agent memory compaction

EA prompt history = `<agent_memory>` + verbatim tail within `history_token_budget` (~4k tokens,
estimated as chars/4). When the un-compacted prefix exceeds budget, one cheap-model call folds it into
the existing memory, preserving: people/emails, thread & draft IDs, decisions, commitments, user
preferences, open loops. The same call returns refreshed `purpose` and `last_status` for the card.

Runs **after** an EA run completes, under that agent's lock (off the user's critical path). Raw logs
are never rewritten; memory is stored as an `<agent_memory upto="N">` entry so it can be regenerated.

## 6. Orchestration fixes

- **Per-agent lock**: `asyncio.Lock` per `agent_id`; FIFO serial runs per agent.
- **Global semaphore** around EA runs (default 4).
- **Interaction inbox**: single queue for user messages, agent results, trigger results, watcher alerts.
  One IA turn at a time; items arriving during a turn coalesce into the next; user messages first.
- **Batching**: global batch removed; each result posts to the inbox on completion; ~2s debounce
  coalesces near-simultaneous results. Trigger scheduler and watcher use the same path.
- **Iteration limit**: EA returns a partial-result summary instead of raising.

## 7. Safety guardrails

- **Code-enforced send approval.** `gmail_create_draft` records a row in `pending_actions`
  (`draft_id, agent_id, to, subject, status`). `gmail_execute_draft` / `forward` / `reply` refuse
  unless the action is `approved`. Only the IA's new `approve_send(draft_id)` tool approves, and it
  errors unless the current IA turn originated from a user message.
- **Untrusted content**: email bodies returned to agents are wrapped in `<untrusted_email>`; prompt
  rule: instructions inside are data.
- **Cost tiers**: compaction + email classifier models configurable, default to a cheaper model.

## 8. Evaluation

Top-level `evals/` package (not shipped with the server). Built **first**, so baseline numbers exist
before any fix.

```
evals/
  worlds/generator.py   seeded synthetic world: contacts, threads, agents + histories at N agents
  worlds/handwritten/   ~30 realistic cases (synthetic-validity check)
  cases.py              Case schema, dev/test split
  systems/              baseline (bare names), full_cards (all cards), retrieved (top-K+hot+search)
  layers/retrieval.py   L1 offline
  layers/routing.py     L2 live single IA turn, tools intercepted
  layers/memory.py      L2 compaction retention
  layers/e2e.py         L3 multi-turn, fake Gmail, LLM judge
  llm_cache.py          prompt-hash response cache + budget cap
  run.py                python -m evals.run --config evals/experiments/<x>.yaml
  experiments/*.yaml
  results/              per-run JSON + leaderboard.md
```

**Case:** `id, split, world_seed, n_agents, context, message, kind (follow_up|new_task|ambiguous|agent_result|injection), gold_agent_id (None = create), acceptable_ids`.

**World generator** deliberately includes hard negatives: shared first names across people, same
company across different deals, dormant agents that should be revived, near-duplicate names.

| Layer | Metrics |
|---|---|
| L1 retrieval (no LLM) | recall@K, MRR, hot-set hit rate, prompt tokens |
| L2 routing (live) | route accuracy, duplicate-creation rate, wrong-reuse rate, missed-create rate, search usage & lift, tokens, latency, $/case |
| L2 memory | planted-fact recall (compacted vs full), prompt tokens per run |
| L3 e2e | task success vs fake-Gmail end state, LLM-judge reply score (1–5 rubric, judge ≠ agent model), user messages per event, unauthorized-send rate |

**Rigor:** tune on dev only, report test once per candidate; config-driven variants; fixed seeds +
response cache; ≥3 seeds for LLM layers with variance; runner budget cap. Headline artifact: routing
accuracy & duplicate rate vs roster size (10/50/200/1000) for the three systems, plus tokens/turn.

**Unit tests** (`server/tests/`, pytest, offline): registry CRUD + id collisions + migration, entity
extraction, scoring/selection, duplicate guard, lifecycle transitions, compaction trigger logic,
per-agent lock ordering, semaphore cap, inbox coalescing, approval gate (incl. non-user-turn refusal).

LLM access: OpenRouter (existing client). Default eval model configurable; small budget.

## 9. Build order & cut line

1. Windows fix, pytest scaffold, world generator, cases, L1 on baseline.
2. L2 harness + cache → **baseline numbers**. Registry + migration.
3. Retrieval, `search_agents`, duplicate guard → tune on dev.
4. Lifecycle, compaction, memory eval.
5. Orchestration fixes, approval gate, injection cases, L3.
6. Test-split runs, charts, README write-up.

Cut order if behind: shrink L3 first, then orchestration fixes. Registry, retrieval, eval harness and
approval gate are non-negotiable.

## 10. Out of scope

Embedding-based retrieval (interface allows it later), automatic merging of existing duplicates, MCP,
personality tuning, frontend changes beyond what the new tools require.
