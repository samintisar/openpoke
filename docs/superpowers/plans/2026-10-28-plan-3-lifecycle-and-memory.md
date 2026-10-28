# Plan 3: Agent lifecycle and memory compaction

**Goal:** spec §4 (lifecycle) and §5 (per-agent memory compaction), each with a measurement.

- **Lifecycle:** stale agents stop competing for attention, but they can still be revived. Check this on the lived
  world: `dormant_revival` probes must not regress, and L1 must keep recall@shown at 1.00. Then run `retrieved` live
  with lifecycle statuses on.
- **Compaction:** bound the execution agent's history (F3) without losing what it needs. The new memory eval measures
  planted-fact recall with compacted vs full history, and prompt tokens.
  - **Targets:** identifiers 100% (they're preserved by code), prose facts within 5 points of full history, and prompt
    tokens ≤ ~4.5k at a 40-run history.

**Already in place:**
- `AgentDirectory.update_card(id, *, purpose, last_status, status)`, in both the in-memory and SQLite versions.
- Retrieval keeps archived cards out of `select` and dormant cards out of the hot set.
- `touch` sets `active`.

**Conventions:** as in Plan 2. Subagents don't commit or run `evals.integrity --update`.

## Task 1: lifecycle

- **`server/services/agents/lifecycle.py`**
  - `LifecycleConfig(dormant_after_days=7, archive_after_days=30)`.
  - `next_status(card, now, has_open_work, cfg) -> str`, per spec §4: open work means `active`.
  - `sweep(directory, now, open_work_ids, cfg) -> dict[str, str]` returns the changed ids mapped to their new
    status, and applies the change through `update_card`.
- **Open work:** an active trigger whose next fire is within 24h. Add a helper in the triggers service:
  `agent_ids_with_triggers_due(now, within=timedelta(hours=24))`. Pending drafts get added in Plan 4; leave a
  documented hook.
- **When the sweep runs:** at app startup (`app.py`, after `get_agent_directory()`), then hourly inside the trigger
  scheduler loop. A failure is logged and never crashes the loop.
- **Revival:** `touch` already sets `active`. A trigger fire must also revive its agent, by calling `touch`, or
  `update_card(status="active")` plus `last_active_at`.
- **Eval:** `evals/systems/cards_from_world.py` sets each card's status from `next_status(card, world.now, False)`.
  Add `lifecycle: bool = True` to `RetrievedSystem(...)` so the effect can be ablated. Register a system
  `retrieved_nolife` with lifecycle off, and give it a config.
- **Tests:** the status boundaries (exactly 7 and 30 days), open work never demoted, the sweep changing only what's
  needed, revival on touch and on trigger fire, `cards_from_world` statuses, and the scheduler hook calling the sweep
  (stubbed).

## Task 2: memory compaction (server)

**`server/services/agents/memory.py`.** The pure parts take injected callables; no I/O except through those.

- `estimate_tokens(text) = len(text) // 4`.
- `Entry(tag, text, timestamp)`, parsed from the log store's entries.
- `split_for_budget(entries, budget_tokens, upto) -> (prefix, tail)`. The tail is the newest entries that fit in the
  budget. The prefix is everything after the existing memory's `upto` index and before the tail.
- `extract_identifiers(texts) -> list[str]` covers email addresses, Gmail-style thread and message ids (hex 16+), and
  the `draft_id`/`thread_id`/`message_id` values that appear as `key: value`, `"key": "value"` or `key=value` in tool
  calls and responses. It's ordered and deduped.
- `COMPACTION_PROMPT` asks for JSON: `{"memory": str, "purpose": str, "last_status": str}`.
  - `memory` keeps people, decisions, commitments, user preferences and open loops, in under ~300 words.
  - `purpose` is one sentence about the agent's whole matter.
  - `last_status` is one line.
  - The prompt says identifiers are kept separately, so they need not be repeated.
- `async def compact(prefix, existing_memory, purpose, *, complete) -> CompactionResult(memory, purpose, last_status)`.
  `complete(system, user) -> str` is injected. Invalid JSON, or an empty memory, raises `CompactionError`.
- `render_history(memory, identifiers, tail_entries) -> str` gives
  `<agent_memory>…</agent_memory>\n<agent_identifiers>…</agent_identifiers>\n` plus the tail rendered as the log
  store renders it.

**Log store:**
- Add `record_memory(agent_id, memory, identifiers, upto)` (tag `agent_memory`, payload JSON) and
  `latest_memory(agent_id)`.
- Raw entries are never rewritten. `iter_entries` and `load_transcript` must skip `agent_memory` entries.

**Execution agent (`agent.py`):**
- `build_system_prompt_with_history()` uses the latest memory, the identifiers and a tail within
  `history_token_budget` (settings, default 4000). With no memory it's the full transcript, as today.
- The prompt's `Purpose:` uses the card's purpose from the directory, falling back to the name.

**After each execution:**
- In the batch manager, once the run completes, schedule `maybe_compact(agent_id)` as a background task.
- It runs only if the un-compacted prefix exceeds the budget, under a per-agent `asyncio.Lock`. It uses the
  `summarizer_model` setting, or the existing summarization model if the config has one, through
  `request_chat_completion`.
- On success it calls `record_memory` and `directory.update_card(purpose, last_status)`.
- On any failure it logs and keeps the full history; it retries after the next run.

**Tests:**
- splitting by budget, including when a memory already exists;
- identifier extraction on realistic Gmail tool payloads;
- compaction JSON parsing and failure cases;
- the render format;
- the system prompt with and without memory;
- `maybe_compact` skips under budget, records on success, keeps history on failure, and takes the lock;
- `load_transcript` ignores memory entries.

## Task 3: memory eval (L2 memory)

- **`evals/worlds/memory_gen.py`**
  - `build_histories(seed, n_agents=12, runs=40)` produces synthetic execution-agent logs: requests, tool calls
    with Gmail-like payloads, and responses.
  - **Planted facts** are placed at random early positions, so they fall into the compacted prefix:
    - 2 identifiers: a thread id and a draft id;
    - 1 email address;
    - 1 user preference ("keep replies formal");
    - 1 decision ("we agreed on $4,200");
    - 1 commitment ("send the signed copy by Friday").
  - Each fact has a question and a gold answer string.
  - Deterministic; golden hash test.
- **`evals/layers/memory.py`**
  - **Compacted:** run the production `compact` on the prefix (via the eval's cached `OpenRouterLLM`, adapted to
    `complete(system, user)`), then `render_history`.
  - **Full:** the full transcript.
  - **QA:** for each question, ask the model with the execution agent's system prompt plus that history: "Answer
    from your history only: …". Grade by normalized substring match against the gold answer.
  - **Report per condition:** recall overall and by fact type, prompt tokens, compaction cost, and compaction
    failures.
- **CLI:** `python -m evals.memory_run --config evals/experiments/memory.yaml`. Config: model, seed, n_agents,
  runs, budget_tokens, budget_usd. Results go to `evals/results/<ts>-memory-dev.json`, with the same
  meta/validity/integrity conventions as `evals/run.py` (reuse its helpers).
- **Protection:** add the new files to `PROTECTED`.
- **Tests:** offline with `ScriptedLLM`. Cover generator determinism, facts landing in the prefix, grading
  normalization, and the end-to-end layer with a scripted compaction and QA.

## Task 4: measure (controller)

1. **Lifecycle:** check L1 offline with lifecycle on vs off. Then run `retrieved-lived` live, about $0.75, and compare
   `dormant_revival` and the overall result with v3.
2. **Memory:** run the memory eval, about $0.20.
3. Write journal entries and push.
