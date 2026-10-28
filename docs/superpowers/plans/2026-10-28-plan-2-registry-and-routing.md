# Plan 2: Agent registry and routing at scale

**Goal:** fix routing overload (F1/F2) per spec §2–3, measured on the lived-in world.

**Baseline to beat** (`20261028-101527-baseline-lived-dev.json`), pass@1 / pass^k / dup:

| Size | Result | Prompt tokens per call |
|---|---|---|
| lived-200 | 0.92 / 0.89 / 0.02 | 23.7k |
| lived-1000 | 0.85 / 0.71 / 0.11 | 56k |

**Targets:** for both sizes, pass@1 ≥ 0.92 and dup ≤ 0.03; at lived-1000, pass^k ≥ 0.85 and prompt tokens ≤ 18k.
The lived regression suite must not regress.

**Principle:** routing logic lives in pure server modules. The eval's new systems call those same functions over
an in-memory directory, so the eval measures production code. The baseline system is frozen to today's prompt and
tools, so its numbers never drift.

**Conventions:** as in Plan 1 and 1b. Use `.venv/Scripts/python` and offline tests, and match the local code style:
`server/` uses the services pattern (`services/triggers/` is the model to follow). Don't touch `.serena/`,
`pyproject.toml` or `uv.lock`. Subagents don't commit or run `evals.integrity --update`; the controller does both.

## Task 1: pure routing core (`server/services/agents/`)

No I/O. Everything takes `now: datetime` explicitly.

- **`cards.py`**
  - `AgentCard(id, name, purpose, entities: list[str], last_status: str, status: str, created_at, last_active_at, run_count)`.
  - `extract_entities(texts) -> list[str]`, lowercased and deduped. It takes email addresses and capitalized
    tokens that aren't sentence-initial stopwords or common words: a small stoplist (I, The, Can, Please, Monday…),
    plus months and weekdays. Example: "Draft a reply to Gary Whitman about the lease" → `["gary", "whitman"]`.
  - `normalize_name(name)`: lowercase, strip punctuation, collapse spaces.
  - `slug_id(name, taken)`: a slug, with a `-2`, `-3`… suffix on collision.
- **`retrieval.py`**
  - `RetrievalConfig`: w_kw=0.45, w_ent=0.40, w_rec=0.15, recency_days=7, last_routed_boost=0.15, k=8, hot_max=5,
    hot_window_hours=1, min_score=0.1, fuzzy_min_len=4, fuzzy_weight=0.5.
  - `tokenize`.
  - BM25 implemented in-repo, over `name + purpose + last_status`, normalized by the max in the candidate set.
  - Entity overlap: query tokens against card entities and name tokens. Fuzzy: tokens of at least
    `fuzzy_min_len` chars within edit distance 1 count at `fuzzy_weight`.
  - Recency: `exp(-age_days / recency_days)`.
  - `score_cards(query, cards, now, last_routed, cfg) -> list[tuple[AgentCard, float]]`.
  - `select(query, cards, now, last_routed, cfg, pinned=(), open_work=()) -> list[AgentCard]`: pinned first, then
    the hot set (open work, or active within `hot_window_hours`, capped at `hot_max`, most recent first), then the
    top K by score among non-archived cards with score ≥ min_score. Deduped.
  - `search(query, cards, now, cfg, limit=10)` covers all cards, including archived ones.
  - `build_query(message, recent_texts) -> str`: the new message plus the last 3 turns.
  - `pinned_from_agent_message(message, cards)`: parses `[SUCCESS|FAILED] <name>:` lines into the matching cards.
- **`render.py`**
  - `render_cards(selected, total, now) -> str` produces the following, with relative times ("2h ago",
    "3d ago") and the purpose and status truncated to about 160 chars:

    ```
    <active_agents shown="N" total="M">
      <agent id="…" name="…" last_active="…">
        purpose: …
        status: …
      </agent>
    </active_agents>
    ```

  - Empty: `<active_agents shown="0" total="M">None</active_agents>`.
- **`dedup.py`**
  - `find_duplicate(name, purpose, cards, now, cfg) -> AgentCard | None`. A card is a duplicate when normalized-name
    similarity (difflib ratio) is at least 0.85, or when it shares a non-trivial entity and has a
    `score_cards(purpose)` score of at least 0.5.
  - Archived cards are included.
- **`directory.py`**
  - `AgentDirectory` protocol: `list_cards()`, `get(id)`, `create(name, purpose, instructions, now) -> AgentCard`,
    `touch(id, instructions, now)`, `last_routed() -> list[str]`, `set_last_routed(ids)`.
  - `InMemoryDirectory(cards)` implements it.
  - `create` extracts entities from the name, purpose and instructions; `touch` adds entities from the new
    instructions and updates `last_active_at`/`run_count`.
- **`dispatch.py`** holds the tool semantics, shared by the server and the eval.
  - `dispatch(directory, *, instructions, agent_id=None, name=None, purpose=None, force_new=False, now, cfg) -> DispatchResult(action, card, payload)`:
    - `agent_id` given and known → `touch`, action `reused`.
    - `agent_id` unknown → action `error`, payload listing the nearest 3 by search.
    - Otherwise both `name` and `purpose` are required, else `error`. Unless `force_new`, `find_duplicate`
      → `possible_duplicate`, whose payload has the candidate's card text and an instruction to reuse via
      `agent_id`, or to resend with `force_new=true` if it's truly different work.
    - Otherwise `create`, action `created`.
  - `search_payload(directory, query, now, cfg)` returns the rendered results.
- **`schemas.py`:** tool schemas for the new tools, so the IA wiring and the eval share them.
  - `send_message_to_agent(instructions, agent_id?, name?, purpose?, force_new?)`, with `instructions` required.
    Descriptions say to reuse with `agent_id` and to create with `name` and `purpose`.
  - `search_agents(query)`.
  - Plus the unchanged `send_message_to_user`, `send_draft` and `wait`, copied from today's `TOOL_SCHEMAS`.
- **`server/agents/interaction_agent/system_prompt.md`:** replace the agent-routing guidance with the reuse policy
  from spec §3. Keep everything else verbatim.
  - The roster shows only the most relevant agents (`shown` of `total`).
  - Reuse with `agent_id`.
  - Same person and same topic → reuse. Same person, new topic → new agent (`force_new` if the guard fires).
    Several tasks → one dispatch each.
  - An ambiguous person with no disambiguating context → ask the user, never message several agents to cover
    it.
  - If nothing listed fits but it sounds like ongoing work → `search_agents` before creating.
  - On `possible_duplicate`, reuse that agent unless the topic truly differs.
- **Tests** (`server/tests/test_agents_*.py`):
  - entity extraction;
  - BM25 ordering;
  - fuzzy "baetriz" → "beatriz";
  - the recency, last_routed, pinned and hot-set caps;
  - min_score and archived exclusion;
  - render format, plus the empty case;
  - the dedup name and entity rules;
  - dispatch: reuse, unknown id, missing name/purpose, possible_duplicate, force_new, created entities;
  - schema shape.

## Task 2: SQLite registry and server wiring

- **`server/services/agents/registry.py`**
  - `SqliteAgentDirectory(path)` implements `AgentDirectory`, following the `services/triggers/` store pattern.
  - Table `agents` per spec §2. `last_routed` is kept in a small `meta` table.
  - `get_agent_directory()` is a singleton at `server/data/agents.db`.
- **Migration:** on first use, if `roster.json` exists and the registry is empty, import each name with
  `purpose = name` and entities from its log. Then rename the file to `roster.json.migrated`.
- **IA wiring:**
  - `tools.py` uses `schemas.py`.
  - `send_message_to_agent` calls `dispatch(get_agent_directory(), …)`, then starts the execution for `reused`
    or `created`, keyed by the card id.
  - Add `search_agents`.
  - The `handle_tool_call` error behaviour stays unchanged.
- **`agent.py`:**
  - `prepare_message_with_history(latest_text, transcript, message_type)` builds the query from the latest text
    and the last 3 turns of the conversation log.
  - On agent messages it pins via `pinned_from_agent_message`.
  - It renders with `render_cards(select(...))`.
  - It updates `set_last_routed` after a user turn's dispatches; do this in the runtime, where dispatches are
    known.
- **Execution side:**
  - Logs, batch entries and triggers are keyed by the agent **id**; the display name is kept for prompts and the
    `[SUCCESS] <name>:` payload. This fixes the F2 slug collision.
  - `ExecutionAgentRuntime(agent_id, name)`.
  - The trigger store gets an `agent_id` column. When an old row has a name only, it resolves through the
    directory by name.
- **Clean-up:**
  - Delete `server/services/execution/roster.py` and `server/tests/test_roster_lock.py`. The `fcntl` import goes
    with them.
  - Update the exports.
  - `routes/chat.py` clear also clears the registry.
- **Tests:**
  - registry CRUD, id collisions and the migration;
  - `last_routed` persistence;
  - the `send_message_to_agent` tool with a temp directory and a stubbed batch manager: created, reused, duplicate,
    with execution started only for created and reused;
  - `prepare_message_with_history` renders cards and pins on agent messages;
  - trigger lookup by id.

## Task 3: eval routing layer accepts system-provided prompt, tools and handler

- `RoutingSystem` gains `system_prompt() -> str`, `tool_schemas() -> list`, and `new_turn(world, case) -> TurnHandler`.
  `TurnHandler.handle(name, args, decision) -> dict` records the effects on the `Decision` and returns the tool
  result.
- `evals/layers/routing.py` uses those instead of production's `build_system_prompt` and `get_tool_schemas`. Keep the
  argument-error shims: each system declares its tool signatures for the TypeError text.
- **Freeze the baseline:**
  - Copy today's prompt (from `git show HEAD:server/agents/interaction_agent/system_prompt.md`) to
    `evals/systems/frozen/baseline_system_prompt.md`.
  - Copy today's `TOOL_SCHEMAS` (from HEAD's `tools.py`) to `evals/systems/frozen/baseline_tools.json`.
  - `BaselineSystem` uses them, and its handler is the current `_Turn` logic. Take these from HEAD, not the
    working tree, because Task 1 edits the prompt in parallel.
- **Check:** rerunning `baseline-lived.yaml` and `baseline.yaml` must produce byte-identical cache keys, so every call
  is a cache hit, with spend $0.00 and the same summaries. Add an offline test that the frozen prompt and schemas
  hash to fixed values.
- Add the frozen files to `PROTECTED`.

## Task 4: eval systems `retrieved` and `full_cards`

- **`evals/systems/cards_from_world.py`:** `cards_for(world)`.
  - Card fields are `id`, `name` and `purpose`. Entities come from `extract_entities([name, purpose, *history
    requests])`. `last_status` is `""`. `created_at`/`last_active_at` are `world.now - days`, and status is
    `active`.
  - It must never read `threads`, `person`, `topic`, `core` or `roles` (gold). A test enforces this by poisoning
    those fields.
- **`retrieved` system:**
  - The prompt is the server's `build_system_prompt()`, and the schemas come from `schemas.py`.
  - The roster is built with `select(build_query(case.message, last 6 context texts), …)`. `last_routed` is the
    cards active within 1h of `world.now`, a stand-in for the previous turn's routing. On agent messages it pins.
  - The handler runs `dispatch(...)` over an `InMemoryDirectory` copy, made fresh for each turn:
    - `reused` → `decision.dispatched_ids`;
    - `created` → `decision.created_names`;
    - `possible_duplicate` and `error` → no effect;
    - `search_agents` → `search_payload`.
  - `shown_ids` returns the selected ids for L1.
- **`full_cards` system:** the same as `retrieved`, but it shows every card. This is the ablation that separates cards
  from retrieval.
- Register both in `SYSTEMS`.
- **Tests:** offline with `ScriptedLLM`. Check reuse by id, create, the duplicate guard path (possible_duplicate then
  force_new), search then dispatch, pinning on agent messages, L1 recall on a toy world, and that gold fields are
  never read.
- Configs `evals/experiments/retrieved-lived.yaml` and `full-cards-lived.yaml`: same settings as baseline-lived.

## Task 5: measure and iterate (controller)

1. Run `retrieved-lived` and `full-cards-lived`, plus `retrieved` on the synthetic world as an internal check.
2. Error analysis on the failures. One experiment per hypothesis, as journal entries: for example K, weights, the
   dedup threshold, prompt wording. Dev split only.
3. Stop when the targets are met or after about 4 experiments. Add the fixed cases to the regression suite. Push.
