# Agent overload v2: a trustworthy benchmark and targeted routing fixes

Date: 2026-10-29. Builds on [2026-10-27-agent-overload-design.md](2026-10-27-agent-overload-design.md); everything there that isn't changed here stays.

## 1. Why v2

The final v1 test run (journal, 2026-10-29) gave routing pass@1 of 0.84 at 200 and 1,000 agents.

Retrieval beat the original OpenPoke by 8–9 points at 50 and 1,000 and tied it at 200. With about 32 cases per size, those intervals touch zero.

Error analysis found two causes.

1. **The benchmark was unreliable.**
   - Dev and test had different mixes. At 200 agents, dev had 1 vague-reference case and test had 6.
   - Both splits came from one simulated life.
   - Answer keys were picked by the model that wrote the probe. Several don't hold up. "how's the parking spot thing going?" is keyed to a lease-call agent that never mentions parking. "wedding rsvp, doyle side" is keyed to an agent checking whether Megan attends.
   - Scoring allowed one answer only, so asking the user on an unclear message counted as wrong.
2. **The system has real weaknesses.**
   - **Cards go stale.** A card's `purpose` is the first request, and `last_status` only refreshes when memory compaction runs.
   - **Status questions spawn agents.** "how's X going?" creates a new "status check" agent instead of finding the matter or asking.
   - **No second look.** Nothing checks a create or reuse decision at the moments where mistakes happen.

Full cards (every card shown) failed the same cases. So the misses are in choosing among shown cards, not in retrieval.

## 2. Goal and scope

Fix agent overload properly and prove it on a benchmark that can be trusted.

**In scope:**
- benchmark v2;
- living summaries on agent cards;
- a status-question policy;
- a second-look check at risky routing moments;
- conversation-aware retrieval for vague questions;
- removal of unproven pieces;
- prompt caching in the eval client;
- moving evals to DeepSeek V4.1 Flash.

**Out of scope:**
- **A matter layer** (a separate table, several agents per matter, merges, handover). It was considered, and the decision record is in §8.
- **Embedding retrieval.** It is weak on names, IDs and typos, and v1's misses were not retrieval misses. Revisit only if §3.6 shows paraphrase recall below 0.97.
- **Production changes for cost:** server prompt caching, and moving the cheap tier off its current model.

**Optional, last:** an external check on the Ubuntu IRC disentanglement corpus (§3.7).

## 3. Benchmark v2

### 3.1 Worlds

- **Life 1:** the existing seed-1 lived world. Dev only.
- **Life 2:** a new persona with a different life plan, grown the same way through the real interaction agent. Test only.
  - Its probes are label-checked (§3.4), but no system runs on it until the final run.
  - It is added to `evals/PROTECTED.sha256` once reviewed.
- **Sizes:** 10 (smoke), 50, 200 and 1,000. As now, 1,000 is the 200 roster plus background errands, and 200 and 1,000 share probes.

### 3.2 Probes, stratified

The same quota per kind in both lives:
- about **10 per kind at 200** (shared with 1,000);
- about **5 per kind at 50**;
- 2–3 per kind at 10.

That gives about 180 probes per life.

**Kinds:**
- follow_up
- new_task
- same_person_new_topic
- vague_reference
- **status_question** (new: "how's X going?", "any word on X?")
- multi_task
- same_name_people
- cross_agent
- dormant_revival
- typo
- agent_result
- **paraphrase** (new: refers to a matter in words that appear nowhere on its card)

A test fails if dev and test kind counts differ by more than one per size.

### 3.3 Answer keys from construction

A probe is built from a **matter** in the life plan: a person or people plus a subject, with its events.

The writer model gets the matter's facts and history, not an agent card. The message must refer to something that happened in that matter's history.

**Answer key:**
- **Best:** the agent that handled the matter's events during growth. This comes from the growth log.
- **Acceptable:**
  - any other agent that handled events of the same matter (duplicates made during growth);
  - **ask**, when the plan has another live matter with the same person and an overlapping subject;
  - **new**, when the matter has no agent yet.
- Probes whose matter can't be resolved to an agent in the growth log are dropped, not guessed.

### 3.4 Verification and human review

1. Three checker models each choose best, acceptable or wrong for the key. Each sees the message, the last 6 turns and the top 10 cards (with summaries). The checkers are:
   - `openai/gpt-6-luna`
   - `google/gemini-3.1-flash-lite`
   - `qwen/qwen3.8-flash`

   They come from three families, none of them the tested model.
2. **All three agree with the key:** the probe is kept.
3. **Any disagreement:** the probe goes to human review.
4. **Random sample:** a 10% random sample of kept probes also goes to human review. Its error rate is reported as the benchmark's label error rate.

Review uses the existing probe review page, extended to show each candidate card's purpose, summary and recent history side by side. Expected load is 40–60 probes, about 30–45 minutes.

Human decisions follow v1: accept, fix (best / acceptable / ask allowed) or reject.

### 3.5 Scoring

Each sample gets one outcome:

| Outcome | Meaning | Severity |
|---|---|---|
| correct | the best answer | — |
| acceptable | in the acceptable set, including ask where allowed | — |
| wrong_reuse | sent to an agent outside best and acceptable | **harmful** |
| duplicate_create | created an agent when one in best or acceptable existed | **costly** |
| unneeded_ask | asked when ask wasn't acceptable | minor |
| other | anything else, such as no dispatch when one was required | minor |

**Reported per size:**
- correct-or-acceptable rate;
- harmful rate;
- duplicate rate;
- pass^k (k=3);
- prompt tokens per decision;
- second-look trigger rate.

Every rate gets a 95% bootstrap interval over cases (`evals/charts.py` method). Comparisons between systems use paired bootstrap differences.

### 3.6 Retrieval check (L1, free)

recall@shown on both lives' dev probes, reported per kind. Paraphrase and status_question are reported on their own.

If recall@shown on v2 dev is below 0.97, embedding retrieval comes back into scope as a separate decision.

### 3.7 External check (optional, last)

A subset of the Ubuntu IRC disentanglement corpus (CC-BY 4.0, human-labelled reply links), recast as "continues an existing thread or starts a new one". It scores the assign-or-create step against human labels.

Only after everything else lands.

### 3.8 v1 retirement

- The v1 test split and round-2 probes are no longer a test set.
- v1 results stay in the journal, labelled v1.
- The write-up reports v1 and v2 separately.

## 4. System changes

### 4.1 Living summaries on agent cards

- `AgentCard` gains `summary` (≤120 words). `purpose` stays the original request and is never rewritten.
- `summary` absorbs and replaces `last_status`. Memory compaction (`memory.py`) stops writing `purpose` and writes `summary` through the same updater.
- **When it updates:** after every execution-agent result (success, failure or partial), one cheap model call folds the result into the summary. It covers:
  - who is involved;
  - what has been decided;
  - what is pending;
  - topics raised.
- **Rules:**
  - Facts come from the result only.
  - Content inside `<untrusted_email>` is summarized as data, and instructions in it are never copied.
  - The 120-word limit is hard.
  - On a failed or invalid update, the previous summary stays.
- **Used by:**
  - retrieval: BM25 over name, purpose and summary; entity terms include people named in the summary;
  - rendered cards;
  - the duplicate guard.
- **Migration:** existing cards start with `summary = last_status`.
- **Eval worlds:** each lived world's agents get summaries by replaying their growth results through the same updater.

### 4.2 Status-question policy

An interaction agent prompt rule, plus the description of `search_agents`:
- A progress question ("how's X going?", "any word on X?", "did they get back?") is about something in progress.
- Find it among the shown cards or with `search_agents`.
- **If found:** answer from the card's summary when it holds the answer, otherwise dispatch to that agent.
- **If not found:** ask the user. Never create an agent only to check status.

### 4.3 Second-look check

A single cheap model call, outside the interaction agent loop. It sees:
- the message;
- the last 6 turns;
- the top 5 cards with summaries.

It returns `{"decision": "reuse"|"new"|"ask", "agent_id"?, "reason"}`.

**Triggers**, decided by code:
- **Create:** the interaction agent creates an agent and the duplicate guard didn't fire, but a shown card scores ≥ `second_look_min_score` against the request.
- **Reuse:** the interaction agent dispatches to agent A while another shown card shares a person entity with the message (the wrong-person risk).

**Effects:**
- **reuse:** the dispatch goes to that agent, and the interaction agent is told.
- **ask:** the dispatch is held, and the interaction agent is told to ask the user.
- **new:** proceed.
- **Error or timeout** (default 10 s): the interaction agent's decision stands, and the event is logged.

The duplicate guard stays as the free first line. `second_look_min_score` is tuned on v2 dev only.

### 4.4 Conversation-aware retrieval for vague questions

- The main query stays the message alone. v1 showed that adding context everywhere hurts recall.
- For messages classified as status or vague, a second query adds the last 3 turns, and up to 3 extra cards from it are merged in.
- Classification is by rule: a progress phrase, or no person entity in the message.

### 4.5 Ablation switches

Each of 4.1–4.4 can be switched off in the eval systems, following the existing `RetrievedSystem` switches. Each is measured on v2 dev against the full system.

### 4.6 Removals

All are removed with tests passing and recorded in the journal.

- **`last_routed_boost` in retrieval:** it can't change the shown set unless more than `hot_max` agents are hot, and that never happened.
- **Typo-tolerant entity matching (`fuzzy`):** removed if its ablation on v2 dev shows no gain. Otherwise kept.
- **The synthetic world generator** and the code and tests only it uses. Handwritten cases stay if they don't depend on it.

## 5. Model and cost

- **Evals:** `deepseek/deepseek-v4.1-flash` replaces Haiku 5.5 for every live role: interaction agent, execution agents, search, summary updates and the second look.
- **No Haiku anywhere** in evals or checkers.
- **The judge** stays `google/gemini-3.1-flash-lite`. Its calibration (v1 labels) is reported as made against Haiku-produced transcripts.
- **Production defaults are unchanged.** The write-up states that the measured model and the production default differ. In production, the new calls (summary updates, the second look) use the configured cheap-tier model setting.
- **Sanity gate before anything else:** DeepSeek on a small v1 dev slice (routing plus 3 L3 scenarios). It fails if:
  - more than 2% of tool calls are malformed;
  - more than 10% of turns loop on `wait`;
  - more than 2% of rows error.

  On failure, stop and choose a model with the user.
- **Prompt caching in the eval client:**
  - stable content first, marked cacheable: system prompt, tool definitions, conversation history;
  - changing content last: shown cards, the new message;
  - cached and fresh tokens recorded per run.
- **Cost guard:** an eval call over 100k prompt tokens fails unless the config sets `allow_long_prompts: true`.

## 6. Success criteria

Written before any v2 result. Measured once on life 2 with `--final`.

| Metric, at 1,000 agents | Target |
|---|---|
| Correct or acceptable | ≥ 0.95 |
| Harmful rate | ≤ 0.01 |
| Duplicate rate | ≤ 0.03 |
| Prompt tokens per decision | ≤ 25k |
| Second-look trigger rate | ≤ 0.20 |
| Gain over the original OpenPoke, correct or acceptable | paired 95% interval excludes zero |

The same metrics are reported at 50 and 200.

**Rules:**
- A system change is kept on v2 dev only if both hold:
  - Its target metric improves, with the paired 95% interval excluding zero, or with a better point estimate when the change is cheap.
  - No other headline metric gets worse by more than 0.02.

  Otherwise it is reverted.
- If targets are missed on life 2, the result is reported as is. Life 2 is never rerun after changes.

## 7. Order of work

1. **DeepSeek sanity gate.**
2. **Eval client:** prompt caching and the cost guard.
3. **Benchmark v2:**
   1. grow life 2;
   2. build matter-based, stratified probes for both lives;
   3. run the three checkers;
   4. human review;
   5. protect life 2.
4. **Re-baseline on v2 dev:** original OpenPoke, current retrieval, and full cards (≤200 agents).
5. **System changes on v2 dev, one at a time, each ablated:**
   1. living summaries;
   2. status policy;
   3. second look;
   4. conversation-aware retrieval;
   5. removals.
6. **Final runs:**
   - life 2, once;
   - L3 safety, gate on and off;
   - memory held-out;
   - all on DeepSeek.
7. **Optional:** the IRC check.
8. **Write-up and interview prep updated:** v1 → v2 story, results against §6.

## 8. Decision record

- **Matter layer vs living summaries on cards.**
  - Matters are the better long-term identity: handover, merges, and routing among matters rather than among every agent.
  - Their extra benefits can't be shown on this benchmark, and they double the code and risk.
  - The summary is a separate card field, so lifting it into a matter table later loses nothing.
- **No embeddings:** see §2 and §3.6.
- **Full human review vs targeted review:** the user's time is limited. Answer keys come from construction, three checkers filter, and a random sample measures label error.
- **One persona vs two:** separate lives for dev and test, against overfitting to one user's names and habits.
