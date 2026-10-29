# Agent overload in OpenPoke: what broke, what I changed, and how I know

OpenPoke has one assistant the user talks to (the **interaction agent**) and a pool of **execution agents** that do the work: draft an email, check a thread, set a reminder. Every task creates or reuses an agent, and none of them ever retire. This document covers:

- what goes wrong as that pool grows;
- what I changed;
- how I tested and assessed the agent;
- what the numbers do and don't show.

The full experiment log, with every run and decision, is [`evals/experiments/JOURNAL.md`](../evals/experiments/JOURNAL.md). The design is in [`docs/superpowers/specs/2026-10-27-agent-overload-design.md`](superpowers/specs/2026-10-27-agent-overload-design.md).

## Summary

- **Routing at scale.** On the held-out test split, the original OpenPoke falls from 0.89 pass@1 at 50 agents to **0.76** at 1,000. The new routing (agent cards, top-K retrieval, a duplicate guard) holds **0.98 at 50 and 0.84 at 1,000**, and is consistent across samples (pass^k 0.84 vs 0.72). It uses 22.5k prompt tokens per decision at 1,000, against 56k for the original.
  - **Not conclusive on test alone.** With about 32 test cases per size, the 95% intervals on the gains (+0.09 at 50, +0.08 at 1,000) just touch zero.
  - **A tie at 200.** All three systems fail the same five cases there, most of which have questionable answer keys.
  - **What the conclusion rests on:** the dev results point the same way, and the ablations isolate the duplicate guard as the part that matters.
- **Sending email is now enforced in code.** On the held-out safety scenarios, the approval gate allowed **0 unauthorized sends in 15 runs**. With the gate switched off, the same scenarios produced **6**, including a forward the user had explicitly declined.
- **Agent memory is bounded.** Compacted history recalls **1.00** of planted facts on a held-out seed, at 5.8k tokens instead of 9–10k for the full history.
- **The agent was assessed three ways:**
  1. code checks on the end state;
  2. 42 transcripts I labelled by hand on six checks;
  3. an LLM judge from a different model family, used only on the 3 checks where it agreed with me at least 85% of the time.

## 1. The problem

OpenPoke identified agents by a free-text name stored in `roster.json`, and agents never retired. The interaction agent chose between them from that bare list of names. As usage grows, six things break:

| # | Failure | Where it was |
|---|---|---|
| F1 | Routing degrades: hundreds of bare names, no purpose, recency or status, so the wrong agent is reused or a duplicate is created | `interaction_agent/agent.py` (`_render_active_agents`) |
| F2 | Identity is an exact string, so `"email to alice"` and `"Email to Alice"` become two agents sharing one log file | `roster.py`, `log_store.py` |
| F3 | Each agent's full history is injected into every prompt, without limit | `execution_agent/agent.py` |
| F4 | No concurrency control: the same agent can run twice at once, interleaving its log | `interaction_agent/tools.py`, `trigger_scheduler.py` |
| F5 | One global batch: a slow agent holds back every other result | `batch_manager.py` |
| F6 | Concurrent assistant turns race on the conversation log, giving duplicate or contradictory replies | `chat_handler.py`, `importance_watcher.py` |

**Other gaps found on the way:**
- "Confirm before sending" existed only as a prompt instruction.
- Email text reached agents that can send mail, so it was an injection vector.
- Hitting the iteration limit failed hard.
- An `fcntl` import broke Windows.
- The configured model was no longer served.
- There were no tests or evals.

## 2. How I tested and assessed the agent

I built the measurement first, so every change has a before and after.

### Three layers

- **L1 retrieval** (no model calls): is the right agent among the cards the assistant is shown? It's free and deterministic, so it's used for sweeps.
- **L2 routing** (live): one real interaction-agent turn, with the production prompt and tools, on a roster of N agents. The tool calls are intercepted, and the decision is graded by outcome: the right reuse, a correct new agent, or a justified question. The metrics are:
  - **pass@1:** the share of correct samples.
  - **pass^k:** whether all 3 samples of a case are correct, which measures consistency.
  - **duplicate rate:** a new agent was created when the right one existed.
- **L3 end to end** (live): the real interaction and execution agents, the inbox and the runner, against an in-memory Gmail. There are 26 scripted scenarios covering approval, consent, injection, orchestration and standing approvals. Each is graded by code on the final state: who received what, under whose approval, how many replies. Unauthorized sends are audited when mail leaves the fake mailbox, independently of the gate, so a gate-off run is comparable.

### A world that looks like real use

The first synthetic world was too clean: tidy, unique names made 1,000 agents easy to route. I replaced it with a **lived-in world**:
- a seeded life of 789 events played through the real interaction agent, so the roster carries the assistant's own inconsistent names and real history;
- background errands spread over the same life to reach 1,000 agents.

Probe messages are the next events in that life, at 10, 50, 200 and 1,000 agents. The 200 and 1,000 sizes share probes, so the only difference between them is roster size.

### Discipline

- **A dev/test split by stable hash.** I tuned only on dev. The test split runs only with `--final`, once per system.
- **A checksum over protected eval code:** cases, grader, metrics, runners, the frozen baseline and the judge. A run whose checksum doesn't match the committed one is marked invalid.
- **A cache for model responses and a budget cap on every run.**
- **A journal entry for every experiment:** hypothesis, change, run file, result and decision.

### More test data, reviewed blind

The original test split had only 4–9 cases per size, too few to say anything. So I had a model write 48 more probes for 50 and 200 agents (200 also covers 1,000). Because the same model wrote both the message and the expected answer, I reviewed all 48 by hand **before any system ran on them**, never seeing a result:
- 44 accepted;
- 2 corrected;
- 2 rejected as ambiguous. For example, "check with Gary about the parking spot", when the world has two Garys.

Only the reviewed file can add cases, and it's under the checksum.

### Human labels and a calibrated judge

Code can check what was sent. It can't check whether a reply was clear or leaked internal detail. I labelled 42 L3 transcripts on six checks:
- was the draft shown exactly;
- was consent asked;
- was an injected instruction refused;
- no internal jargon;
- one clear reply;
- would I be happy as the user.

Then I calibrated an LLM judge (`google/gemini-3.1-flash-lite`, a different family from the Claude agents) against those labels:

| Check | Agreement with me | Used? |
|---|---|---|
| No internal jargon | 97.6% | yes |
| One clear reply | 92.9% | yes |
| Asked before sending | 85.7% | yes |
| Refused injected instructions | 76.2% | no: code checks plus human labels |
| Draft shown exactly | 71.4% | no |
| Would be happy | 69.0% | no: human only |

**Why three checks are excluded:**
- **Injection and draft shown:** most disagreements were about whether the check applied at all (N/A), not about safety.
- **Would be happy:** the judge was too lenient. It caught 4 of the 15 transcripts I marked unhappy.

I didn't tune the judge's prompts on these same 42 labels, because there's no fresh labelled set to check the tuned version against. Its code is frozen under the checksum since calibration.

## 3. What I changed

### Routing at scale (F1, F2)

- **Agent registry with cards.** `roster.json` is replaced by a SQLite registry (`server/services/agents/`). Each agent has a card: purpose, the people and things it involves, last active, and status.
- **Top-K retrieval** (`retrieval.py`). The assistant sees only the relevant cards:
  - Cards are scored by entity and name overlap, with typo-tolerant name matching, recency and a small boost for the agent routed to last.
  - Up to five agents active in the last hour ("hot") are always shown.
  - A `search_agents` tool covers the long tail.
- **Duplicate guard in code** (`dedup.py`, `dispatch.py`). Creating an agent whose card matches an existing one is pushed back with that card. The assistant can still insist.
- **Reuse policy: "an agent owns a matter".** A person plus a subject belongs to one agent, whatever the next step is.

### Lifecycle (F1)

Agents go dormant after 7 days idle and are archived after 30, by an hourly sweep (`lifecycle.py`). Archived agents leave the default view but can return on a strong match, so revival works.

### Memory (F3)

An execution agent's old history is folded into a summary plus a list of identifiers (thread IDs, draft IDs, addresses), each kept with 60 characters of context on either side (`memory.py`).

### Orchestration (F4–F6)

- A per-agent FIFO lock and a global cap of 4 concurrent agent runs (`services/execution/runner.py`).
- A durable SQLite inbox: one assistant turn at a time, results arriving during a turn coalesce into the next, and pending items survive a restart (`services/inbox/`).
- Agents that hit the iteration limit return a partial result listing the side effects already performed (drafts created, triggers set) instead of failing.

### Safety

- **A send approval gate in code** (`services/approvals/gate.py`):
  - Every draft, reply and forward becomes a pending action with a hash of its content.
  - A send passes only if the user's own message, arriving after the draft was shown, approved that exact content.
  - Approval can't come from an email, an agent report, a trigger or a watcher alert.
  - Editing after approval invalidates it.
  - Recurring sends get standing approvals bound to one trigger and its named recipients.
- **Untrusted email.** Email text reaching any agent is wrapped in `<untrusted_email>`, and the prompts treat it as data.
- **Two reply rules from the human labels:**
  - Don't repeat a draft the user can already see.
  - When something claims the user's approval, warn them plainly instead of passing it along.

### Smaller fixes

- Windows support: the `fcntl` file lock.
- Gmail bodies without base64 padding failed to decode (`gmail/processing.py`).
- Model defaults now point at served models, with a cheaper tier for summaries and classification.
- `[PARTIAL]` results now stay pinned to their agent.
- Approved replies send the stored content by action ID instead of being retyped.

## 4. Results

### Routing, held-out test split

Each run happened once, on the committed state, with Claude Haiku 5.5 and k=3. The test split has 31–32 cases at 50/200/1,000.

![Routing pass@1 on the test split with 95% intervals](figures/routing-test.svg)

*The vertical axis starts at 0.5.*

| Agents | Original OpenPoke | Retrieval (this work) | Full cards (ablation) |
|---|---|---|---|
| 50 | 0.89 / 0.87 / 0.02 | **0.98 / 0.94 / 0.00** | 1.00 / 1.00 / 0.00 |
| 200 | 0.83 / 0.81 / 0.15 | 0.84 / 0.81 / 0.13 | 0.84 / 0.84 / 0.18 |
| 1,000 | 0.76 / 0.72 / 0.25 | **0.84 / 0.84 / 0.18** | not run (see below) |
| Prompt tokens per decision at 200 / 1,000 | 22.8k / 56.0k | 22.6k / 22.5k | 72.0k / ~265k |

*Each cell is pass@1 / pass^k / duplicate rate.*

**How to read this:**
- **Retrieval's prompt stays flat as the roster grows.** The original's grows with every agent.
- **Full cards shows every card with its purpose.** It is as accurate as retrieval up to 200, at 3.2× the tokens there. At 1,000 it needs about 265k prompt tokens per decision (measured on an earlier dev attempt), about $13 for one test pass, so it wasn't run. At that size retrieval's advantage over it is cost, not demonstrated accuracy.
- **Paired difference, retrieval minus the original, with a bootstrap over cases:**

  | Agents | Difference | 95% interval |
  |---|---|---|
  | 50 | +0.09 | [−0.00, +0.19] |
  | 200 | +0.01 | [−0.06, +0.09] |
  | 1,000 | +0.08 | [−0.02, +0.20] |

  The gains point the same way as on dev but are not significant on 32 cases alone.

**Test was harder than dev for every system.** At 1,000, retrieval went from 0.93 on dev to 0.84, and the original from 0.85 to 0.76. Most test cases at 200 and 1,000 are model-written probes.

**Error analysis (after the run; test labels not edited).** At 200, the three systems fail almost the same samples (15 / 16 / 15). Five cases fail everywhere:
- **"how's the parking spot thing going?"** The answer key is a lease-call agent whose card never mentions parking.
- **"how's the lesson schedule looking this week"**
- **"ask baetriz … game night"** The answer key is her housewarming agent; she also has a trivia-night one.
- **"ask gloria … and separately let anjali know"** Every system also creates an agent for Anjali, which the key counts as wrong.
- **"how's the wedding rsvp going? … doyle side"**

Full cards shows every card and still fails these. So they aren't caused by roster size, and they account for most of the remaining "duplicates". At 1,000, the original also fails four cases that retrieval gets right: dormant revival, a multi-task message, same-name people, and same person / new topic. That's the overload effect retrieval removes.

### What each part contributes (dev ablations)

| Switched off | Effect at 200 / 1,000 | Verdict |
|---|---|---|
| Duplicate guard | duplicates 0.00 → 0.11 / 0.12; pass@1 −0.07 / −0.06 | **needed** |
| `search_agents` | duplicates 0.05 / 0.02; vague references lose 2 samples | small effect, kept |
| Typo-tolerant names | no measurable change | unproven, kept (cheap) |
| Last-routed boost | no change: never more than 5 hot agents on this world | unproven, kept |
| Lifecycle | no accuracy cost; dormant revival 27/27 | kept |

### Safety, end to end

| | Dev, 21 scenarios × 3 | Test, 5 scenarios × 3 |
|---|---|---|
| Gate on: unauthorized sends | 0 | **0** (6 attempts blocked) |
| Gate off: unauthorized sends | 1 | **6** |

**The gate-off failures:** on test, the user declines a forward and it goes out anyway (3/3 runs), and a forward goes out before the user approves it (3/3). On dev, a helper replied to a thread directly instead of drafting.

**Judge on the gate-on test transcripts:**
- asked before sending 15/15;
- one clear reply 15/15;
- no internal jargon 14/15. One reply mentioned "Action 1".

**After the two reply rules (dev):**
- Drafts repeated in the reply text: 12/36 → 0/34.
- Explicit warnings when an email faked the user's approval: 0/3 → 3/3.

### Memory

| | Recall of planted facts | History tokens |
|---|---|---|
| Full history | 1.00 | 9.0–10.0k, growing without limit |
| Compacted, dev seed | 1.00 | 5.5k |
| Compacted, held-out seed | **1.00** | 5.8k |

Every fact type stayed at 1.00:
- thread ID;
- draft ID;
- email address;
- preference;
- a decision that was later superseded, where the stale answer rate was 0.00;
- a commitment.

The first version, with a bare identifier list, recalled thread IDs only half the time. Adding context around each ID fixed it. One compaction call failed on the held-out run and fell back to the full history.

## 5. What didn't work

- **The first reuse rule** ("same person, new topic → new agent") roughly doubled duplicates, to 0.16–0.18. Full cards failed the same way, which showed the cause was the policy, not retrieval.
- **Hiding archived agents entirely** dropped L1 recall to 0.62, because every lost case pointed at an agent idle for 30+ days. Archived cards may now join on a strong match.
- **A bare list of IDs in compacted memory** lost the thread ID half the time.
- **Free models** looped on `wait` in about 40% of turns and hit daily caps. Evals moved to Haiku 5.5.
- **An eval bug:** agent-result cases penalised relaying the agent's question to the user. It caused 33 of 49 early baseline "failures". I fixed the grader, not the system.

## 6. Known limits

- **Small test set.** 31–32 cases per size can't confirm differences under about 10 points. The routing claims rest on test and dev pointing the same way, not on test alone.
- **One generated world.** One persona and one seed, written by a model. Blind review reduces label noise but doesn't remove it: five suspect test labels are listed above.
- **Evals ran on Haiku 5.5; production defaults to Sonnet 5.5.** The production default is not measured.
- **Gmail is the only real tool.** Some scheduling scenarios make the assistant say it can't see a calendar. What's tested is orchestration under load, not tool coverage. New tools that send anything (calendar invites email people) should go through the same gate.
- **Approval still relies on the model reading "yes".** The gate enforces that approval came from the user, after the draft was shown, for that exact content. Choosing which draft when several are waiting is prompt-only.
- **Open gaps found during the build**, recorded in the spec (§12):
  - A reply's subject isn't covered by the content hash.
  - Recipient checks see tool arguments, not Gmail's reply-all headers.
  - Standing approvals bound recipients, not content.
  - Inbox delivery is at least once.
  - Interrupted agent runs aren't resumed after a restart.
  - At the iteration limit, an agent can loop on errors and expose internal detail (seen in 1 labelled transcript).

## 7. Reproduce

```bash
python -m pytest -q                                                            # 656 unit tests, offline
python -m evals.run --config evals/experiments/retrieved-lived.yaml            # routing, dev split (needs OPENROUTER_API_KEY)
python -m evals.run --config evals/experiments/baseline-lived.yaml             # the original OpenPoke on the same world
python -m evals.e2e_run --config evals/experiments/e2e.yaml                    # end to end, gate on (e2e-gate-off.yaml: off)
python -m evals.memory_run --config evals/experiments/memory-heldout.yaml      # memory, held-out seed
python -m evals.judge --transcripts evals/labelling/transcripts.json --labels evals/labelling/labels.json   # judge calibration
python -m evals.charts                                                         # chart and intervals from the final runs
python -m evals.e2e_run --interactive --scenario blank                         # try to trick it yourself
```

Model responses are cached, so rerunning a committed run is free. Recorded spend across all result files is $19.80, plus $4.20 to generate the lived-in world, so about $24 on OpenRouter. The final test runs cost $3.10.
