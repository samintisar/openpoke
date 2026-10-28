# Lived-in world: baseline error analysis

- **Run:** `20261028-101527-baseline-lived-dev.json` (baseline system, `anthropic/claude-haiku-5.5`, k=3, $1.00).
- **World:** `evals/worlds/lived/world-s1.json`. The roster was grown by the IA from 789 life events, with
  the IA's own agent names; probes are held-out next events with real conversation history. The
  1,000-agent size uses **the same probes** as the 200 size, so those two columns compare like for like.
- **Sampled:** all 37 failing samples (excluding handwritten), read in full.

## Headline

| Size | pass@1 | pass^k | duplicate rate | prompt tokens/call |
|---|---|---|---|---|
| 10 | 0.87 | 0.86 | 0.00 | 13.8k |
| 50 | 0.93 | 0.90 | 0.06 | 15.0k |
| 200 | 0.92 | 0.89 | 0.02 | 23.7k |
| **1000** | **0.85** | **0.71** | **0.11** | **56.0k** |

From 200 to 1,000 agents, on the same probes, pass^k falls 0.89 → 0.71 and the duplicate rate rises
0.02 → 0.11, while prompt tokens grow 2.4×. The synthetic world did not show this. Its tidy
names let the model find the right agent at any size.

## Failure categories

| Category | Definition | Samples | Sizes | Example |
|---|---|---|---|---|
| Near-duplicate created | The right agent exists but the IA creates a new one with a similar name | 8 | 50, 1000 | "ask kwame if they need me to bring anything to book club": made "Kwame Book Club Bring Item Note" beside "Kwame Book Club Dates"; "tell maya that nora signed off on the OKRs": made "Maya Nora OKR Signoff Note" beside "Maya Quarterly OKRs Draft" |
| Same person, new topic, reused | A new task for a known person goes to that person's existing agent | 7 | 10, 50 | "tell elena i'm excited about the moving help" went to "Elena Christmas Gifts" |
| Ambiguous name not asked | "check in with ben" with two Bens: created "Ben Status Check" or messaged all six Ben agents | 6 | 200, 1000 | `lived-s1-200-same_name_people-00` |
| Asked about an obvious typo | "any update from baetriz on trivia night?" asks "did you mean Beatriz?" instead of using "Beatriz Trivia Night Note" | 6 | 200, 1000 | `lived-s1-200-typo-*` |
| Fan-out on a vague reference | "any update on that?" went to the right agent plus an unrelated one | 3 | 10 | `lived-s1-10-vague_reference-*` |
| **Label noise:** writer blended a multi | The writer attached the errand to the person ("ask pablo if the prague hotel is still under our name"), so reusing the person's agent is reasonable | 5 | 10, 200, 1000 | Pablo/Prague, Owen/Denver |
| **Label noise:** muddled roster | During growth the IA filed Marco's ski trip in "Marco Book Club Note"; creating "Marco Ski Trip Update" is defensible | 2 | 50 | `lived-s1-50-typo-*` |

7 of 37 failing samples (19%) are label noise. They stay in the set: removing them by hand would bias
the benchmark toward what the baseline gets right. They are named here so that a fix's gain is never
attributed to them.

## Plan 2 targets

1. **Near-duplicates at scale (F1/F2).** The IA scans ~1,000 bare names and misses close matches. Fix:
   retrieve the top K=8 cards by person and topic, plus the hot set, so the right agent is near the top
   of a short list. Add a create-time dedup guard: same person plus overlapping topic words means
   "reuse, or confirm new". Success means the 1000 column matches the 200 column.
2. **Identity beyond the name.** "Elena Christmas Gifts" looks like Elena's agent. Fix: cards that
   show purpose and entities, so a new topic for a known person is visibly not that agent's job.
3. **Ambiguity.** When a first name matches two people, the cards for both should be shown, plus a
   reuse policy in the prompt: ask, don't broadcast.
4. **Typos.** Fuzzy name matching in retrieval puts "Beatriz" cards in front of "baetriz", so the
   model has no reason to ask.
5. **Tokens.** 56k per call at 1,000 agents. Top-K retrieval should bring this near the 10-agent size.

## Surprises

- n=10 scores below n=50 and n=200. With few agents, a person's only agent ("Elena Christmas Gifts") is
  the obvious match for anything about that person, so same-person new topics get reused.
- The IA sometimes broadcasts to every agent matching a name (six Ben agents at once) rather than asking.

## After Plan 2 (`20261028-111346-retrieved-lived-dev.json`)

| Category | Samples | Note |
|---|---|---|
| Same person, new topic, reused | 6 | Includes the debatable Gavin case (role vs signing bonus of one offer) |
| Vague reference, wrong agent or duplicate | 6 | "house coverage" vs "Amy Mia Coverage Next Week" (name mismatch); "any update on that?" sent to Paul with Monica's card shown first |
| Typo created a duplicate | 3 | "mraco … ski trip": the muddled-roster label (ski trip filed under "Marco Book Club Note") |
| Multi-task partial | 2 | Pablo/Prague, where the writer blended the two requests (label noise) |
| Asked when the answer was clear | 2 | Noah export issue at 1,000 (another person has an export-issue agent) |

Near-duplicates at scale, the Plan 2 target, are gone: the duplicate rate is 0.00 at 200 and 1,000 (was 0.02 and 0.11).
