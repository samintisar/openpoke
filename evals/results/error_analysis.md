# Baseline error analysis

- **Runs:** `20261028-074635-baseline-dev.json` (first grading) and
  `20261028-075013-baseline-dev.json` (regraded after the eval fix below; same cached responses, $0).
- **Model:** `anthropic/claude-haiku-5.5`, k=3, roster sizes 10 / 50 / 200 / 1000 plus handwritten.
- **Sampled:** all failing samples were read in full: 49 in the first grading, 18 after the fix.

## First pass: an eval bug, not a model failure

33 of the 49 first-pass failures were `agent_result` cases graded `unnecessary_clarify`. Every
agent-result template asks for something only the user can supply ("asked for the final
numbers", "need a signature by Monday"), and the model relayed it with a question such as
"How many people are coming?". That is the right behaviour. The cases now set
`clarify_ok: True` (commit "Let agent-result cases accept relaying the question to the user").
The regraded run is the baseline from here on.

## Remaining failures (regraded run)

| Category | Definition | Count | Sizes | Example case ids |
|---|---|---|---|---|
| Same person, new topic, reused | A new task for a known person goes to that person's generic agent ("Email to Ben") instead of a new one | 10 | 10, 50, 200, 1000 | `same_person_new_topic-s1-01` |
| Same topic, different person, reused | "reach out to Theo Brooks about the wedding RSVP" goes to "Wedding RSVP with Victor" | 3 | 10 | `new_task-s1-02` |
| Guessed between two people | "did Alice get back to me?" with two Alices: picked one instead of asking | 3 | handwritten | `hw-06-same-name-people` |
| Vague reference, wrong agent | "any reply yet?" went to "Email to Ben (Oct)" instead of the Ben offsite agent | 1 | 1000 | `vague_reference-s1-00` |
| Fan-out on an agent result | Routed the result to its own agent and to a second related one | 1 | 10 | `agent_result-s1-01` |

18 failures of 282 samples. No duplicates, no errors, no turn hit the iteration cap.

## Biggest categories → Plan 2 targets

1. **Identity by name only (13 of 18).** The roster shows bare names, so "Email to Ben" looks like
   it covers anything about Ben, and "Wedding RSVP with Victor" looks like it covers any wedding
   RSVP. This is failure mode F2. Fix: agent cards with purpose and entities (person + topic), so
   the model can see that the gala is not this agent's job, plus a create-time dedup guard that
   compares entities rather than strings.
2. **Ambiguity is not surfaced (3).** Two agents match a person name and the model guesses.
   Fix: when retrieval returns two close matches on different entities, show both cards with
   their distinguishing entities. Cheap, and the cards in (1) give the model what it needs to ask.
3. **Cost grows linearly with roster size.** Mean prompt tokens per call: 10.2k at n=10,
   11.3k at n=50, 18.0k at n=200, **50.7k at n=1000**. That's 5× the tokens for the same accuracy.
   Fix: show the top K=8 plus a hot set of up to 5 instead of the whole roster; the target is
   n=1000 prompts at about n=10 size.

## Surprises

- **Accuracy does not fall as the roster grows** (pass@1 0.89 / 0.95 / 0.95 / 0.97). Haiku 5.5
  reads a 1,000-name roster well when names are descriptive. On this world, overload shows up
  as **cost and consistency** (pass^k drops from 0.95 to 0.90 at n=1000), not as wrong agents.
  Plan 2 therefore has to keep accuracy while cutting tokens. A harder world (opaque names,
  more same-person agents) is needed to stress routing quality itself.
- The n=10 roster scores lowest: with few agents, a loosely related agent ("Wedding RSVP with
  Victor") is more likely to be the only plausible match, so the model reuses it.
- Retrieval MRR for the baseline is meaningless (it shows every agent in creation order); recall
  is trivially 1.0. Both become informative once Plan 2's retriever ranks agents.
- A free model (`nemotron-3-ultra:free`, smoke only) looped on `wait` in about 40% of turns until
  the 8-round cap. Production raises an error there. That's the iteration-limit gap, measured.
