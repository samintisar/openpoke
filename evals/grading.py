"""Grade a routing decision by outcome: which agents got work, not how the model got there.

Known limitations:
- multi_task can't tell a near-duplicate of the gold agent from the needed new agent: one
  create counts as the new task whatever it was named.
- Clarifying while also acting correctly isn't penalized.
- MISSED_CREATE is also used when the model dispatched to a wrong agent and also created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cases import Case

CORRECT = "correct"
WRONG_REUSE = "wrong_reuse"  # sent work to an agent that doesn't own it
DUPLICATE_CREATE = "duplicate_create"  # created a new agent when one should be reused
MISSED_CREATE = "missed_create"  # reused an agent when a new one was needed
EXTRA_CREATE = "extra_create"  # right agent, plus an unneeded new one
PARTIAL = "partial"  # multi-target case only half done
SHOULD_CLARIFY = "should_clarify"  # ambiguous request; guessing or doing nothing is wrong
UNNECESSARY_CLARIFY = "unnecessary_clarify"
NO_ACTION = "no_action"
ERROR = "error"


@dataclass
class Decision:
    """What one Interaction Agent turn did, as seen through intercepted tools."""

    dispatched_ids: list[str] = field(default_factory=list)
    created_names: list[str] = field(default_factory=list)
    clarified: bool = False
    user_messages: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_keys: list[str] = field(default_factory=list)


def grade(case: Case, decision: Decision) -> str:
    predicted = set(decision.dispatched_ids)
    distinct_created = len(set(decision.created_names))
    created = distinct_created > 0
    expected = set(case.expected_ids)
    acceptable = set(case.acceptable_ids)

    if not predicted and not created:
        if decision.clarified:
            return CORRECT if case.clarify_ok else UNNECESSARY_CLARIFY
        if case.dispatch_optional:
            return CORRECT
        return SHOULD_CLARIFY if case.clarify_ok and not case.gold_targets else NO_ACTION

    if not case.gold_targets:
        return SHOULD_CLARIFY
    if predicted - expected - acceptable:
        return WRONG_REUSE if expected else MISSED_CREATE
    if case.dispatch_optional:
        return DUPLICATE_CREATE if created else CORRECT

    unmatched = [g for g in expected if not predicted & ({g} | acceptable)]
    if unmatched:
        return DUPLICATE_CREATE if created and not case.expects_new else PARTIAL
    if created and not case.expects_new:
        return EXTRA_CREATE
    if case.expects_new and distinct_created > 1:
        return EXTRA_CREATE
    if case.expects_new and not created:
        return PARTIAL if expected else MISSED_CREATE
    return CORRECT
