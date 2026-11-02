"""L1 retrieval eval: is the right agent among what the system shows? No LLM."""

from __future__ import annotations

from typing import Any

from ..cases import Case
from ..systems.base import RoutingSystem
from ..worlds.model import World


def run_retrieval_case(case: Case, world: World, system: RoutingSystem) -> dict[str, Any]:
    shown = system.shown_ids(world, case)
    block = system.render_roster(world, case)
    recall: float | None = None
    mrr: float | None = None
    targets = case.expected_ids
    if targets:
        positions = {agent_id: index for index, agent_id in enumerate(shown)}
        hits = 0
        best: int | None = None
        for gold in targets:
            found = [positions[option] for option in (gold, *case.acceptable_ids) if option in positions]
            if found:
                hits += 1
                best = min(found) if best is None else min(best, *found)
        recall = hits / len(targets)
        mrr = 0.0 if best is None else 1.0 / (best + 1)
    return {"shown_count": len(shown), "roster_tokens": len(block) // 4, "recall": recall, "mrr": mrr}
