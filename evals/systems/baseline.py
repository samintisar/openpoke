"""Today's behaviour: every agent ever created, as a bare name, matched by exact string.

The prompt and tool schemas are frozen copies of production's at the time Plan 2 started
(frozen/), so baseline numbers never drift when the server's prompt changes.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Callable

from ..cases import Case
from ..grading import Decision
from ..worlds.model import World

FROZEN_DIR = Path(__file__).parent / "frozen"
# Read like production's agent.py: read_text(...).strip().
_SYSTEM_PROMPT = (FROZEN_DIR / "baseline_system_prompt.md").read_text(encoding="utf-8").strip()
_TOOL_SCHEMAS = json.loads((FROZEN_DIR / "baseline_tools.json").read_text(encoding="utf-8"))


# No-op shims with the IA tools' signatures and names: calling one with the model's arguments
# raises the same TypeError, word for word, that production's tool(**args) raises.
def send_message_to_agent(agent_name: str, instructions: str) -> None:
    return None


def send_message_to_user(message: str) -> None:
    return None


def send_draft(to: str, subject: str, body: str) -> None:
    return None


def wait(reason: str) -> None:
    return None


_SHIMS = {f.__name__: f for f in (send_message_to_agent, send_message_to_user, send_draft, wait)}


class BaselineTurn:
    """Per-run tool state: the stub has to remember agents created earlier in the turn."""

    def __init__(self, world: World, system: BaselineSystem) -> None:
        self.world = world
        self.system = system
        self.created: set[str] = set()

    def handle(self, name: str, args: dict[str, Any], decision: Decision) -> dict[str, Any]:
        """Record a valid call's effect and return the stubbed tool result."""
        if name == "send_message_to_agent":
            agent_name = str(args.get("agent_name", ""))  # raw, like production: no normalization
            agent_id = self.system.resolve_name(self.world, agent_name)
            is_new = agent_id is None and agent_name not in self.created
            if agent_id is None:
                if is_new:
                    self.created.add(agent_name)
                    decision.created_names.append(agent_name)
            elif agent_id not in decision.dispatched_ids:
                decision.dispatched_ids.append(agent_id)
            result = {"status": "submitted", "agent_name": agent_name, "new_agent_created": is_new}
        elif name == "send_message_to_user":
            decision.user_messages.append(str(args.get("message", "")))
            result = {"status": "delivered"}
        elif name == "send_draft":
            result = {"status": "draft_recorded", "to": args.get("to"), "subject": args.get("subject")}
        else:  # wait; unknown names never reach here
            result = {"status": "waiting", "reason": args.get("reason")}
        return {"tool": name, "status": "success", "arguments": args, "result": result}


class BaselineSystem:
    name = "baseline"

    def render_roster(self, world: World, case: Case | None) -> str:
        agents = world.roster_order()
        if not agents:
            body = "None"
        else:
            body = "\n".join(f'<agent name="{escape(a.name or "agent", quote=True)}" />' for a in agents)
        return f"<active_agents>\n{body}\n</active_agents>"

    def shown_ids(self, world: World, case: Case | None) -> list[str]:
        return [agent.id for agent in world.roster_order()]

    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def tool_schemas(self) -> list[dict[str, Any]]:
        return _TOOL_SCHEMAS

    def tool_signatures(self) -> dict[str, Callable[..., Any]]:
        return _SHIMS

    def new_turn(self, world: World, case: Case) -> BaselineTurn:
        return BaselineTurn(world, self)

    def resolve_name(self, world: World, name: str) -> str | None:
        """Map a name the IA passed to send_message_to_agent to an agent id; None = creates."""
        for agent in world.agents:
            if agent.name == name:
                return agent.id
        return None
