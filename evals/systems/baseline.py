"""Today's behaviour: every agent ever created, as a bare name, matched by exact string."""

from __future__ import annotations

from html import escape

from ..cases import Case
from ..worlds.model import World


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

    def resolve_name(self, world: World, name: str) -> str | None:
        for agent in world.agents:
            if agent.name == name:
                return agent.id
        return None
