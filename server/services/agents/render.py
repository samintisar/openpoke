"""Render agent cards into the `<active_agents>` block the interaction agent reads."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Mapping, Optional, Sequence

from .cards import AgentCard, as_utc

TRUNCATE_CHARS = 160


def relative_time(then: datetime, now: datetime) -> str:
    seconds = max(0, int((as_utc(now) - as_utc(then)).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def truncate(text: str, limit: int = TRUNCATE_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def render_card(card: AgentCard, now: datetime, open_work: Optional[str] = None) -> str:
    attrs = (
        f'id="{escape(card.id, quote=True)}" name="{escape(card.name, quote=True)}" '
        f'last_active="{relative_time(card.last_active_at, now)}"'
    )
    if card.status == "archived":
        attrs += ' status="archived"'
    if open_work:
        attrs += f' open_work="{escape(open_work, quote=True)}"'
    lines = [f"  <agent {attrs}>", f"    purpose: {escape(truncate(card.purpose), quote=False)}"]
    if card.last_status.strip():
        lines.append(f"    status: {escape(truncate(card.last_status), quote=False)}")
    lines.append("  </agent>")
    return "\n".join(lines)


def render_cards(
    selected: Sequence[AgentCard],
    total: int,
    now: datetime,
    open_work: Optional[Mapping[str, str]] = None,
) -> str:
    """`<active_agents shown="N" total="M">` with one block per card; `None` when nothing is shown."""
    if not selected:
        return f'<active_agents shown="0" total="{total}">None</active_agents>'
    open_work = open_work or {}
    body = "\n".join(render_card(card, now, open_work.get(card.id)) for card in selected)
    return f'<active_agents shown="{len(selected)}" total="{total}">\n{body}\n</active_agents>'


__all__ = ["relative_time", "render_card", "render_cards", "truncate"]
