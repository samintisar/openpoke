from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...config import get_settings
from .service import InteractionInbox
from .store import KINDS, InboxItem, InboxStore


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_default_db_path = _DATA_DIR / "inbox.db"
_inbox: Optional[InteractionInbox] = None


def get_inbox() -> InteractionInbox:
    """The shared inbox; its store is opened on first use."""
    global _inbox
    if _inbox is None:
        _inbox = InteractionInbox(InboxStore(_default_db_path), debounce_seconds=get_settings().inbox_debounce_seconds)
    return _inbox


def post(kind: str, payload: str, agent_id: Optional[str] = None) -> InboxItem:
    """Post an item to the shared inbox."""
    return get_inbox().post(kind, payload, agent_id)


__all__ = ["InboxItem", "InboxStore", "InteractionInbox", "KINDS", "get_inbox", "post"]
