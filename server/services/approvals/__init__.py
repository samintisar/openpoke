"""Send approvals: the `pending_actions` store and the gate that every outgoing email passes through."""

from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def default_db_path() -> Path:
    return _DATA_DIR / "approvals.db"


from . import gate  # noqa: E402
from .gate import (  # noqa: E402
    APPROVAL_TTL,
    ApprovalError,
    Decision,
    SendContent,
    agent_ids_with_pending,
    approve,
    approve_recurring,
    check_send,
    content_hash,
    expire_stale,
    get_action,
    get_approval_store,
    mark_sent,
    pending_shown,
    render,
    request_approval,
    revoke_standing,
    set_approval_store,
    show,
)
from .store import PendingAction, PendingActionStore, StandingApproval  # noqa: E402

__all__ = [
    "APPROVAL_TTL",
    "ApprovalError",
    "Decision",
    "PendingAction",
    "PendingActionStore",
    "SendContent",
    "StandingApproval",
    "agent_ids_with_pending",
    "approve",
    "approve_recurring",
    "check_send",
    "content_hash",
    "default_db_path",
    "expire_stale",
    "gate",
    "get_action",
    "get_approval_store",
    "mark_sent",
    "pending_shown",
    "render",
    "request_approval",
    "revoke_standing",
    "set_approval_store",
    "show",
]
