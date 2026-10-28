"""The send approval gate: nothing leaves the mailbox unless the user approved that exact content.

Flow:
- An execution agent drafts (`gmail_create_draft`) or tries to reply/forward. The gate records a `pending`
  action holding the exact recipients, subject and body, and its `content_hash`.
- The interaction agent shows it with `send_draft(action_id=...)` (`show`), which renders the stored content
  and sets `shown_at`.
- Only a user turn whose message arrived after `shown_at` can `approve` it.
- The agent's send goes through `check_send`, which passes only for an `approved` action of the same agent,
  Gmail object and hash. A draft edited after approval is `invalidated`. A successful send marks it `sent`.

`check_send` is the single place that decides whether a send may proceed (standing approvals hook in there).
`enforce = False` is the ablation switch: the gate still decides and reports, but lets every send through.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import getaddresses
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ...logging_config import logger
from .store import PendingAction, PendingActionStore

# Pending (and approved but unsent) actions older than this expire.
APPROVAL_TTL = timedelta(days=7)

# Ablation switch: when False, `check_send` logs and allows every send, but still returns (and reports) the
# decision it would have made. Production keeps it True.
enforce: bool = True

# Optional observer for every send decision (the end-to-end eval records them); exceptions are swallowed.
decision_sink: Optional[Callable[["Decision"], None]] = None

_store: Optional[PendingActionStore] = None


def get_approval_store() -> PendingActionStore:
    """The shared store; opened on first use."""
    global _store
    if _store is None:
        from . import default_db_path

        _store = PendingActionStore(default_db_path())
    return _store


def set_approval_store(store: Optional[PendingActionStore]) -> None:
    """Point the gate at another store (tests, evals); None reopens the default on next use."""
    global _store
    _store = store


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ApprovalError(Exception):
    """Showing or approving an action failed; the message says why, for the interaction agent."""


# -- content and hashing --------------------------------------------------------------------------------------


def normalize_addresses(values: Iterable[Optional[str]]) -> List[str]:
    """Bare addresses, lowercased, deduplicated and sorted; accepts "Name <a@x>" and comma-separated lists."""
    raw = [value for value in values if isinstance(value, str) and value.strip()]
    addresses = {address.strip().lower() for _name, address in getaddresses(raw) if address.strip()}
    return sorted(addresses)


def normalize_subject(subject: Optional[str]) -> str:
    return " ".join((subject or "").split())


def normalize_body(body: Optional[str]) -> str:
    """Line endings unified, trailing whitespace per line and around the body dropped."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


@dataclass(frozen=True)
class SendContent:
    """What a send would deliver, normalized: the thing a hash, an approval and the user's view cover."""

    to: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    body_format: str = "plain"
    attachments: List[str] = field(default_factory=list)  # file names

    @classmethod
    def build(
        cls,
        *,
        to: Sequence[Optional[str]] = (),
        cc: Sequence[Optional[str]] = (),
        bcc: Sequence[Optional[str]] = (),
        subject: Optional[str] = None,
        body: Optional[str] = None,
        is_html: Optional[bool] = None,
        attachments: Sequence[Optional[str]] = (),
    ) -> "SendContent":
        return cls(
            to=normalize_addresses(to),
            cc=normalize_addresses(cc),
            bcc=normalize_addresses(bcc),
            subject=normalize_subject(subject),
            body=normalize_body(body),
            body_format="html" if is_html else "plain",
            attachments=sorted({name.strip() for name in attachments if isinstance(name, str) and name.strip()}),
        )

    def recipients(self) -> Dict[str, List[str]]:
        """Stored with the action; attachment names ride along so the user sees them."""
        stored: Dict[str, List[str]] = {"to": list(self.to), "cc": list(self.cc), "bcc": list(self.bcc)}
        if self.attachments:
            stored["attachments"] = list(self.attachments)
        return stored

    def all_recipients(self) -> List[str]:
        return sorted({*self.to, *self.cc, *self.bcc})


def content_hash(kind: str, gmail_ref: Optional[str], content: SendContent) -> str:
    """sha256 over the kind, the Gmail object it acts on, the normalized recipients, subject and body."""
    canonical = json.dumps(
        {
            "kind": kind,
            "ref": gmail_ref or "",
            "to": content.to,
            "cc": content.cc,
            "bcc": content.bcc,
            "subject": content.subject,
            "body": content.body,
            "format": content.body_format,
            "attachments": content.attachments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- reading with lazy expiry ---------------------------------------------------------------------------------


def _expired(action: PendingAction, now: datetime) -> bool:
    return action.status in {"pending", "approved"} and now - action.created_at > APPROVAL_TTL


def _fresh(store: PendingActionStore, action: Optional[PendingAction], now: datetime) -> Optional[PendingAction]:
    """The action, expired first if it is past its time."""
    if action is not None and _expired(action, now):
        store.transition(action.id, ("pending", "approved"), "expired")
        action = store.get(action.id)
    return action


def get_action(action_id: int, now: Optional[datetime] = None) -> Optional[PendingAction]:
    store = get_approval_store()
    return _fresh(store, store.get(action_id), now or _now())


def expire_stale(now: Optional[datetime] = None) -> int:
    """Expire pending and approved-but-unsent actions past `APPROVAL_TTL`; the count expired."""
    return get_approval_store().expire_older_than((now or _now()) - APPROVAL_TTL, ("pending", "approved"))


def agent_ids_with_pending(now: Optional[datetime] = None) -> set[str]:
    """Agents with a pending action created within `APPROVAL_TTL`: open work for the lifecycle."""
    now = now or _now()
    return {
        action.agent_id
        for action in get_approval_store().with_status(("pending",))
        if action.agent_id and now - action.created_at <= APPROVAL_TTL
    }


def pending_shown(now: Optional[datetime] = None) -> List[PendingAction]:
    """Pending actions the user has been shown, oldest first."""
    store = get_approval_store()
    now = now or _now()
    actions = [_fresh(store, action, now) for action in store.with_status(("pending",))]
    return [action for action in actions if action and action.status == "pending" and action.shown_at]


# -- EA side: recording and the send decision -----------------------------------------------------------------


def request_approval(
    kind: str, agent_id: Optional[str], gmail_ref: Optional[str], content: SendContent
) -> PendingAction:
    """Record a pending action for this exact content, reusing an identical pending one (an agent's retry)."""
    store = get_approval_store()
    now = _now()
    digest = content_hash(kind, gmail_ref, content)
    for existing in store.find(kind=kind, gmail_ref=gmail_ref, agent_id=agent_id, statuses=("pending",)):
        existing = _fresh(store, existing, now)
        if existing and existing.status == "pending" and existing.content_hash == digest:
            return existing
    if kind == "draft" and gmail_ref:
        # A draft has one current content: approvals waiting on an older version of it are void.
        for stale in store.find(kind=kind, gmail_ref=gmail_ref, statuses=("pending", "approved")):
            store.transition(stale.id, ("pending", "approved"), "invalidated")
    action = store.insert(
        kind=kind,
        agent_id=agent_id,
        gmail_ref=gmail_ref,
        recipients=content.recipients(),
        subject=content.subject,
        body=content.body,
        body_format=content.body_format,
        content_hash=digest,
        created_at=now,
    )
    logger.info("send approval requested", extra={"action_id": action.id, "kind": kind, "agent_id": agent_id})
    return action


@dataclass(frozen=True)
class Decision:
    """Whether a send may proceed. `would_allow` is what the gate decided; `allowed` is what happens (they
    differ only with `enforce` off). `action_id` is the approved action that covers the send, if any."""

    allowed: bool
    would_allow: bool
    reason: str  # "approved", "not_approved", "content_changed", "no_agent"
    kind: str
    agent_id: Optional[str]
    gmail_ref: Optional[str]
    recipients: List[str]
    content_hash: str
    action_id: Optional[int] = None
    enforced: bool = True


def _report(decision: Decision) -> Decision:
    logger.info(
        "send gate decision",
        extra={
            "allowed": decision.allowed,
            "would_allow": decision.would_allow,
            "reason": decision.reason,
            "kind": decision.kind,
            "agent_id": decision.agent_id,
            "action_id": decision.action_id,
            "enforced": decision.enforced,
        },
    )
    sink = decision_sink
    if sink is not None:
        try:
            sink(decision)
        except Exception as exc:  # pragma: no cover - observers never break a send
            logger.warning("send decision sink failed", extra={"error": str(exc)})
    return decision


def check_send(
    kind: str,
    agent_id: Optional[str],
    recipients: Sequence[str],
    content_hash: str,
    gmail_ref: Optional[str],
) -> Decision:
    """Decide whether `agent_id` may send this exact content now: the one place sends are allowed.

    It passes only when an `approved`, unexpired action of the same kind, Gmail object and agent has the same
    hash. A draft whose approved action no longer matches what Gmail holds was edited after approval: that
    action becomes `invalidated`. (Task 4's standing approvals add a second way to pass, here.)
    """
    store = get_approval_store()
    now = _now()
    action_id: Optional[int] = None
    if not agent_id:
        reason = "no_agent"
    else:
        approved = [
            action
            for action in (
                _fresh(store, row, now)
                for row in store.find(kind=kind, gmail_ref=gmail_ref, agent_id=agent_id, statuses=("approved",))
            )
            if action is not None and action.status == "approved"
        ]
        match = next((action for action in approved if action.content_hash == content_hash), None)
        if match is not None:
            reason, action_id = "approved", match.id
        else:
            reason = "not_approved"
            if kind == "draft" and approved:
                # The draft changed since the user approved it: that approval no longer covers anything.
                for action in approved:
                    store.transition(action.id, ("approved",), "invalidated")
                reason = "content_changed"
    would_allow = reason == "approved"
    decision = Decision(
        allowed=would_allow or not enforce,
        would_allow=would_allow,
        reason=reason,
        kind=kind,
        agent_id=agent_id,
        gmail_ref=gmail_ref,
        recipients=sorted(set(recipients)),
        content_hash=content_hash,
        action_id=action_id,
        enforced=enforce,
    )
    if not would_allow and not enforce:
        logger.warning("send gate off: allowing an unapproved send", extra={"reason": reason, "kind": kind})
    return _report(decision)


def mark_sent(action_id: int) -> bool:
    return get_approval_store().transition(action_id, ("approved",), "sent")


# -- IA side: showing and approving ---------------------------------------------------------------------------


def _parse_id(action_id: object) -> Optional[int]:
    if isinstance(action_id, bool):
        return None
    if isinstance(action_id, int):
        return action_id
    if isinstance(action_id, str) and re.fullmatch(r"\s*\d+\s*", action_id):
        return int(action_id)
    return None


def _describe_shown(now: datetime) -> str:
    shown = pending_shown(now)
    if not shown:
        return "No action is waiting for approval."
    listed = ", ".join(f"{action.id} ({', '.join(action.recipients.get('to') or []) or 'no recipient'})"
                       for action in shown)
    return f"Actions waiting for approval: {listed}."


def _lookup(action_id: object, now: datetime) -> PendingAction:
    parsed = _parse_id(action_id)
    if parsed is None:
        raise ApprovalError(
            f"Ambiguous action id {action_id!r}: pass exactly one numeric action_id. {_describe_shown(now)}"
        )
    store = get_approval_store()
    action = _fresh(store, store.get(parsed), now)
    if action is None:
        raise ApprovalError(f"Unknown action id {parsed}. {_describe_shown(now)}")
    return action


def render(action: PendingAction) -> str:
    """The exact stored content, as the user sees it."""
    lines = []
    if action.kind == "reply":
        lines.append("Reply in the existing email thread")
    elif action.kind == "forward":
        lines.append("Forward of the original email")
    recipients = action.recipients
    lines.append(f"To: {', '.join(recipients.get('to') or [])}")
    if recipients.get("cc"):
        lines.append(f"Cc: {', '.join(recipients['cc'])}")
    if recipients.get("bcc"):
        lines.append(f"Bcc: {', '.join(recipients['bcc'])}")
    if action.subject or action.kind == "draft":
        lines.append(f"Subject: {action.subject}")
    if recipients.get("attachments"):
        lines.append(f"Attachments: {', '.join(recipients['attachments'])}")
    header = "\n".join(lines)
    return f"{header}\n\n{action.body}" if action.body else header


def show(action_id: object, now: Optional[datetime] = None) -> PendingAction:
    """Mark a pending action shown to the user now; returns it. Errors unless it is pending."""
    now = now or _now()
    action = _lookup(action_id, now)
    if action.status != "pending":
        raise ApprovalError(f"Action {action.id} is {action.status}; only pending actions can be shown.")
    if not get_approval_store().set_shown(action.id, now):  # pragma: no cover - raced with a status change
        raise ApprovalError(f"Action {action.id} is no longer pending.")
    return get_approval_store().get(action.id)  # type: ignore[return-value]


def approve(
    action_id: object,
    *,
    origin: Optional[str],
    user_message_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> PendingAction:
    """Approve a pending action for the turn described by `origin` and `user_message_at`.

    Errors unless the turn was started by the user, the user's message arrived after the action was shown,
    and the action is pending and unexpired.
    """
    now = now or _now()
    if origin != "user" or user_message_at is None:
        raise ApprovalError(
            "Only the user can approve a send, in their own message. This turn was not started by a user message."
        )
    action = _lookup(action_id, now)
    if action.status != "pending":
        raise ApprovalError(f"Action {action.id} is {action.status}, not pending; it cannot be approved.")
    if action.shown_at is None:
        raise ApprovalError(
            f"Action {action.id} has not been shown to the user. Show it with send_draft(action_id={action.id}) "
            "and wait for their reply."
        )
    if not user_message_at.astimezone(timezone.utc) > action.shown_at:
        raise ApprovalError(
            f"The user's message came before action {action.id} was shown, so it cannot approve it. Wait for "
            "the user's reply to what they were shown."
        )
    if not get_approval_store().transition(action.id, ("pending",), "approved", approved_at=now):
        raise ApprovalError(f"Action {action.id} is no longer pending.")  # pragma: no cover - raced
    logger.info("send approved", extra={"action_id": action.id, "agent_id": action.agent_id})
    return get_approval_store().get(action.id)  # type: ignore[return-value]


__all__ = [
    "APPROVAL_TTL",
    "ApprovalError",
    "Decision",
    "SendContent",
    "agent_ids_with_pending",
    "approve",
    "check_send",
    "content_hash",
    "decision_sink",
    "enforce",
    "expire_stale",
    "get_action",
    "get_approval_store",
    "mark_sent",
    "normalize_addresses",
    "normalize_body",
    "normalize_subject",
    "pending_shown",
    "render",
    "request_approval",
    "set_approval_store",
    "show",
]
