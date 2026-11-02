"""A fake mailbox for trying the app without a real Gmail account (`OPENPOKE_FAKE_GMAIL=1`).

Every Gmail call goes to an in-memory mailbox instead of Composio: the agents search, draft, reply and send as
usual, the send gate still decides, and the importance watcher polls it. Nothing leaves the machine. The mailbox
lives in the server process, so a restart (including `--reload`) starts it again from the seed emails.

The mailbox is the end-to-end eval's `FakeGmail`, with real timestamps instead of the eval's virtual clock so
the watcher sees new mail as new.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

FAKE_USER_ID = "fake-gmail-user"
FAKE_OWNER = "you@openpoke.test"

# Mail already in the inbox at start; the watcher's first poll marks it as seen, so it raises no alerts.
SEED_EMAILS = (
    ("Carl Brenner <carl@brennerbuild.test>", "Bathroom remodel quote",
     "Hi! I can do the bathroom for $8,400, tile included. I could start the week of the 16th. "
     "Does Friday work to go over the details?\n\nCarl", 26),
    ("Bright Smile Dental <frontdesk@brightsmile.test>", "Appointment reminder",
     "This is a reminder of your cleaning on Tuesday at 10:30am with Dr. Ito. "
     "Reply to this email if you need to reschedule.", 20),
    ("Dana Ruiz <dana.ruiz@maplecourt.test>", "Parking spot for next month",
     "Hi, spot 14 is free from the 1st if you still want it. It's $95 a month. Let me know by Friday.\n\nDana", 8),
    ("Priya Shah <priya@shahlab.test>", "Dinner Saturday?",
     "Are we still on for Saturday? I was thinking the Thai place on 5th around 7.", 3),
)

_lock = threading.Lock()
_mailbox: Any = None


def enabled() -> bool:
    return os.getenv("OPENPOKE_FAKE_GMAIL", "").strip().lower() in {"1", "true", "yes", "on"}


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def mailbox() -> Any:
    """The process-wide fake mailbox, created and seeded on first use."""
    global _mailbox
    with _lock:
        if _mailbox is None:
            from .fake_mailbox import FakeGmail  # imported lazily: only fake mode needs it

            class _LiveFakeGmail(FakeGmail):
                def _now(self) -> str:
                    return _iso(datetime.now(timezone.utc))

            box = _LiveFakeGmail(owner=FAKE_OWNER)
            now = datetime.now(timezone.utc)
            for sender, subject, body, hours_ago in SEED_EMAILS:
                box.add_message(sender=sender, subject=subject, body=body,
                                timestamp=_iso(now - timedelta(hours=hours_ago)))
            from ..approvals import gate

            gate.decision_sink = box.record_decision  # so the sent folder shows which approval let each send out
            _mailbox = box
        return _mailbox


def execute(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return mailbox().execute_gmail_tool(tool_name, FAKE_USER_ID, arguments=arguments)


def deliver(*, sender: str, subject: str, body: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """An email arriving from outside, as if someone had sent it to the user."""
    message = mailbox().add_message(sender=sender, subject=subject, body=body, thread_id=thread_id or None)
    return _message(message)


def _message(message: Any) -> Dict[str, Any]:
    return {
        "id": message.id,
        "thread_id": message.thread_id,
        "from": message.sender,
        "to": list(message.to),
        "cc": list(message.cc),
        "subject": message.subject,
        "body": message.body,
        "timestamp": message.timestamp,
        "labels": list(message.labels),
    }


def snapshot() -> Dict[str, Any]:
    box = mailbox()
    inbox: List[Dict[str, Any]] = sorted((_message(m) for m in box.messages.values()),
                                         key=lambda m: m["timestamp"], reverse=True)
    drafts = [
        {"id": d.id, "thread_id": d.thread_id, "to": d.to, "cc": d.cc, "bcc": d.bcc, "subject": d.subject,
         "body": d.body}
        for d in box.drafts.values()
    ]
    sent = [
        {"kind": s.kind, "to": s.to, "cc": s.cc, "bcc": s.bcc, "subject": s.subject, "body": s.note,
         "thread_id": s.thread_id, "approved": bool(s.decision and s.decision.get("allowed")),
         "reason": (s.decision or {}).get("reason")}
        for s in reversed(box.sent)
    ]
    blocked = [
        {"kind": d["kind"], "recipients": d["recipients"], "reason": d["reason"]}
        for d in reversed(box.decisions) if not d["allowed"]
    ]
    return {"owner": box.owner, "inbox": inbox, "drafts": drafts, "sent": sent, "blocked": blocked}


__all__ = ["FAKE_OWNER", "FAKE_USER_ID", "deliver", "enabled", "execute", "mailbox", "snapshot"]
