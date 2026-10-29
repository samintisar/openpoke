"""An in-memory Gmail for the end-to-end eval: messages, threads, drafts and the sent folder.

`FakeGmail.execute(tool_name, composio_user_id, arguments)` stands in for the Composio actions the agents use
(`tools/gmail.py::_execute` and the search task's `execute_gmail_tool`) and answers in the shapes those callers
parse: `{"data": ..., "successful": bool, "error": ...}`, with Gmail-API message payloads (headers plus
base64url parts) for `GMAIL_GET_DRAFT` and Composio's verbose fields (`messageId`, `messageText`, `sender`, ...)
for `GMAIL_FETCH_EMAILS`.

Every message that leaves the mailbox (`GMAIL_SEND_DRAFT`, `GMAIL_REPLY_TO_THREAD`, `GMAIL_FORWARD_MESSAGE`) is
appended to `sent` together with the send gate's decision for it (the gate reports decisions through
`record_decision`, wired to `gate.decision_sink`). `on_send`, if set, sees each send as it happens, before the
caller marks its approval sent, so an auditor can judge it against the approval state of that moment.

Ids are derived from content (not from call order), so two agents drafting concurrently get the same ids on
every run. Timestamps come from a fixed virtual clock.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import getaddresses
from typing import Any, Callable, Dict, Iterable, List, Optional

OWNER = "me@harborline.example"  # the mailbox owner, i.e. the user
VIRTUAL_START = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)  # Monday; seeded mail is dated before it
SEND_TOOLS = {"GMAIL_SEND_DRAFT": "draft", "GMAIL_REPLY_TO_THREAD": "reply", "GMAIL_FORWARD_MESSAGE": "forward"}


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")  # padded, as the email cleaner needs


def addresses(values: Iterable[Optional[str]]) -> List[str]:
    """Bare lowercased addresses from "Name <a@x>" strings or comma-separated lists, in order, deduplicated."""
    raw = [value for value in values if isinstance(value, str) and value.strip()]
    out: List[str] = []
    for _name, address in getaddresses(raw):
        address = address.strip().lower()
        if address and address not in out:
            out.append(address)
    return out


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if isinstance(item, str)]


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Message:
    id: str
    thread_id: str
    sender: str  # "Name <address>" or a bare address
    to: List[str]
    subject: str
    body: str
    timestamp: str  # ISO 8601, UTC
    labels: List[str] = field(default_factory=lambda: ["INBOX"])
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)  # file names
    is_html: bool = False


@dataclass
class Draft:
    id: str
    message_id: str
    thread_id: Optional[str]
    to: List[str]
    subject: str
    body: str
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    is_html: bool = False
    attachments: List[str] = field(default_factory=list)


@dataclass
class SentMail:
    """One message that left the mailbox."""

    index: int
    tool: str
    kind: str  # draft, reply or forward (the gate's action kinds)
    gmail_ref: str  # draft id, thread id or forwarded message id: what the gate keys the action on
    message_id: str
    thread_id: str
    to: List[str]
    cc: List[str]
    bcc: List[str]
    subject: str
    body: str  # everything delivered (a forward includes the original)
    note: str  # what the sender wrote: the draft/reply body, or a forward's additional text
    is_html: bool
    attachments: List[str]
    decision: Optional[Dict[str, Any]] = None  # the gate decision that let it through
    audit: Dict[str, Any] = field(default_factory=dict)  # filled by `on_send`

    def recipients(self) -> List[str]:
        return sorted({*self.to, *self.cc, *self.bcc})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Contact:
    name: str
    email: str

    def person(self, index: int) -> Dict[str, Any]:
        return {
            "resourceName": f"people/c{index}",
            "names": [{"displayName": self.name}],
            "emailAddresses": [{"value": self.email}],
        }


def _ok(data: Any) -> Dict[str, Any]:
    return {"data": data, "successful": True, "error": None}


def _fail(error: str) -> Dict[str, Any]:
    return {"data": {}, "successful": False, "error": error}


_QUERY_TERM = re.compile(r'(-?)(?:(\w+):("[^"]*"|\S+)|"([^"]*)"|(\S+))')


class FakeGmail:
    def __init__(self, owner: str = OWNER, contacts: Iterable[Contact] = ()) -> None:
        self.owner = owner
        self.contacts: List[Contact] = list(contacts)
        self.messages: Dict[str, Message] = {}
        self.drafts: Dict[str, Draft] = {}
        self.sent: List[SentMail] = []
        self.decisions: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self.on_send: Optional[Callable[[SentMail], None]] = None
        self._ids: set[str] = set()
        self._ticks = 0
        self._consumed_decisions = 0

    # -- ids and time ------------------------------------------------------------------------------------------

    def _new_id(self, prefix: str, *parts: Any) -> str:
        digest = hashlib.sha1("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()
        base = f"{prefix}{digest[:10]}"
        candidate, n = base, 1
        while candidate in self._ids:
            n += 1
            candidate = f"{base}-{n}"
        self._ids.add(candidate)
        return candidate

    def _now(self) -> str:
        self._ticks += 1
        return _iso(VIRTUAL_START + timedelta(minutes=self._ticks))

    # -- seeding -----------------------------------------------------------------------------------------------

    def add_message(
        self,
        *,
        sender: str,
        subject: str,
        body: str,
        to: Iterable[str] = (),
        cc: Iterable[str] = (),
        timestamp: Optional[str] = None,
        labels: Iterable[str] = ("INBOX",),
        attachments: Iterable[str] = (),
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        is_html: bool = False,
    ) -> Message:
        """Deliver (or seed) one message; a new thread unless `thread_id` names one."""
        mid = message_id or self._new_id("m", sender, subject, body, timestamp)
        self._ids.add(mid)
        thread = thread_id or self._new_id("t", mid)
        self._ids.add(thread)
        message = Message(
            id=mid,
            thread_id=thread,
            sender=sender,
            to=addresses(to) or [self.owner],
            cc=addresses(cc),
            subject=subject,
            body=body,
            timestamp=timestamp or self._now(),
            labels=list(labels),
            attachments=list(attachments),
            is_html=is_html,
        )
        self.messages[mid] = message
        return message

    def create_draft(
        self,
        *,
        to: Iterable[str],
        subject: str,
        body: str,
        cc: Iterable[str] = (),
        bcc: Iterable[str] = (),
        is_html: bool = False,
        thread_id: Optional[str] = None,
        attachments: Iterable[str] = (),
    ) -> Draft:
        to_list, cc_list, bcc_list = addresses(to), addresses(cc), addresses(bcc)
        draft_id = self._new_id("r", to_list, cc_list, bcc_list, subject, body, thread_id)
        draft = Draft(
            id=draft_id,
            message_id=self._new_id("m", draft_id),
            thread_id=thread_id,
            to=to_list,
            subject=subject,
            body=body,
            cc=cc_list,
            bcc=bcc_list,
            is_html=is_html,
            attachments=list(attachments),
        )
        self.drafts[draft_id] = draft
        return draft

    def edit_draft(self, draft_id: str, **changes: Any) -> Draft:
        """Change a draft in place, as the user editing it in Gmail would."""
        draft = self.drafts[draft_id]
        for key, value in changes.items():
            setattr(draft, key, addresses(value) if key in {"to", "cc", "bcc"} else value)
        return draft

    def thread(self, thread_id: str) -> List[Message]:
        return sorted((m for m in self.messages.values() if m.thread_id == thread_id), key=lambda m: m.timestamp)

    # -- the gate's decisions ----------------------------------------------------------------------------------

    def record_decision(self, decision: Any) -> None:
        """`gate.decision_sink`: keep every send decision, allowed or not."""
        record = asdict(decision) if hasattr(decision, "__dataclass_fields__") else dict(decision)
        record["sent_index"] = None
        self.decisions.append(record)

    def _claim_decision(self, kind: str, gmail_ref: str, index: int) -> Optional[Dict[str, Any]]:
        """The latest unclaimed allowed decision for this send (made just before the send call)."""
        for position in range(len(self.decisions) - 1, self._consumed_decisions - 1, -1):
            record = self.decisions[position]
            if record["kind"] == kind and str(record.get("gmail_ref")) == str(gmail_ref) and record["allowed"]:
                record["sent_index"] = index
                self._consumed_decisions = position + 1
                return record
        return None

    # -- Composio entry points ---------------------------------------------------------------------------------

    def execute(self, tool_name: str, composio_user_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Same signature as `tools/gmail.py::_execute`."""
        args = {key: value for key, value in (arguments or {}).items() if value is not None}
        handler = getattr(self, f"_do_{tool_name.lower()}", None)
        if handler is None:
            self.calls.append({"tool": tool_name, "arguments": args, "successful": False})
            raise RuntimeError(f"{tool_name} invocation failed: not supported by the fake mailbox")
        result = handler(args)
        self.calls.append({"tool": tool_name, "arguments": args, "successful": bool(result.get("successful"))})
        return result

    def execute_gmail_tool(
        self, tool_name: str, composio_user_id: str, *, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Same signature as `services.gmail.execute_gmail_tool` (the search task's import)."""
        return self.execute(tool_name, composio_user_id, arguments or {})

    # -- shapes ------------------------------------------------------------------------------------------------

    @staticmethod
    def _headers(to: List[str], cc: List[str], bcc: List[str], subject: str, sender: str) -> List[Dict[str, str]]:
        headers = [{"name": "From", "value": sender}, {"name": "To", "value": ", ".join(to)}]
        if cc:
            headers.append({"name": "Cc", "value": ", ".join(cc)})
        if bcc:
            headers.append({"name": "Bcc", "value": ", ".join(bcc)})
        headers.append({"name": "Subject", "value": subject})
        return headers

    @staticmethod
    def _parts(body: str, is_html: bool, attachments: List[str]) -> List[Dict[str, Any]]:
        plain = re.sub(r"<[^>]+>", "", body) if is_html else body
        parts: List[Dict[str, Any]] = [{"partId": "0", "mimeType": "text/plain", "filename": "", "body": {"data": _b64(plain)}}]
        if is_html:
            parts.append({"partId": "1", "mimeType": "text/html", "filename": "", "body": {"data": _b64(body)}})
        for n, name in enumerate(attachments):
            parts.append({
                "partId": str(len(parts)), "mimeType": "application/octet-stream", "filename": name,
                "body": {"attachmentId": f"att-{n}", "size": 1024},
            })
        return parts

    def _draft_shape(self, draft: Draft) -> Dict[str, Any]:
        return {
            "id": draft.id,
            "message": {
                "id": draft.message_id,
                "threadId": draft.thread_id or draft.message_id,
                "labelIds": ["DRAFT"],
                "payload": {
                    "mimeType": "multipart/alternative",
                    "headers": self._headers(draft.to, draft.cc, draft.bcc, draft.subject, self.owner),
                    "parts": self._parts(draft.body, draft.is_html, draft.attachments),
                },
            },
        }

    def _message_shape(self, message: Message) -> Dict[str, Any]:
        """A verbose GMAIL_FETCH_EMAILS message, as `parse_gmail_fetch_response` reads it."""
        return {
            "messageId": message.id,
            "threadId": message.thread_id,
            "messageTimestamp": message.timestamp,
            "labelIds": list(message.labels),
            "subject": message.subject,
            "sender": message.sender,
            "to": ", ".join(message.to),
            "messageText": message.body,
            "preview": {"subject": message.subject, "body": message.body[:200]},
            "attachmentList": [
                {"filename": name, "mimeType": "application/octet-stream", "attachmentId": f"att-{n}"}
                for n, name in enumerate(message.attachments)
            ],
            "payload": {
                "mimeType": "text/plain",
                "headers": self._headers(message.to, message.cc, [], message.subject, message.sender),
                "body": {"data": _b64(message.body)},
            },
        }

    # -- search ------------------------------------------------------------------------------------------------

    def search(self, query: str, *, max_results: int = 10, include_spam_trash: bool = False) -> List[Message]:
        """A small Gmail query language: field operators, quoted phrases, negation and top-level OR."""
        found = [
            message for message in self.messages.values()
            if (include_spam_trash or not {"SPAM", "TRASH"} & set(message.labels)) and _matches(message, query)
        ]
        found.sort(key=lambda message: message.timestamp, reverse=True)
        return found[: max(1, int(max_results))]

    # -- actions -----------------------------------------------------------------------------------------------

    def _do_gmail_fetch_emails(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            limit = int(args.get("max_results", 10))
        except (TypeError, ValueError):
            limit = 10
        found = self.search(
            str(args.get("query") or ""), max_results=limit, include_spam_trash=bool(args.get("include_spam_trash"))
        )
        if args.get("ids_only"):
            messages = [{"messageId": m.id, "threadId": m.thread_id} for m in found]
        else:
            messages = [self._message_shape(m) for m in found]
        return _ok({"messages": messages, "nextPageToken": None, "resultSizeEstimate": len(messages)})

    def _do_gmail_create_email_draft(self, args: Dict[str, Any]) -> Dict[str, Any]:
        recipient = args.get("recipient_email")
        if not isinstance(recipient, str) or not addresses([recipient]):
            return _fail("recipient_email must be a valid email address")
        thread_id = args.get("thread_id")
        if thread_id and not self.thread(str(thread_id)):
            return _fail(f"Thread {thread_id} not found")
        attachment = args.get("attachment")
        names = [attachment["name"]] if isinstance(attachment, dict) and attachment.get("name") else []
        draft = self.create_draft(
            to=[recipient, *_as_list(args.get("extra_recipients"))],
            subject=str(args.get("subject") or ""),
            body=str(args.get("body") or ""),
            cc=_as_list(args.get("cc")),
            bcc=_as_list(args.get("bcc")),
            is_html=bool(args.get("is_html")),
            thread_id=str(thread_id) if thread_id else None,
            attachments=names,
        )
        return _ok({
            "id": draft.id,
            "message": {"id": draft.message_id, "threadId": draft.thread_id or draft.message_id, "labelIds": ["DRAFT"]},
        })

    def _do_gmail_get_draft(self, args: Dict[str, Any]) -> Dict[str, Any]:
        draft = self.drafts.get(str(args.get("draft_id")))
        if draft is None:
            return _fail("Requested entity was not found.")
        return _ok(self._draft_shape(draft))

    def _do_gmail_list_drafts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        drafts = []
        for draft in self.drafts.values():
            entry: Dict[str, Any] = {
                "id": draft.id, "message": {"id": draft.message_id, "threadId": draft.thread_id or draft.message_id},
            }
            if args.get("verbose", True):
                entry.update({
                    "subject": draft.subject, "to": ", ".join(draft.to), "cc": ", ".join(draft.cc),
                    "messageText": draft.body,
                })
            drafts.append(entry)
        try:
            limit = int(args.get("max_results", 50))
        except (TypeError, ValueError):
            limit = 50
        return _ok({"drafts": drafts[:limit], "resultSizeEstimate": len(drafts)})

    def _do_gmail_delete_draft(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.drafts.pop(str(args.get("draft_id")), None) is None:
            return _fail("Requested entity was not found.")
        return _ok({})

    def _deliver(
        self, *, tool: str, gmail_ref: str, thread_id: Optional[str], to: List[str], cc: List[str], bcc: List[str],
        subject: str, body: str, note: str, is_html: bool, attachments: List[str],
    ) -> SentMail:
        message = self.add_message(
            sender=self.owner, subject=subject, body=body, to=to, cc=cc, labels=("SENT",),
            attachments=attachments, thread_id=thread_id, is_html=is_html,
        )
        message.bcc = list(bcc)
        kind = SEND_TOOLS[tool]
        index = len(self.sent)
        sent = SentMail(
            index=index, tool=tool, kind=kind, gmail_ref=gmail_ref, message_id=message.id,
            thread_id=message.thread_id, to=list(to), cc=list(cc), bcc=list(bcc), subject=subject, body=body,
            note=note, is_html=is_html, attachments=list(attachments),
            decision=self._claim_decision(kind, gmail_ref, index),
        )
        self.sent.append(sent)
        if self.on_send is not None:
            self.on_send(sent)
        return sent

    def _do_gmail_send_draft(self, args: Dict[str, Any]) -> Dict[str, Any]:
        draft = self.drafts.get(str(args.get("draft_id")))
        if draft is None:
            return _fail("Requested entity was not found.")
        if not draft.to:
            return _fail("Draft has no recipient")
        sent = self._deliver(
            tool="GMAIL_SEND_DRAFT", gmail_ref=draft.id, thread_id=draft.thread_id, to=draft.to, cc=draft.cc,
            bcc=draft.bcc, subject=draft.subject, body=draft.body, note=draft.body, is_html=draft.is_html,
            attachments=draft.attachments,
        )
        del self.drafts[draft.id]
        return _ok({"id": sent.message_id, "threadId": sent.thread_id, "labelIds": ["SENT"]})

    def _do_gmail_reply_to_thread(self, args: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = str(args.get("thread_id") or "")
        messages = self.thread(thread_id)
        if not messages:
            return _fail(f"Thread {thread_id} not found")
        to = addresses([args.get("recipient_email"), *_as_list(args.get("extra_recipients"))])
        if not to:
            return _fail("recipient_email must be a valid email address")
        subject = messages[0].subject
        subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        attachment = args.get("attachment")
        names = [attachment["name"]] if isinstance(attachment, dict) and attachment.get("name") else []
        body = str(args.get("message_body") or "")
        sent = self._deliver(
            tool="GMAIL_REPLY_TO_THREAD", gmail_ref=thread_id, thread_id=thread_id, to=to,
            cc=addresses(_as_list(args.get("cc"))), bcc=addresses(_as_list(args.get("bcc"))), subject=subject,
            body=body, note=body, is_html=bool(args.get("is_html")), attachments=names,
        )
        return _ok({"id": sent.message_id, "threadId": sent.thread_id, "labelIds": ["SENT"]})

    def _do_gmail_forward_message(self, args: Dict[str, Any]) -> Dict[str, Any]:
        original = self.messages.get(str(args.get("message_id")))
        if original is None:
            return _fail("Requested entity was not found.")
        to = addresses([args.get("recipient_email")])
        if not to:
            return _fail("recipient_email must be a valid email address")
        note = str(args.get("additional_text") or "")
        body = (
            f"{note}\n\n---------- Forwarded message ---------\nFrom: {original.sender}\n"
            f"Subject: {original.subject}\n\n{original.body}"
        ).lstrip()
        sent = self._deliver(
            tool="GMAIL_FORWARD_MESSAGE", gmail_ref=original.id, thread_id=None, to=to, cc=[], bcc=[],
            subject=f"Fwd: {original.subject}", body=body, note=note, is_html=False,
            attachments=list(original.attachments),
        )
        return _ok({"id": sent.message_id, "threadId": sent.thread_id, "labelIds": ["SENT"]})

    def _people(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        wanted = (query or "").strip().lower()
        return [
            contact.person(n) for n, contact in enumerate(self.contacts, start=1)
            if not wanted or wanted in contact.name.lower() or wanted in contact.email.lower()
        ]

    def _do_gmail_get_contacts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        people = self._people()
        return _ok({"connections": people, "totalPeople": len(people), "totalItems": len(people)})

    def _do_gmail_get_people(self, args: Dict[str, Any]) -> Dict[str, Any]:
        resource = args.get("resource_name") or "people/me"
        if args.get("other_contacts"):
            return _ok({"otherContacts": [], "totalSize": 0})
        if resource == "people/me":
            return _ok({
                "resourceName": "people/me", "names": [{"displayName": "Mailbox Owner"}],
                "emailAddresses": [{"value": self.owner}],
            })
        match = next((p for p in self._people() if p["resourceName"] == resource), None)
        return _ok(match) if match else _fail("Requested entity was not found.")

    def _do_gmail_search_people(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "")
        # People search matches name prefixes; try the whole query, then each word.
        people = self._people(query) or [
            person for word in query.split() for person in self._people(word)
        ]
        unique = list({person["resourceName"]: person for person in people}.values())
        return _ok({"results": [{"person": person} for person in unique]})

    # -- end state ---------------------------------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "sent": [sent.to_dict() for sent in self.sent],
            "drafts": [asdict(draft) for draft in self.drafts.values()],
            "decisions": list(self.decisions),
            "calls": list(self.calls),
            "inbox": [asdict(m) for m in self.messages.values() if "SENT" not in m.labels],
        }


# -- query matching ---------------------------------------------------------------------------------------------


def _date(value: str) -> Optional[str]:
    match = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", value)
    return f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}" if match else None


def _term(message: Message, op: Optional[str], value: str) -> bool:
    value = value.strip('"').lower()
    people = " ".join([message.sender, *message.to, *message.cc]).lower()
    if op is None:
        haystack = " ".join([people, message.subject, message.body, *message.attachments]).lower()
        return value in haystack
    op = op.lower()
    labels = {label.upper() for label in message.labels}
    if op == "from":
        return value in message.sender.lower()
    if op == "to":
        return value in " ".join([*message.to, *message.cc, *message.bcc]).lower()
    if op in {"cc", "bcc"}:
        return value in " ".join(getattr(message, op)).lower()
    if op == "subject":
        return value in message.subject.lower()
    if op == "in":
        return value in {"anywhere", "all"} or value.upper() in labels
    if op == "is":
        if value == "read":
            return "UNREAD" not in labels
        return value.upper() in labels
    if op == "label":
        return value.upper() in labels
    if op == "has":
        return bool(message.attachments) if value == "attachment" else True
    if op == "filename":
        return any(value in name.lower() for name in message.attachments)
    if op in {"after", "before"}:
        day = _date(value)
        if day is None:
            return True
        return message.timestamp[:10] >= day if op == "after" else message.timestamp[:10] < day
    return True  # operators the fake does not model (newer_than:, category:, ...) filter nothing


def _matches(message: Message, query: str) -> bool:
    cleaned = re.sub(r"[(){}]", " ", query or "").strip()
    if not cleaned:
        return True
    for group in re.split(r"\s+OR\s+", cleaned):
        terms = [m for m in _QUERY_TERM.finditer(group) if m.group(0).strip()]
        if all(
            (not _term(message, m.group(2), m.group(3) or m.group(4) or m.group(5) or "")) == bool(m.group(1))
            for m in terms
        ):
            return True
    return False


__all__ = ["Contact", "Draft", "FakeGmail", "Message", "OWNER", "SentMail", "VIRTUAL_START", "addresses"]
