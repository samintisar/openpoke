"""Mark email content as untrusted data before it reaches an agent.

Every email body or snippet handed to an LLM is wrapped as
``<untrusted_email from="..." thread_id="...">...</untrusted_email>``. The prompts tell agents that
anything inside the tag is data, so instructions in an email never authorize a send, forward or new
recipient. The content cannot break out of the tag: any ``<untrusted_email`` or ``</untrusted_email``
inside it is escaped, and attribute values are HTML-escaped.
"""

from __future__ import annotations

import html
import re
from typing import Any, Mapping, Optional

TAG = "untrusted_email"

# Opening or closing tag, case-insensitive, tolerating whitespace after "<" or "</".
_TAG_PATTERN = re.compile(r"<(\s*/?\s*untrusted_email)", re.IGNORECASE)

# Fields in Composio/Gmail message payloads that carry message text.
MESSAGE_TEXT_FIELDS = ("messageText", "snippet", "textBody", "htmlBody")


def escape_untrusted(text: str) -> str:
    """Neutralize any untrusted_email tag inside ``text`` so it cannot close or open a wrapper."""

    return _TAG_PATTERN.sub(r"&lt;\1", text)


def _attribute(value: Optional[str]) -> str:
    cleaned = " ".join(str(value or "").split())
    return html.escape(cleaned, quote=True)


def wrap_untrusted_email(text: Optional[str], *, sender: Optional[str], thread_id: Optional[str]) -> str:
    """Return ``text`` wrapped in an ``<untrusted_email>`` tag with escaped content and attributes."""

    body = escape_untrusted(text or "")
    return (
        f'<{TAG} from="{_attribute(sender)}" thread_id="{_attribute(thread_id)}">'
        f"{body}</{TAG}>"
    )


def _message_sender(message: Mapping[str, Any]) -> Optional[str]:
    value = message.get("sender") or message.get("from")
    return value if isinstance(value, str) else None


def _message_thread(message: Mapping[str, Any]) -> Optional[str]:
    value = message.get("threadId") or message.get("thread_id")
    return value if isinstance(value, str) else None


def wrap_message_text_fields(payload: Any) -> Any:
    """Return a copy of a Composio Gmail result with every message-text field wrapped.

    Walks dicts and lists. In any dict, string values under ``MESSAGE_TEXT_FIELDS`` and
    ``preview.body`` are wrapped with that dict's sender and thread id. Everything else (ids,
    labels, headers, base64 payload parts) is left as is.
    """

    if isinstance(payload, list):
        return [wrap_message_text_fields(item) for item in payload]
    if not isinstance(payload, dict):
        return payload

    sender = _message_sender(payload)
    thread_id = _message_thread(payload)
    result: dict = {}
    for key, value in payload.items():
        if key in MESSAGE_TEXT_FIELDS and isinstance(value, str):
            result[key] = wrap_untrusted_email(value, sender=sender, thread_id=thread_id)
        elif key == "preview" and isinstance(value, dict) and isinstance(value.get("body"), str):
            preview = dict(value)
            preview["body"] = wrap_untrusted_email(value["body"], sender=sender, thread_id=thread_id)
            result[key] = preview
        else:
            result[key] = wrap_message_text_fields(value)
    return result


__all__ = [
    "MESSAGE_TEXT_FIELDS",
    "TAG",
    "escape_untrusted",
    "wrap_message_text_fields",
    "wrap_untrusted_email",
]
