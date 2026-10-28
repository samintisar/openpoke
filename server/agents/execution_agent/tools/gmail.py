"""Gmail tool schemas and actions for the execution agent."""

from __future__ import annotations

import base64
import json
from email.header import decode_header, make_header
from typing import Any, Callable, Dict, List, Optional

from server.logging_config import logger
from server.services.approvals import gate
from server.services.approvals.gate import SendContent
from server.services.execution import get_execution_agent_logs
from server.services.gmail import execute_gmail_tool, get_active_gmail_user_id, wrap_message_text_fields

_GMAIL_AGENT_NAME = "gmail-execution-agent"

_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "gmail_create_draft",
            "description": "Create a Gmail draft via Composio, supporting html/plain bodies, cc/bcc, and attachments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_email": {
                        "type": "string",
                        "description": "Primary recipient email for the draft.",
                    },
                    "subject": {"type": "string", "description": "Email subject."},
                    "body": {
                        "type": "string",
                        "description": "Email body. Use HTML markup when is_html is true.",
                    },
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of CC recipient emails.",
                    },
                    "bcc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of BCC recipient emails.",
                    },
                    "extra_recipients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional recipients if the draft should include more addresses.",
                    },
                    "is_html": {
                        "type": "boolean",
                        "description": "Set true when the body contains HTML content.",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Existing Gmail thread id if this draft belongs to a thread.",
                    },
                    "attachment": {
                        "type": "object",
                        "description": "Single attachment metadata (requires Composio-uploaded asset).",
                        "properties": {
                            "s3key": {"type": "string", "description": "S3 key of uploaded file."},
                            "name": {"type": "string", "description": "Attachment filename."},
                            "mimetype": {"type": "string", "description": "Attachment MIME type."},
                        },
                        "required": ["s3key", "name", "mimetype"],
                    },
                },
                "required": ["recipient_email", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_execute_draft",
            "description": (
                "Send a previously created Gmail draft. It sends only if the user approved this exact draft; "
                "otherwise it returns approval_required with an action_id and sends nothing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "Identifier of the Gmail draft to send.",
                    },
                },
                "required": ["draft_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_forward_email",
            "description": (
                "Forward an existing Gmail message with optional additional context. It sends only if the user "
                "approved this exact forward; otherwise it returns approval_required with an action_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message id to forward.",
                    },
                    "recipient_email": {
                        "type": "string",
                        "description": "Email address to receive the forwarded message.",
                    },
                    "additional_text": {
                        "type": "string",
                        "description": "Optional text to prepend when forwarding.",
                    },
                },
                "required": ["message_id", "recipient_email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_reply_to_thread",
            "description": (
                "Send a reply within an existing Gmail thread. It sends only if the user approved this exact "
                "reply; otherwise it returns approval_required with an action_id and sends nothing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "Gmail thread id to reply to.",
                    },
                    "recipient_email": {
                        "type": "string",
                        "description": "Primary recipient for the reply (usually the original sender).",
                    },
                    "message_body": {
                        "type": "string",
                        "description": "Reply body. Use HTML markup when is_html is true.",
                    },
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of CC recipient emails.",
                    },
                    "bcc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of BCC recipient emails.",
                    },
                    "extra_recipients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional recipients if needed.",
                    },
                    "is_html": {
                        "type": "boolean",
                        "description": "Set true when the body contains HTML content.",
                    },
                    "attachment": {
                        "type": "object",
                        "description": "Single attachment metadata (requires Composio-uploaded asset).",
                        "properties": {
                            "s3key": {"type": "string", "description": "S3 key of uploaded file."},
                            "name": {"type": "string", "description": "Attachment filename."},
                            "mimetype": {"type": "string", "description": "Attachment MIME type."},
                        },
                        "required": ["s3key", "name", "mimetype"],
                    },
                },
                "required": ["thread_id", "recipient_email", "message_body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_delete_draft",
            "description": "Delete a specific Gmail draft using the Composio Gmail integration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "Identifier of the Gmail draft to delete.",
                    },
                },
                "required": ["draft_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_get_contacts",
            "description": "Retrieve Google contacts (connections) available to the authenticated Gmail account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "Resource name to read contacts from, defaults to people/me.",
                    },
                    "person_fields": {
                        "type": "string",
                        "description": "Comma-separated People API fields to include (e.g. emailAddresses,names).",
                    },
                    "include_other_contacts": {
                        "type": "boolean",
                        "description": "Include other contacts (directory suggestions) when true.",
                    },
                    "page_token": {
                        "type": "string",
                        "description": "Pagination token for retrieving the next page of contacts.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_get_people",
            "description": "Retrieve detailed Google People records or other contacts via Composio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "Resource name to fetch (defaults to people/me).",
                    },
                    "person_fields": {
                        "type": "string",
                        "description": "Comma-separated People API fields to include in the response.",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Maximum number of people records to return per page.",
                    },
                    "page_token": {
                        "type": "string",
                        "description": "Token to continue fetching the next set of results.",
                    },
                    "sync_token": {
                        "type": "string",
                        "description": "Sync token for incremental sync requests.",
                    },
                    "other_contacts": {
                        "type": "boolean",
                        "description": "Set true to list other contacts instead of connections.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_list_drafts",
            "description": "List Gmail drafts for the connected account using Composio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of drafts to return.",
                    },
                    "page_token": {
                        "type": "string",
                        "description": "Pagination token from a previous drafts list call.",
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Include full draft details such as subject and body when true.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_search_people",
            "description": "Search Google contacts and other people records associated with the Gmail account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to match against names, emails, phone numbers, etc.",
                    },
                    "person_fields": {
                        "type": "string",
                        "description": "Comma-separated fields from the People API to include in results.",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Maximum number of people records to return.",
                    },
                    "other_contacts": {
                        "type": "boolean",
                        "description": "Include other contacts results when true.",
                    },
                    "page_token": {
                        "type": "string",
                        "description": "Pagination token to continue a previous search.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

_LOG_STORE = get_execution_agent_logs()


# Return Gmail tool schemas
def get_schemas() -> List[Dict[str, Any]]:
    """Return Gmail tool schemas."""
    
    return _SCHEMAS


# Execute a Gmail tool and record the action for the execution agent journal
def _execute(tool_name: str, composio_user_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a Gmail tool and record the action for the execution agent journal."""

    payload = {k: v for k, v in arguments.items() if v is not None}
    payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True) if payload else "{}"
    try:
        result = execute_gmail_tool(tool_name, composio_user_id, arguments=payload)
    except Exception as exc:
        _LOG_STORE.record_action(
            _GMAIL_AGENT_NAME,
            description=f"{tool_name} failed | args={payload_str} | error={exc}",
        )
        raise

    _LOG_STORE.record_action(
        _GMAIL_AGENT_NAME,
        description=f"{tool_name} succeeded | args={payload_str}",
    )
    return result


# -- send approval gate helpers -------------------------------------------------------------------------------


def _current_agent_id() -> Optional[str]:
    """The agent whose run is calling the tool (None outside a run, which the gate refuses)."""
    from server.services.execution.runner import current_run  # imported late: the runner imports the runtime

    run = current_run.get()
    return run.agent_id if run else None


def _succeeded(result: Any) -> bool:
    return isinstance(result, dict) and not result.get("error") and result.get("successful") is not False


def _result_data(result: Any) -> Any:
    """Composio's `data` field (parsed when it is a JSON string), else the result itself."""
    data = result.get("data", result) if isinstance(result, dict) else result
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None
    return data


def _find_key(data: Any, keys: tuple) -> Optional[str]:
    """The first of `keys` found in `data`, breadth first (a draft's id sits above its message's id)."""
    queue = [data]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            for key in keys:
                value = node.get(key)
                if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
                    return str(value)
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return None


def _approval_required(action: Any, note: str) -> Dict[str, Any]:
    return {
        "status": "approval_required",
        "action_id": action.id,
        "message": (
            f"{note} Nothing was sent. Report action_id {action.id} to Poke and stop; do not retry the send. "
            "It goes through once the user approves this exact content."
        ),
    }


def _refused(reason: str) -> Dict[str, Any]:
    return {"error": f"Not sent: {reason}", "status": "refused"}


def _attachment_names(attachment: Optional[Dict[str, Any]]) -> List[str]:
    name = attachment.get("name") if isinstance(attachment, dict) else None
    return [name] if isinstance(name, str) and name else []


def _decode_part(data: Any) -> Optional[str]:
    if not isinstance(data, str) or not data:
        return None
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return None


def _header_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:  # pragma: no cover - malformed encoded words: keep the raw value
        return value


def _walk_parts(part: Dict[str, Any], bodies: Dict[str, str], attachments: List[str]) -> None:
    filename = part.get("filename")
    mime = str(part.get("mimeType") or "").lower()
    if isinstance(filename, str) and filename:
        attachments.append(filename)
    elif mime in {"text/plain", "text/html"} and mime not in bodies:
        text = _decode_part((part.get("body") or {}).get("data"))
        if text is not None:
            bodies[mime] = text
    for child in part.get("parts") or []:
        if isinstance(child, dict):
            _walk_parts(child, bodies, attachments)


def _find_message_with_payload(data: Any) -> Optional[Dict[str, Any]]:
    queue = [data]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            if isinstance(node.get("payload"), dict):
                return node
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return None


def _draft_content(result: Any, body_format: str) -> Optional[SendContent]:
    """What the draft holds in Gmail right now, from a raw GMAIL_GET_DRAFT result; None if unreadable.

    Reads the Gmail API shape (message.payload headers and parts) and falls back to flat fields
    (to/cc/bcc/subject/messageText) for processed payloads. The body is the part matching `body_format`.
    """
    data = _result_data(result)
    if not isinstance(data, (dict, list)):
        return None
    message = _find_message_with_payload(data)
    if message is not None:
        payload = message["payload"]
        headers = {
            str(header.get("name", "")).lower(): _header_text(header.get("value"))
            for header in payload.get("headers") or []
            if isinstance(header, dict)
        }
        bodies: Dict[str, str] = {}
        attachments: List[str] = []
        _walk_parts(payload, bodies, attachments)
        mime = "text/html" if body_format == "html" else "text/plain"
        if mime not in bodies:
            return None
        return SendContent.build(
            to=[headers.get("to")],
            cc=[headers.get("cc")],
            bcc=[headers.get("bcc")],
            subject=headers.get("subject"),
            body=bodies[mime],
            is_html=body_format == "html",
            attachments=attachments,
        )
    flat = data if isinstance(data, dict) else None
    if flat is not None and isinstance(flat.get("draft"), dict):
        flat = flat["draft"]
    if flat is None:
        return None
    body = next((flat[key] for key in ("messageText", "body", "message_body") if isinstance(flat.get(key), str)), None)
    to = flat.get("to") or flat.get("recipient_email")
    if body is None or not to:
        return None

    def _as_list(value: Any) -> List[Any]:
        return list(value) if isinstance(value, list) else [value]

    return SendContent.build(
        to=_as_list(to),
        cc=_as_list(flat.get("cc")),
        bcc=_as_list(flat.get("bcc")),
        subject=flat.get("subject"),
        body=body,
        is_html=body_format == "html",
        attachments=[
            item.get("filename") or item.get("name") if isinstance(item, dict) else item
            for item in _as_list(flat.get("attachments"))
        ],
    )


# Create a Gmail draft via Composio with support for HTML, attachments, and threading
def gmail_create_draft(
    recipient_email: str,
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    extra_recipients: Optional[List[str]] = None,
    is_html: Optional[bool] = None,
    thread_id: Optional[str] = None,
    attachment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "recipient_email": recipient_email,
        "subject": subject,
        "body": body,
        "cc": cc,
        "bcc": bcc,
        "extra_recipients": extra_recipients,
        "is_html": is_html,
        "thread_id": thread_id,
        "attachment": attachment,
    }
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    result = _execute("GMAIL_CREATE_EMAIL_DRAFT", composio_user_id, arguments)
    if not _succeeded(result):
        return result
    draft_id = _find_key(_result_data(result), ("draft_id", "id"))
    if not draft_id:
        logger.warning("draft created without a readable id; it cannot be approved for sending")
        return result
    content = SendContent.build(
        to=[recipient_email, *(extra_recipients or [])],
        cc=cc or [],
        bcc=bcc or [],
        subject=subject,
        body=body,
        is_html=is_html,
        attachments=_attachment_names(attachment),
    )
    action = gate.request_approval("draft", _current_agent_id(), draft_id, content)
    return {**result, "action_id": action.id}


# Send a previously created Gmail draft using Composio, only if the user approved exactly what it holds now
def gmail_execute_draft(
    draft_id: str,
) -> Dict[str, Any]:
    arguments = {"draft_id": draft_id}
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    agent_id = _current_agent_id()
    known = gate.get_approval_store().find(kind="draft", gmail_ref=draft_id)
    body_format = known[0].body_format if known else "plain"

    # Hash the draft as it is in Gmail now (raw fields, not the agent-facing wrapped ones); unreadable refuses.
    try:
        fetched = _execute("GMAIL_GET_DRAFT", composio_user_id, {"draft_id": draft_id, "format": "full"})
    except Exception as exc:
        return _refused(f"could not read draft {draft_id} to verify it ({exc}).")
    content = _draft_content(fetched, body_format) if _succeeded(fetched) else None
    if content is None:
        return _refused(f"could not read draft {draft_id} to verify its content.")

    digest = gate.content_hash("draft", draft_id, content)
    decision = gate.check_send("draft", agent_id, content.all_recipients(), digest, draft_id)
    if not decision.allowed:
        if not agent_id:
            return _refused("no agent run is in progress.")
        action = gate.request_approval("draft", agent_id, draft_id, content)
        note = (
            "The draft changed after the user approved it, so that approval was invalidated."
            if decision.reason == "content_changed"
            else "The user has not approved this draft."
        )
        return _approval_required(action, note)

    result = _execute("GMAIL_SEND_DRAFT", composio_user_id, arguments)
    if decision.action_id is not None and _succeeded(result):
        gate.mark_sent(decision.action_id)
    return result


# Forward an existing Gmail message with optional additional context, once the user approved it
def gmail_forward_email(
    message_id: str,
    recipient_email: str,
    additional_text: Optional[str] = None,
) -> Dict[str, Any]:
    arguments = {
        "message_id": message_id,
        "recipient_email": recipient_email,
        "additional_text": additional_text,
    }
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    content = SendContent.build(to=[recipient_email], body=additional_text)
    return _gated_send("forward", message_id, content, "GMAIL_FORWARD_MESSAGE", composio_user_id, arguments)


def _gated_send(
    kind: str,
    gmail_ref: str,
    content: SendContent,
    tool_name: str,
    composio_user_id: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a reply or forward only when the user approved this exact content; else ask for approval."""
    agent_id = _current_agent_id()
    digest = gate.content_hash(kind, gmail_ref, content)
    decision = gate.check_send(kind, agent_id, content.all_recipients(), digest, gmail_ref)
    if not decision.allowed:
        if not agent_id:
            return _refused("no agent run is in progress.")
        action = gate.request_approval(kind, agent_id, gmail_ref, content)
        return _approval_required(action, f"This {kind} needs the user's approval.")
    result = _execute(tool_name, composio_user_id, arguments)
    if decision.action_id is not None and _succeeded(result):
        gate.mark_sent(decision.action_id)
    return result


# Send a reply within an existing Gmail thread via Composio
def gmail_reply_to_thread(
    thread_id: str,
    recipient_email: str,
    message_body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    extra_recipients: Optional[List[str]] = None,
    is_html: Optional[bool] = None,
    attachment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    arguments = {
        "thread_id": thread_id,
        "recipient_email": recipient_email,
        "message_body": message_body,
        "cc": cc,
        "bcc": bcc,
        "extra_recipients": extra_recipients,
        "is_html": is_html,
        "attachment": attachment,
    }
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    content = SendContent.build(
        to=[recipient_email, *(extra_recipients or [])],
        cc=cc or [],
        bcc=bcc or [],
        body=message_body,
        is_html=is_html,
        attachments=_attachment_names(attachment),
    )
    return _gated_send("reply", thread_id, content, "GMAIL_REPLY_TO_THREAD", composio_user_id, arguments)


# Delete a specific Gmail draft using the Composio Gmail integration
def gmail_delete_draft(
    draft_id: str,
) -> Dict[str, Any]:
    arguments = {"draft_id": draft_id}
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    result = _execute("GMAIL_DELETE_DRAFT", composio_user_id, arguments)
    if _succeeded(result):
        store = gate.get_approval_store()
        for action in store.find(kind="draft", gmail_ref=draft_id, statuses=("pending", "approved")):
            store.transition(action.id, ("pending", "approved"), "invalidated")
    return result


def gmail_get_contacts(
    resource_name: Optional[str] = None,
    person_fields: Optional[str] = None,
    include_other_contacts: Optional[bool] = None,
    page_token: Optional[str] = None,
) -> Dict[str, Any]:
    arguments = {
        "resource_name": resource_name,
        "person_fields": person_fields,
        "include_other_contacts": include_other_contacts,
        "page_token": page_token,
    }
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    return _execute("GMAIL_GET_CONTACTS", composio_user_id, arguments)


def gmail_get_people(
    resource_name: Optional[str] = None,
    person_fields: Optional[str] = None,
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
    sync_token: Optional[str] = None,
    other_contacts: Optional[bool] = None,
) -> Dict[str, Any]:
    arguments = {
        "resource_name": resource_name,
        "person_fields": person_fields,
        "page_size": page_size,
        "page_token": page_token,
        "sync_token": sync_token,
        "other_contacts": other_contacts,
    }
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    return _execute("GMAIL_GET_PEOPLE", composio_user_id, arguments)


def gmail_list_drafts(
    max_results: Optional[int] = None,
    page_token: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> Dict[str, Any]:
    arguments = {
        "max_results": max_results,
        "page_token": page_token,
        "verbose": verbose,
    }
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    # Verbose drafts carry message text (e.g. quoted replies), so mark it untrusted
    return wrap_message_text_fields(_execute("GMAIL_LIST_DRAFTS", composio_user_id, arguments))


def gmail_search_people(
    query: str,
    person_fields: Optional[str] = None,
    page_size: Optional[int] = None,
    other_contacts: Optional[bool] = None,
    page_token: Optional[str] = None,
) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "query": query,
        "person_fields": person_fields,
        "other_contacts": other_contacts,
    }
    if page_size is not None:
        arguments["pageSize"] = page_size
    if page_token is not None:
        arguments["pageToken"] = page_token
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        return {"error": "Gmail not connected. Please connect Gmail in settings first."}
    return _execute("GMAIL_SEARCH_PEOPLE", composio_user_id, arguments)


# Return Gmail tool callables
def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:  # noqa: ARG001
    """Return Gmail tool callables."""
    
    return {
        "gmail_create_draft": gmail_create_draft,
        "gmail_execute_draft": gmail_execute_draft,
        "gmail_delete_draft": gmail_delete_draft,
        "gmail_forward_email": gmail_forward_email,
        "gmail_reply_to_thread": gmail_reply_to_thread,
        "gmail_get_contacts": gmail_get_contacts,
        "gmail_get_people": gmail_get_people,
        "gmail_list_drafts": gmail_list_drafts,
        "gmail_search_people": gmail_search_people,
    }


__all__ = [
    "build_registry",
    "get_schemas",
    "gmail_create_draft",
    "gmail_execute_draft",
    "gmail_delete_draft",
    "gmail_forward_email",
    "gmail_reply_to_thread",
    "gmail_get_contacts",
    "gmail_get_people",
    "gmail_list_drafts",
    "gmail_search_people",
]
