"""Tool definitions for interaction agent."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ...logging_config import logger
from ...services import approvals
from ...services.agents import TOOL_SCHEMAS, dispatch, search_payload
from ...services.agents.registry import get_agent_directory
from ...services.conversation import get_conversation_log
from ...services.execution import get_execution_agent_logs
from ...services.execution.runner import get_agent_runner
from .turn import current_turn


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False


# Reuse or create an execution agent through the registry and dispatch instructions asynchronously
def send_message_to_agent(
    instructions: str,
    agent_id: Optional[str] = None,
    name: Optional[str] = None,
    purpose: Optional[str] = None,
    force_new: bool = False,
) -> ToolResult:
    """Send instructions to an execution agent, reusing `agent_id` or creating one from `name`/`purpose`."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    result = dispatch(
        get_agent_directory(),
        instructions=instructions,
        agent_id=agent_id,
        name=name,
        purpose=purpose,
        force_new=bool(force_new),
        now=datetime.now(timezone.utc),
    )
    if result.action not in {"reused", "created"} or result.card is None:
        # possible_duplicate or error: nothing was sent; the payload says what to do next.
        logger.info(f"send_message_to_agent not dispatched: {result.action}")
        return ToolResult(success=False, payload=result.payload)

    card = result.card
    get_execution_agent_logs().record_request(card.id, instructions)
    logger.info(f"{result.action.capitalize()} agent: {card.name} ({card.id})")

    async def _execute_async() -> None:
        try:
            outcome = await get_agent_runner().run(card.id, instructions, origin="interaction", agent_name=card.name)
            status = "PARTIAL" if outcome.partial else ("SUCCESS" if outcome.success else "FAILED")
            logger.info(f"Agent '{card.name}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{card.name}' failed: {str(exc)}")

    loop.create_task(_execute_async())

    return ToolResult(success=True, payload=result.payload)


# Search every agent card, archived included, when nothing listed fits
def search_agents(query: str) -> ToolResult:
    """Return the rendered best-matching agent cards for `query`."""
    payload = search_payload(get_agent_directory(), query, datetime.now(timezone.utc))
    return ToolResult(success=True, payload=payload)


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    log = get_conversation_log()
    log.record_reply(message)

    return ToolResult(
        success=True,
        payload={"status": "delivered"},
        user_message=message,
        recorded_reply=True,
    )


SEND_DRAFT_ARGUMENTS_ERROR = "send_draft() needs action_id, or all of to, subject and body"


# Format and record email draft for user review
def send_draft(
    to: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    action_id: Optional[Any] = None,
) -> ToolResult:
    """Record a draft for the user to review.

    With `action_id`, shows exactly the stored content of that pending send (not the arguments) and marks it
    shown now, so the user's next message can approve it. Without one, records the given text and approves
    nothing.
    """
    log = get_conversation_log()

    if action_id is not None:
        try:
            action = approvals.show(action_id)
        except approvals.ApprovalError as exc:
            return ToolResult(success=False, payload={"error": str(exc)})
        log.record_reply(approvals.render(action))
        logger.info(f"Pending action {action.id} shown to the user")
        return ToolResult(
            success=True,
            payload={
                "status": "draft_recorded",
                "action_id": action.id,
                "to": ", ".join(action.recipients.get("to") or []),
                "subject": action.subject,
            },
            recorded_reply=True,
        )

    if to is None or subject is None or body is None:
        # Raised like a signature error, so handle_tool_call reports it as one (and eval shims can mirror it).
        raise TypeError(SEND_DRAFT_ARGUMENTS_ERROR)

    message = f"To: {to}\nSubject: {subject}\n\n{body}"

    log.record_reply(message)
    logger.info(f"Draft recorded for: {to}")

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "to": to,
            "subject": subject,
        },
        recorded_reply=True,
    )


# Approve one pending send; only the user's own message, written after they saw it, can do this
def approve_send(action_id: Any) -> ToolResult:
    """Approve a shown, pending send for its owning agent to deliver."""
    turn = current_turn.get()
    try:
        action = approvals.approve(
            action_id,
            origin=turn.origin if turn else None,
            user_message_at=turn.user_message_at if turn else None,
        )
    except approvals.ApprovalError as exc:
        return ToolResult(success=False, payload={"error": str(exc)})
    return ToolResult(
        success=True,
        payload={
            "status": "approved",
            "action_id": action.id,
            "agent_id": action.agent_id,
            "next": (
                f"Tell agent {action.agent_id} (send_message_to_agent with agent_id={action.agent_id}) that the "
                f"user approved action {action.id} and to send it now, unchanged."
            ),
        },
    )


# Record silent wait state to avoid duplicate responses
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return the shared tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
def handle_tool_call(name: str, arguments: Any) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        if name == "send_message_to_agent":
            return send_message_to_agent(**args)
        if name == "search_agents":
            return search_agents(**args)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "approve_send":
            return approve_send(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
