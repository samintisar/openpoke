"""Interaction-agent tool schemas, shared by the server wiring and the eval systems."""

from __future__ import annotations

from typing import Any

SEND_MESSAGE_TO_AGENT = {
    "type": "function",
    "function": {
        "name": "send_message_to_agent",
        "description": (
            "Deliver instructions to an execution agent. To reuse an agent listed in <active_agents> (or found "
            "with search_agents), pass its agent_id. To create a new agent, omit agent_id and pass name and "
            "purpose. If the result is possible_duplicate, nothing was sent: reuse the suggested agent_id, or "
            "resend with force_new=true if the work is truly different."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                "agent_id": {
                    "type": "string",
                    "description": "Id of an existing agent to reuse (the id attribute in <active_agents>).",
                },
                "name": {
                    "type": "string",
                    "description": (
                        "New agents only: short display name naming the person and topic "
                        "(e.g., 'Gary Whitman Lease Renewal Call')."
                    ),
                },
                "purpose": {
                    "type": "string",
                    "description": (
                        "New agents only: one sentence on the ongoing work this agent owns, naming the people "
                        "involved."
                    ),
                },
                "force_new": {
                    "type": "boolean",
                    "description": "Create a new agent even though a possible duplicate was reported.",
                },
            },
            "required": ["instructions"],
            "additionalProperties": False,
        },
    },
}

SEARCH_AGENTS = {
    "type": "function",
    "function": {
        "name": "search_agents",
        "description": (
            "Search every agent, including old and archived ones, by person, email or topic. Use it when no "
            "listed agent fits but the request sounds like ongoing work, before creating a new agent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "People, emails or topic words to search for."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TO_USER = {
    "type": "function",
    "function": {
        "name": "send_message_to_user",
        "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
}

SEND_DRAFT = {
    "type": "function",
    "function": {
        "name": "send_draft",
        "description": "Record an email draft so the user can review the exact text.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email for the draft.",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject for the draft.",
                },
                "body": {
                    "type": "string",
                    "description": "Email body content (plain text).",
                },
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    },
}

WAIT = {
    "type": "function",
    "function": {
        "name": "wait",
        "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                },
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    SEND_MESSAGE_TO_AGENT,
    SEARCH_AGENTS,
    SEND_MESSAGE_TO_USER,
    SEND_DRAFT,
    WAIT,
]


def get_tool_schemas() -> list[dict[str, Any]]:
    return TOOL_SCHEMAS


__all__ = [
    "SEARCH_AGENTS",
    "SEND_DRAFT",
    "SEND_MESSAGE_TO_AGENT",
    "SEND_MESSAGE_TO_USER",
    "TOOL_SCHEMAS",
    "WAIT",
    "get_tool_schemas",
]
