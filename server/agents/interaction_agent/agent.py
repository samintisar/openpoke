"""Interaction agent helpers for prompt construction."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ...services.agents import build_query, pinned_from_agent_message, render_cards, select
from ...services.agents.registry import get_agent_directory
from ...services.conversation import get_conversation_log

# Conversation turns added to the routing query, besides the new message.
RECENT_TURNS = 3

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
    now: Optional[datetime] = None,
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, the relevant agent cards, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(_render_active_agents(latest_text, message_type, now or datetime.now(timezone.utc)))
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# The conversation turns before the latest message, for the routing query
def _recent_turn_texts(latest_text: str) -> List[str]:
    texts = get_conversation_log().recent_texts(RECENT_TURNS + 1)
    # The runtime records the latest message before building the prompt; it is already the query's head.
    if texts and texts[-1].strip() == latest_text.strip():
        texts = texts[:-1]
    return texts[-RECENT_TURNS:]


# Render the agent cards most relevant to this turn (agent results pin the agents that reported)
def _render_active_agents(latest_text: str, message_type: str, now: datetime) -> str:
    directory = get_agent_directory()
    cards = directory.list_cards()
    pinned = pinned_from_agent_message(latest_text, cards) if message_type == "agent" else []
    query = build_query(latest_text, _recent_turn_texts(latest_text))
    selected = select(query, cards, now, directory.last_routed(), pinned=pinned)
    return render_cards(selected, len(cards), now)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
