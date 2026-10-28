"""Execution Agent implementation."""

from pathlib import Path
from typing import List, Optional, Dict, Any

from ...services.agents.memory import load_entries, render_history
from ...services.agents.registry import get_agent_directory
from ...services.execution import get_execution_agent_logs
from ...logging_config import logger


# Load system prompt template from file
_prompt_path = Path(__file__).parent / "system_prompt.md"
if _prompt_path.exists():
    SYSTEM_PROMPT_TEMPLATE = _prompt_path.read_text(encoding="utf-8").strip()
else:
    # Placeholder template - you'll replace this with actual instructions
    SYSTEM_PROMPT_TEMPLATE = """You are an execution agent responsible for completing specific tasks using available tools.

Agent Name: {agent_name}
Purpose: {agent_purpose}

Instructions:
[TO BE FILLED IN BY USER]

You have access to Gmail tools to help complete your tasks. When given instructions:
1. Analyze what needs to be done
2. Use the appropriate tools to complete the task
3. Provide clear status updates on your actions

Be thorough, accurate, and efficient in your execution."""


class ExecutionAgent:
    """Manages state and history for an execution agent."""

    # Initialize execution agent with name, conversation limits, and log store access
    def __init__(
        self,
        agent_id: str,
        name: Optional[str] = None,
        conversation_limit: Optional[int] = None,
        directory: Optional[Any] = None,
    ):
        """
        Initialize an execution agent.

        Args:
            agent_id: Registry id; keys the agent's log
            name: Human-readable agent name for prompts (e.g., 'conversation with keith'); defaults to the id
            conversation_limit: Optional limit on past conversations to include (None = all)
            directory: Agent directory for the card's purpose; defaults to the registry
        """
        self.agent_id = agent_id
        self.name = name or agent_id
        self.conversation_limit = conversation_limit
        self._log_store = get_execution_agent_logs()
        self._directory = directory

    def _purpose(self) -> str:
        """The card's purpose (refreshed by compaction), falling back to the agent name."""
        try:
            card = (self._directory or get_agent_directory()).get(self.agent_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[{self.agent_id}] Registry lookup failed: {exc}")
            card = None
        purpose = (card.purpose or "").strip() if card else ""
        return purpose or self.name

    # Generate system prompt template with agent name and the card's purpose
    def build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.name,
            agent_purpose=self._purpose()
        )

    # Combine base system prompt with conversation history, applying conversation limits
    def build_system_prompt_with_history(self) -> str:
        """
        Build system prompt including agent history.

        Returns:
            System prompt with embedded history transcript
        """
        base_prompt = self.build_system_prompt()

        # With a compacted memory: memory + identifiers + the raw entries after it
        memory = self._log_store.latest_memory(self.agent_id)
        if memory is not None:
            entries = load_entries(self._log_store, self.agent_id)
            history = render_history(
                memory.memory, memory.identifiers, entries[memory.upto:], context_entries=entries[:memory.upto]
            )
            return f"{base_prompt}\n\n# Execution History\n\n{history}"

        # Load history transcript
        transcript = self._log_store.load_transcript(self.agent_id)

        if transcript:
            # Apply conversation limit if needed
            if self.conversation_limit and self.conversation_limit > 0:
                # Parse entries and limit them
                lines = transcript.split('\n')
                request_count = sum(1 for line in lines if '<agent_request' in line)

                if request_count > self.conversation_limit:
                    # Find where to cut
                    kept_requests = 0
                    cutoff_index = len(lines)
                    for i in range(len(lines) - 1, -1, -1):
                        if '<agent_request' in lines[i]:
                            kept_requests += 1
                            if kept_requests == self.conversation_limit:
                                cutoff_index = i
                                break
                    transcript = '\n'.join(lines[cutoff_index:])

            return f"{base_prompt}\n\n# Execution History\n\n{transcript}"

        return base_prompt

    # Format current instruction as user message for LLM consumption
    def build_messages_for_llm(self, current_instruction: str) -> List[Dict[str, str]]:
        """
        Build message array for LLM call.

        Args:
            current_instruction: Current instruction from interaction agent

        Returns:
            List of messages in OpenRouter format
        """
        return [
            {"role": "user", "content": current_instruction}
        ]

    # Log the agent's final response to the execution log store
    def record_response(self, response: str) -> None:
        """Record agent's response to the log."""
        self._log_store.record_agent_response(self.agent_id, response)

    # Log tool invocation and results with truncated content for readability
    def record_tool_execution(self, tool_name: str, arguments: str, result: str) -> None:
        """Record tool execution details."""
        self._log_store.record_action(self.agent_id, f"Calling {tool_name} with: {arguments[:200]}")
        # Record the tool response
        self._log_store.record_tool_response(self.agent_id, tool_name, result[:500])
