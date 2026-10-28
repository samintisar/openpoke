from typing import Optional, Union

from fastapi import status
from fastapi.responses import JSONResponse, PlainTextResponse

from ...config import get_settings
from ...logging_config import logger
from ...models import ChatMessage, ChatRequest
from ...utils import error_response
from ..inbox import get_inbox


# Extract the most recent user message from the chat request payload
def _extract_latest_user_message(payload: ChatRequest) -> Optional[ChatMessage]:
    for message in reversed(payload.messages):
        if message.role.lower().strip() == "user" and message.content.strip():
            return message
    return None


# Accept a chat message: post it to the interaction inbox, whose consumer runs the interaction-agent turn
async def handle_chat_request(payload: ChatRequest) -> Union[PlainTextResponse, JSONResponse]:
    """Queue the user's message for the interaction agent and return 202 at once; the reply shows up in history."""

    # Extract user message
    user_message = _extract_latest_user_message(payload)
    if user_message is None:
        return error_response("Missing user message", status_code=status.HTTP_400_BAD_REQUEST)

    user_content = user_message.content.strip()  # Already checked in _extract_latest_user_message

    logger.info("chat request", extra={"message_length": len(user_content)})

    if not get_settings().openrouter_api_key:
        # Missing API key error (the turn could never run)
        error = "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
        logger.error("configuration error", extra={"error": error})
        return error_response(error, status_code=status.HTTP_400_BAD_REQUEST)

    get_inbox().post("user", user_content)

    return PlainTextResponse("", status_code=status.HTTP_202_ACCEPTED)
