from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from message_sanitizer import coerce_message_content_to_text


class InputContentSanitizerMiddleware(AgentMiddleware):
    """Normalize structured message content into plain text before model calls.

    Some clients send message content as block lists (for example, `text`, `file`,
    `image`). Text-only model paths fail when unsupported block types are passed
    through unchanged. This middleware rewrites message content to plain text while
    preserving message order and roles.
    """

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        changed = False
        rewritten: list[Any] = []

        for message in messages:
            updated_message = self._sanitize_message(message)
            if updated_message is not message:
                changed = True
            rewritten.append(updated_message)

        if not changed:
            return None
        return {"messages": Overwrite(rewritten)}

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Sanitize request messages right before the model call."""
        sanitized_request = self._sanitize_model_request(request)
        return handler(sanitized_request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Async counterpart of wrap_model_call."""
        sanitized_request = self._sanitize_model_request(request)
        return await handler(sanitized_request)

    def _sanitize_model_request(self, request: Any) -> Any:
        messages = list(getattr(request, "messages", []) or [])
        if not messages and not hasattr(request, "system_message"):
            return request

        changed = False
        rewritten_messages: list[Any] = []
        for message in messages:
            updated_message = self._sanitize_message(message)
            if updated_message is not message:
                changed = True
            rewritten_messages.append(updated_message)

        rewritten_system = getattr(request, "system_message", None)
        if rewritten_system is not None:
            updated_system = self._sanitize_message(rewritten_system)
            if updated_system is not rewritten_system:
                rewritten_system = updated_system
                changed = True

        if not changed or not hasattr(request, "override"):
            return request

        update_payload: dict[str, Any] = {"messages": rewritten_messages}
        if hasattr(request, "system_message"):
            update_payload["system_message"] = rewritten_system
        return request.override(**update_payload)

    def _sanitize_message(self, message: Any) -> Any:
        # Tuple-based messages: (role, content)
        if isinstance(message, tuple) and len(message) == 2:
            role, content = message
            sanitized = coerce_message_content_to_text(content)
            if sanitized != content:
                return (role, sanitized)
            return message

        # Dict-based messages: {"role": ..., "content": ...}
        if isinstance(message, dict) and "content" in message:
            content = message.get("content")
            sanitized = coerce_message_content_to_text(content)
            if sanitized != content:
                copy = dict(message)
                copy["content"] = sanitized
                return copy
            return message

        # LangChain message objects (HumanMessage/AIMessage/ToolMessage/...)
        content = getattr(message, "content", None)
        if content is None:
            return message

        sanitized = coerce_message_content_to_text(content)
        if sanitized == content:
            return message

        if hasattr(message, "model_copy"):
            try:
                return message.model_copy(update={"content": sanitized})
            except Exception:
                return message

        return message
