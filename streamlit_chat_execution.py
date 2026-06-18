from __future__ import annotations

import re
from typing import Any, Callable

from agent_registry import AgentType
from message_sanitizer import coerce_message_content_to_text, sanitize_history_pairs
from streamlit_orchestration import AUTO_AGENT_DISPLAY_NAME, infer_auto_fast_route


_GREETING_RE = re.compile(r"^(hi|hello|hey|yo|good morning|good afternoon|good evening)[!. ]*$", re.IGNORECASE)
_THANKS_RE = re.compile(r"^(thanks|thank you|thx)[!. ]*$", re.IGNORECASE)


def get_instant_chat_response(user_text: str) -> str | None:
    """Return an immediate local reply for trivial social turns."""
    text = (user_text or "").strip()
    if not text:
        return None
    if _GREETING_RE.fullmatch(text):
        return "Hi. What do you need?"
    if _THANKS_RE.fullmatch(text):
        return "You're welcome."
    return None


def should_bypass_agent_for_trivial_turn(
    *,
    user_text: str,
    has_attachments: bool = False,
    has_selected_rag_context: bool = False,
) -> bool:
    """Return True when a turn can skip agent orchestration entirely."""
    if has_attachments or has_selected_rag_context:
        return False
    return get_instant_chat_response(user_text) is not None


def resolve_chat_agent_type(
    *,
    user_message: str,
    selected_agent: str,
    agent_display_options: dict[str, str],
    rag_knowledge_only: bool,
    force_write_todos: bool,
    force_agent_type: str | None = None,
    force_main_agent: bool = False,
) -> tuple[str, str | None]:
    """Resolve the effective agent type and optional auto fast-route target."""
    return infer_auto_fast_route(
        user_message,
        selected_agent=selected_agent,
        agent_display_options=agent_display_options,
        rag_knowledge_only=rag_knowledge_only,
        force_write_todos=force_write_todos,
        force_agent_type=force_agent_type,
        force_main_agent=force_main_agent,
    )


def invoke_agent_route(
    *,
    agent_type: str,
    auto_fast_route: str | None,
    user_message: str,
    history: list[tuple[str, str]],
    session_id: str,
    user_id: str | None,
    run_async: Callable[[Any], Any],
    get_cached_data_scientist_bundle: Callable[[], tuple[Any, Callable[..., Any]]],
    get_cached_ragsub_bundle: Callable[[], tuple[Any, Callable[..., Any]]],
    get_cached_presenter_bundle: Callable[[], tuple[Any, Callable[..., Any]]],
    get_cached_specialist_router_bundle: Callable[[], tuple[Any, Callable[..., Any]]],
    build_rag_messages: Callable[[str], list[tuple[str, str]]],
    register_analysis_result: Callable[[str, dict[str, Any], str, str], Any],
    run_main_agent_streaming_fn: Callable[[str, Any, list[tuple[str, str]]], dict],
    run_main_agent_fn: Callable[[str, list[tuple[str, str]]], dict],
    run_direct_main_chat_fn: Callable[[str], dict],
    main_chat_use_direct_model: bool,
    status_container: Any = None,
    status_steps: list[str] | None = None,
) -> dict:
    """Invoke the resolved agent route and return its result."""
    sanitized_history = sanitize_history_pairs(history)
    sanitized_user_message = coerce_message_content_to_text(user_message)

    if agent_type == AgentType.DATA_SCIENTIST.value:
        _, invoke_ds = get_cached_data_scientist_bundle()
        result = run_async(
            invoke_ds(messages=sanitized_history, thread_id=session_id, user_id=user_id)
        )
        if result and "output" in result:
            register_analysis_result(
                "analysis",
                {"output": str(result.get("output", ""))[:500]},
                "Data Scientist",
                session_id,
            )
        return result

    if agent_type == AgentType.RAG_SUB.value:
        _, invoke_rag = get_cached_ragsub_bundle()
        rag_messages = sanitize_history_pairs(build_rag_messages(sanitized_user_message))
        result = run_async(
            invoke_rag(messages=rag_messages, thread_id=session_id, user_id=user_id)
        )
        if result and "output" in result:
            register_analysis_result(
                "rag",
                {"output": str(result.get("output", ""))[:500]},
                "RAG SubAgent",
                session_id,
            )
        return result

    if agent_type == AgentType.PRESENTER.value:
        _, invoke_pres = get_cached_presenter_bundle()
        return run_async(
            invoke_pres(messages=sanitized_history, thread_id=session_id, user_id=user_id)
        )

    if agent_type in {
        AgentType.WEBSEARCH.value,
        AgentType.WRITER.value,
        AgentType.CODER.value,
        AgentType.PLANNER.value,
        AgentType.REVIEWER.value,
    }:
        _, invoke_specialist_router = get_cached_specialist_router_bundle()
        return run_async(
            invoke_specialist_router(
                subagent_type=agent_type,
                messages=sanitized_history,
                thread_id=session_id,
                user_id=user_id,
            )
        )

    if main_chat_use_direct_model:
        return run_direct_main_chat_fn(sanitized_user_message)

    if status_container is not None:
        return run_main_agent_streaming_fn(
            sanitized_user_message,
            status_container,
            messages=sanitized_history,
        )

    return run_main_agent_fn(sanitized_user_message, sanitized_history)
