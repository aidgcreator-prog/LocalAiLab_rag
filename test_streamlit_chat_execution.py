from agent_registry import AgentType
from streamlit_chat_execution import (
    get_instant_chat_response,
    invoke_agent_route,
    resolve_chat_agent_type,
    should_bypass_agent_for_trivial_turn,
)


def test_resolve_chat_agent_type_prefers_rag_knowledge_only():
    agent_type, auto_fast_route = resolve_chat_agent_type(
        user_message="anything",
        selected_agent="Auto (Main Agent Decides)",
        agent_display_options={"Auto (Main Agent Decides)": AgentType.AUTO.value},
        rag_knowledge_only=True,
        force_write_todos=False,
    )

    assert agent_type == AgentType.RAG_SUB.value
    assert auto_fast_route is None


def test_resolve_chat_agent_type_fast_routes_simple_specialist_request():
    agent_type, auto_fast_route = resolve_chat_agent_type(
        user_message="search the web for current ollama release notes",
        selected_agent="Auto (Main Agent Decides)",
        agent_display_options={"Auto (Main Agent Decides)": AgentType.AUTO.value},
        rag_knowledge_only=False,
        force_write_todos=False,
    )

    assert agent_type == AgentType.WEBSEARCH.value
    assert auto_fast_route == AgentType.WEBSEARCH.value


def test_get_instant_chat_response_handles_trivial_social_turns():
    assert get_instant_chat_response("hi") == "Hi. What do you need?"
    assert get_instant_chat_response("Thanks!") == "You're welcome."
    assert get_instant_chat_response("implement a caching layer") is None


def test_should_bypass_agent_for_trivial_turn_only_blocks_real_context():
    assert should_bypass_agent_for_trivial_turn(user_text="hello")
    assert not should_bypass_agent_for_trivial_turn(user_text="hello", has_attachments=True)
    assert not should_bypass_agent_for_trivial_turn(user_text="hello", has_selected_rag_context=True)


def test_invoke_agent_route_uses_specialist_router_branch():
    observed = {}

    def _run_async(result):
        return result

    def _invoke_specialist_router(**kwargs):
        observed.update(kwargs)
        return {"output": "done"}

    result = invoke_agent_route(
        agent_type=AgentType.WEBSEARCH.value,
        auto_fast_route=AgentType.WEBSEARCH.value,
        user_message="search",
        history=[("user", "search")],
        session_id="thread-1",
        user_id="user-1",
        run_async=_run_async,
        get_cached_data_scientist_bundle=lambda: (None, None),
        get_cached_ragsub_bundle=lambda: (None, None),
        get_cached_presenter_bundle=lambda: (None, None),
        get_cached_specialist_router_bundle=lambda: (None, _invoke_specialist_router),
        build_rag_messages=lambda _msg: [],
        register_analysis_result=lambda *_args, **_kwargs: None,
        run_main_agent_streaming_fn=lambda *_args, **_kwargs: {"output": "stream"},
        run_main_agent_fn=lambda *_args, **_kwargs: {"output": "main"},
        main_agent_fast_mode=False,
    )

    assert result == {"output": "done"}
    assert observed["subagent_type"] == AgentType.WEBSEARCH.value
    assert observed["thread_id"] == "thread-1"


def test_invoke_agent_route_uses_rag_builder_and_registers_metadata():
    observed = {"registered": []}

    def _run_async(result):
        return result

    def _invoke_rag(**kwargs):
        observed["rag_call"] = kwargs
        return {"output": "rag answer"}

    result = invoke_agent_route(
        agent_type=AgentType.RAG_SUB.value,
        auto_fast_route=None,
        user_message="rag question",
        history=[("user", "ignored")],
        session_id="thread-2",
        user_id="user-2",
        run_async=_run_async,
        get_cached_data_scientist_bundle=lambda: (None, None),
        get_cached_ragsub_bundle=lambda: (None, _invoke_rag),
        get_cached_presenter_bundle=lambda: (None, None),
        get_cached_specialist_router_bundle=lambda: (None, None),
        build_rag_messages=lambda message: [("user", f"rag::{message}")],
        register_analysis_result=lambda *args: observed["registered"].append(args),
        run_main_agent_streaming_fn=lambda *_args, **_kwargs: {"output": "stream"},
        run_main_agent_fn=lambda *_args, **_kwargs: {"output": "main"},
        main_agent_fast_mode=False,
    )

    assert result == {"output": "rag answer"}
    assert observed["rag_call"]["messages"] == [("user", "rag::rag question")]
    assert observed["registered"][0][0] == "rag"
