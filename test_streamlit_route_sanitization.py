from agent_registry import AgentType
from streamlit_chat_execution import invoke_agent_route


def test_invoke_agent_route_sanitizes_structured_messages_for_main_agent():
    observed = {}

    def _run_async(result):
        return result

    def _run_main_agent(user_message, history):
        observed["user_message"] = user_message
        observed["history"] = history
        return {"output": "ok"}

    result = invoke_agent_route(
        agent_type=AgentType.AUTO.value,
        auto_fast_route=None,
        user_message=[
            {"type": "text", "text": "Draft a bio."},
            {"type": "file", "filename": "bio.docx"},
        ],
        history=[
            (
                "user",
                [
                    {"type": "text", "text": "Use this file"},
                    {"type": "file", "filename": "bio.docx"},
                ],
            )
        ],
        session_id="thread-1",
        user_id="user-1",
        run_async=_run_async,
        get_cached_data_scientist_bundle=lambda: (None, None),
        get_cached_ragsub_bundle=lambda: (None, None),
        get_cached_presenter_bundle=lambda: (None, None),
        get_cached_specialist_router_bundle=lambda: (None, None),
        build_rag_messages=lambda _msg: [],
        register_analysis_result=lambda *_args, **_kwargs: None,
        run_main_agent_streaming_fn=lambda *_args, **_kwargs: {"output": "stream"},
        run_main_agent_fn=_run_main_agent,
        main_agent_fast_mode=True,
    )

    assert result == {"output": "ok"}
    assert observed["user_message"] == "Draft a bio.\n[Attached file: bio.docx]"
    assert observed["history"] == [("user", "Use this file\n[Attached file: bio.docx]")]
