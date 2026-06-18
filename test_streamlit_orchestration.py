from agent_registry import AgentType, get_registry
from streamlit_orchestration import (
    AUTO_AGENT_DISPLAY_NAME,
    build_pipeline_prompt,
    execution_steps_for_track,
    get_pipeline_step_agents,
    infer_auto_fast_route,
    infer_task_track,
    is_simple_direct_task,
    looks_like_rag_presentation_request,
    should_use_full_main_agent,
)


def test_pipeline_prompt_and_agent_flow_are_built_from_shared_steps():
    steps = [
        "📚 RAG Search (RAG SubAgent)",
        "📊 Create Presentation (Presenter)",
    ]

    prompt = build_pipeline_prompt(steps)

    assert "Delegate to the RAG specialist" in prompt
    assert "Delegate to the presenter" in prompt
    assert get_pipeline_step_agents(steps) == [AgentType.RAG_SUB.value, AgentType.PRESENTER.value]


def test_rag_presentation_request_detection_requires_both_signals():
    assert looks_like_rag_presentation_request("Search the RAG db and create a pptx")
    assert not looks_like_rag_presentation_request("Create a pptx about quarterly planning")
    assert not looks_like_rag_presentation_request("Create a pptx from the project documents")
    assert looks_like_rag_presentation_request("Create a pptx with citations from my uploaded files")


def test_infer_task_track_uses_registry_mapping_for_selected_agent():
    agent_display_options = get_registry().get_all_agents()
    selected_agent = next(name for name, agent_type in agent_display_options.items() if agent_type == AgentType.CODER.value)

    assert infer_task_track("anything", selected_agent, agent_display_options) == AgentType.CODER.value
    assert infer_task_track("find the latest benchmark", AUTO_AGENT_DISPLAY_NAME, agent_display_options) == AgentType.WEBSEARCH.value
    assert infer_task_track("summarize this document", AUTO_AGENT_DISPLAY_NAME, agent_display_options) == AgentType.WRITER.value
    assert infer_task_track("answer from my uploaded files with citations", AUTO_AGENT_DISPLAY_NAME, agent_display_options) == AgentType.RAG_SUB.value


def test_simple_direct_task_blocks_multi_step_and_non_web_latest_queries():
    assert is_simple_direct_task("find the latest ollama release notes", AgentType.WEBSEARCH.value)
    assert not is_simple_direct_task("research and write a summary of the latest ollama release notes", AgentType.WEBSEARCH.value)
    assert not is_simple_direct_task("latest regression results", AgentType.DATA_SCIENTIST.value)


def test_infer_auto_fast_route_prefers_simple_specialists_and_respects_rag_mode():
    agent_display_options = get_registry().get_all_agents()

    agent_type, auto_fast_route = infer_auto_fast_route(
        "search the web for current ollama release notes",
        selected_agent=AUTO_AGENT_DISPLAY_NAME,
        agent_display_options=agent_display_options,
        rag_knowledge_only=False,
        force_write_todos=False,
    )
    assert agent_type == AgentType.WEBSEARCH.value
    assert auto_fast_route == AgentType.WEBSEARCH.value

    rag_agent_type, rag_auto_fast_route = infer_auto_fast_route(
        "anything",
        selected_agent=AUTO_AGENT_DISPLAY_NAME,
        agent_display_options=agent_display_options,
        rag_knowledge_only=True,
        force_write_todos=False,
    )
    assert rag_agent_type == AgentType.RAG_SUB.value
    assert rag_auto_fast_route is None


def test_execution_steps_reflect_track_and_mode():
    auto_steps = execution_steps_for_track(AgentType.RAG_SUB.value, is_auto_mode=True)
    direct_steps = execution_steps_for_track(AgentType.PRESENTER.value, is_auto_mode=False)

    assert auto_steps[0] == "⏳ Analyzing Request..."
    assert "⏳ Retrieving and reranking chunks..." in auto_steps
    assert direct_steps[0] == "⏳ Validating Task Inputs..."
    assert "⏳ Generating presentation file..." in direct_steps


def test_should_use_full_main_agent_prefers_heavy_path_for_complex_requests():
    assert should_use_full_main_agent(
        "Research the issue and then write a step by step implementation plan",
        track="general",
    )
    assert should_use_full_main_agent(
        "Create a presentation from the RAG database",
        track=AgentType.PRESENTER.value,
    )
    assert should_use_full_main_agent(
        "quick answer",
        track="general",
        force_write_todos=True,
    )


def test_should_use_full_main_agent_allows_lean_path_for_routine_chat():
    assert not should_use_full_main_agent(
        "Explain what this repository does",
        track="general",
    )
    assert not should_use_full_main_agent(
        "Summarize this design in plain English",
        track=AgentType.WRITER.value,
    )
