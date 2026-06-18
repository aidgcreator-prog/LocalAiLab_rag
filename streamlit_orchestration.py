from __future__ import annotations

from agent_registry import AgentType

AUTO_AGENT_DISPLAY_NAME = "Auto (Main Agent Decides)"

PIPELINE_STEPS = {
    "🌐 Web Search (Websearch)": {
        "agent": AgentType.WEBSEARCH.value,
        "instruction": "Delegate to the websearch specialist to search the web and gather relevant information.",
    },
    "📚 RAG Search (RAG SubAgent)": {
        "agent": AgentType.RAG_SUB.value,
        "instruction": "Delegate to the RAG specialist to retrieve and synthesize indexed document context.",
    },
    "📊 Data Analysis (Data Scientist)": {
        "agent": AgentType.DATA_SCIENTIST.value,
        "instruction": "Delegate to the data scientist to analyze the data, generate statistics, and create visualizations.",
    },
    "📝 Write Content (Writer)": {
        "agent": AgentType.WRITER.value,
        "instruction": "Delegate to the writer to draft polished text, documentation, or a report from the findings so far.",
    },
    "💻 Code Implementation (Coder)": {
        "agent": AgentType.CODER.value,
        "instruction": "Delegate to the coder to implement code changes or build the requested software.",
    },
    "📋 Planning (Planner)": {
        "agent": AgentType.PLANNER.value,
        "instruction": "Delegate to the planner to break down the goal into a concrete step-by-step plan.",
    },
    "🔍 Review (Reviewer)": {
        "agent": AgentType.REVIEWER.value,
        "instruction": "Delegate to the reviewer to critique the outputs for quality, bugs, and completeness.",
    },
    "📊 Create Presentation (Presenter)": {
        "agent": AgentType.PRESENTER.value,
        "instruction": "Delegate to the presenter to create a PowerPoint presentation from the accumulated findings.",
    },
}
PIPELINE_STEP_NAMES = list(PIPELINE_STEPS.keys())


def build_pipeline_prompt(steps: list[str]) -> str:
    """Build a numbered workflow instruction string from pipeline step names."""
    if not steps:
        return ""

    lines: list[str] = []
    for index, step_name in enumerate(steps, 1):
        info = PIPELINE_STEPS.get(step_name)
        if not info:
            continue
        lines.append(f"{index}. {info['instruction']}")
        if index < len(steps):
            lines.append("   → Pass the output to the next step.")
    return "\n".join(lines)


def get_pipeline_step_agents(steps: list[str]) -> list[str]:
    """Return the ordered agent ids represented by pipeline step names."""
    return [PIPELINE_STEPS[step]["agent"] for step in steps if step in PIPELINE_STEPS]


def looks_like_presentation_request(user_input: str) -> bool:
    """Heuristic detection for presentation-generation requests."""
    text = (user_input or "").lower()
    # Check for direct presentation keywords
    direct_keywords = ["presentation", "slide", "slides", "powerpoint", "ppt", "pptx"]
    if any(keyword in text for keyword in direct_keywords):
        return True
    # Check for "convert to pptx" pattern
    if "convert" in text and ("pptx" in text or "powerpoint" in text or "presentation" in text):
        return True
    return False


def looks_like_rag_presentation_request(user_input: str) -> bool:
    """Detect presentation requests that should be grounded through RAG first."""
    text = (user_input or "").lower()
    if not looks_like_presentation_request(text):
        return False

    rag_markers = [
        "rag",
        "rag db",
        "vector db",
        "vector database",
        "database",
        "indexed",
        "index",
        "citation",
        "citations",
        "grounded",
        "knowledge base",
        "knowledgebase",
        "my files",
        "uploaded file",
        "document set",
        "uploaded files",
        "from the docs",
        "from documents",
    ]
    return any(marker in text for marker in rag_markers)


def infer_task_track(
    user_input: str,
    selected_agent: str,
    agent_display_options: dict[str, str],
) -> str:
    """Infer likely task track for status messaging and fast routing."""
    if selected_agent != AUTO_AGENT_DISPLAY_NAME:
        return agent_display_options.get(selected_agent, AgentType.AUTO.value)

    text = (user_input or "").lower()
    if looks_like_presentation_request(text):
        return AgentType.PRESENTER.value
    if any(
        k in text
        for k in [
            "rag",
            "rag db",
            "vector db",
            "vector database",
            "knowledge base",
            "knowledgebase",
            "retrieve",
            "retrieval",
            "with citations",
            "citation-grounded",
            "uploaded files",
            "uploaded file",
            "my files",
            "document set",
            "indexed documents",
            "index these",
        ]
    ):
        return AgentType.RAG_SUB.value
    if any(k in text for k in ["analyze", "analysis", "regression", "dataset", "csv", "eda", "statistics"]):
        return AgentType.DATA_SCIENTIST.value
    if any(k in text for k in ["research", "latest", "search", "find information"]):
        return AgentType.WEBSEARCH.value
    if any(k in text for k in ["code", "debug", "fix", "refactor", "implement"]):
        return AgentType.CODER.value
    if any(k in text for k in ["write", "draft", "document", "summary"]):
        return AgentType.WRITER.value
    if any(k in text for k in ["plan", "roadmap", "milestone"]):
        return AgentType.PLANNER.value
    if any(k in text for k in ["review", "qa", "test"]):
        return AgentType.REVIEWER.value
    return "general"


def is_simple_direct_task(user_input: str, track: str) -> bool:
    """Heuristic to fast-route straightforward single-specialist requests."""
    if track not in {
        AgentType.WEBSEARCH.value,
        AgentType.RAG_SUB.value,
        AgentType.CODER.value,
        AgentType.WRITER.value,
        AgentType.PLANNER.value,
        AgentType.REVIEWER.value,
        AgentType.PRESENTER.value,
    }:
        return False

    text = (user_input or "").strip().lower()
    if not text:
        return False

    multi_step_markers = [
        "\n",
        " and then ",
        " then ",
        " after that ",
        "compare",
        "both",
        "also ",
        "as well as",
        " or ",
        "step by step",
        "roadmap and",
        "research and",
        "write and",
        "fix and",
        "summarize and",
        "analyze and",
    ]
    if any(marker in text for marker in multi_step_markers):
        return False

    if any(marker in text for marker in ["latest", "most recent", "today", "current"]) and track != AgentType.WEBSEARCH.value:
        return False

    return len(text.split()) <= 80


def infer_auto_fast_route(
    user_input: str,
    *,
    selected_agent: str,
    agent_display_options: dict[str, str],
    rag_knowledge_only: bool,
    force_write_todos: bool,
    force_agent_type: str | None = None,
    force_main_agent: bool = False,
) -> tuple[str, str | None]:
    """Resolve the best direct route for routine requests.

    Returns a tuple of ``(agent_type, auto_fast_route)``. ``auto_fast_route`` is set
    only when the request should bypass the coordinator and go straight to a single
    specialist.
    """
    agent_type = (
        AgentType.AUTO.value
        if force_main_agent
        else (force_agent_type or agent_display_options.get(selected_agent, AgentType.AUTO.value))
    )

    if rag_knowledge_only:
        return AgentType.RAG_SUB.value, None

    auto_fast_route = None
    if (
        not force_agent_type
        and not force_main_agent
        and selected_agent == AUTO_AGENT_DISPLAY_NAME
        and not force_write_todos
    ):
        auto_track = infer_task_track(user_input, selected_agent, agent_display_options)
        if is_simple_direct_task(user_input, auto_track):
            agent_type = auto_track
            auto_fast_route = auto_track

    return agent_type, auto_fast_route


def should_use_full_main_agent(
    user_input: str,
    *,
    track: str,
    force_write_todos: bool = False,
    pipeline_steps: list[str] | None = None,
    rag_enable_main_chat: bool = False,
) -> bool:
    """Return whether Streamlit should use the heavy coordinator instead of the lean one."""
    if force_write_todos or bool(pipeline_steps):
        return True

    text = (user_input or "").strip().lower()
    if not text:
        return False

    if looks_like_rag_presentation_request(text):
        return True

    if rag_enable_main_chat and any(marker in text for marker in ["compare", "synthesize", "with citations"]):
        return True

    # Allow simple presentation requests to use the fast-path presenter subagent
    # Only use main agent for complex presentation requests or data scientist tasks
    if track == AgentType.DATA_SCIENTIST.value:
        return True
    
    if track == AgentType.PRESENTER.value and not is_simple_direct_task(user_input, track):
        return True

    complex_markers = [
        "\n",
        " and then ",
        " after that ",
        " step by step ",
        " create a plan ",
        " make a plan ",
        " compare ",
        " evaluate ",
        " review ",
        " critique ",
        " refactor ",
        " implement ",
        " debug ",
        " investigate ",
    ]
    if any(marker in f" {text} " for marker in complex_markers):
        return True

    if len(text.split()) > 120:
        return True

    return False


def execution_steps_for_track(track: str, is_auto_mode: bool) -> list[str]:
    """Return execution steps for the status panel."""
    prefix = (
        ["⏳ Analyzing Request...", "⏳ Selecting Best Specialist..."]
        if is_auto_mode
        else ["⏳ Validating Task Inputs..."]
    )

    track_steps = {
        AgentType.DATA_SCIENTIST.value: [
            "⏳ Loading and inspecting dataset...",
            "⏳ Computing statistics and patterns...",
            "⏳ Building analysis summary...",
        ],
        AgentType.RAG_SUB.value: [
            "⏳ Ingesting/reusing indexed documents...",
            "⏳ Retrieving and reranking chunks...",
            "⏳ Building citation-grounded response...",
        ],
        AgentType.PRESENTER.value: [
            "⏳ Planning slide structure...",
            "⏳ Generating presentation file...",
            "⏳ Preparing downloadable artifact...",
        ],
        AgentType.WEBSEARCH.value: [
            "⏳ Collecting relevant references...",
            "⏳ Synthesizing findings...",
        ],
        AgentType.CODER.value: [
            "⏳ Inspecting implementation context...",
            "⏳ Applying code changes...",
            "⏳ Validating outputs...",
        ],
        AgentType.WRITER.value: [
            "⏳ Structuring response draft...",
            "⏳ Polishing final content...",
        ],
        AgentType.PLANNER.value: [
            "⏳ Decomposing goals and dependencies...",
            "⏳ Building actionable execution plan...",
        ],
        AgentType.REVIEWER.value: [
            "⏳ Scanning for risks and regressions...",
            "⏳ Summarizing findings by severity...",
        ],
        "general": [
            "⏳ Processing with selected specialist...",
            "⏳ Synthesizing results...",
        ],
    }
    return prefix + track_steps.get(track, track_steps["general"])
