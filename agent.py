"""Combined Agent Orchestrator

A multi-specialist agent that delegates complex tasks to specialized subagents.
Demonstrates multi-agent orchestration patterns with DeepAgents.

Includes production-grade features:
- LangSmith observability and tracing
- Auto-summarization for long conversations
- Prompt caching for cost reduction
- Filesystem permissions for security
- LangGraph memory store for persistence
- Task planning with write_todos tool

Exported as 'agent' for LangGraph Studio deployment.

Usage:
    uv run python agent.py "Build a product roadmap for an AI coding assistant"
    uv run langgraph dev  # For LangGraph Studio
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not required, environment variables can be set manually

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from conversation_memory import record_learning, session_search
from translation_tool import translate_text
from model_config import create_chat_model, get_agent_model
from specialist_catalog import build_subagent_specs, get_specialist_tool_map
from agent_input_middleware import InputContentSanitizerMiddleware
from agent_runtime import (
    LANGGRAPH_CHECKPOINTER,
    LANGGRAPH_PERSISTENCE_STATUS,
    LANGGRAPH_STORE,
    build_agent_config,
)

PROJECT_DIR = Path(__file__).parent
PROMPT_CACHE_STATUS = "[INFO] Prompt caching status not initialized"
FILESYSTEM_PERMISSION_STATUS = "[INFO] Filesystem permission status not initialized"
DEEPAGENTS_VERSION = "unknown"


def _using_langgraph_api_runtime() -> bool:
    """Detect when the graph is being loaded inside LangGraph API/dev server.

    LangGraph API manages persistence itself and rejects user-supplied
    checkpointers/stores on exported graphs.
    """
    if any(name.startswith("langgraph_api") for name in sys.modules):
        return True
    if any(name.startswith("langgraph_runtime") for name in sys.modules):
        return True
    return any(key.startswith("LANGGRAPH") for key in os.environ)


# ===== Production Features Setup =====

def setup_langsmith_tracing() -> None:
    """Enable LangSmith tracing for observability.

    LangSmith allows you to:
    - Trace every agent decision and tool call
    - Debug agent behavior in detail
    - Evaluate agent outputs
    - Monitor production performance

    Also enables OpenTelemetry export to AI Toolkit's local trace viewer
    at http://localhost:4318 for local debugging without a LangSmith account.
    """
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
    os.environ["LANGSMITH_TRACING"] = "true" if tracing_enabled else "false"
    if not tracing_enabled:
        os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
        print("[INFO] LangSmith tracing disabled")
        return

    # Enable OpenTelemetry export only if a local collector is reachable
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    otel_available = False
    try:
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(otel_endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 4318
        with socket.create_connection((host, port), timeout=1):
            otel_available = True
    except (OSError, ValueError):
        pass

    if otel_available:
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
        print(f"[OK] OpenTelemetry tracing enabled → {otel_endpoint} (AI Toolkit)")
    else:
        os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
        print(f"[INFO] OpenTelemetry collector not reachable at {otel_endpoint} - OTLP export disabled")

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    project = os.getenv("LANGSMITH_PROJECT", "lv-combined-agents")

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project
        print(f"[OK] LangSmith cloud tracing enabled - Project: {project}")
    else:
        print("[INFO] LangSmith API key not set - using local OTLP tracing only")


def setup_prompt_caching() -> None:
    """Enable prompt caching when configured and supported by installed LangChain modules."""
    global PROMPT_CACHE_STATUS

    cache_enabled = os.getenv("LANGCHAIN_ENABLE_CACHE", "").lower() == "true"
    if not cache_enabled:
        PROMPT_CACHE_STATUS = "[WARN] Prompt caching disabled (set LANGCHAIN_ENABLE_CACHE=true)"
        print(PROMPT_CACHE_STATUS)
        return

    try:
        from langchain_core.caches import InMemoryCache
        from langchain_core.globals import set_llm_cache
    except Exception as e:
        PROMPT_CACHE_STATUS = (
            f"[WARN] Prompt caching requested but cache backend is unavailable: {e}"
        )
        print(PROMPT_CACHE_STATUS)
        return

    try:
        set_llm_cache(InMemoryCache())
        PROMPT_CACHE_STATUS = "[OK] Prompt caching enabled (LangChain InMemoryCache)"
        print(PROMPT_CACHE_STATUS)
    except Exception as e:
        PROMPT_CACHE_STATUS = f"[WARN] Failed to initialize prompt cache: {e}"
        print(PROMPT_CACHE_STATUS)


def _parse_major_minor(version_str: str) -> tuple[int, int]:
    """Parse major/minor integers from a version string."""
    parts = version_str.split(".")
    nums: list[int] = []
    for part in parts[:2]:
        digits = "".join(ch for ch in part if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 2:
        nums.append(0)
    return nums[0], nums[1]


def setup_filesystem_permission_status() -> None:
    """Set and print filesystem permission capability status based on installed DeepAgents version."""
    global FILESYSTEM_PERMISSION_STATUS, DEEPAGENTS_VERSION

    try:
        DEEPAGENTS_VERSION = pkg_version("deepagents")
    except PackageNotFoundError:
        DEEPAGENTS_VERSION = "unknown"

    major, minor = _parse_major_minor(DEEPAGENTS_VERSION)
    if (major, minor) >= (0, 6):
        FILESYSTEM_PERMISSION_STATUS = (
            f"[INFO] DeepAgents {DEEPAGENTS_VERSION} detected; fine-grained filesystem permissions may be available with explicit integration"
        )
    else:
        FILESYSTEM_PERMISSION_STATUS = (
            f"[INFO] DeepAgents {DEEPAGENTS_VERSION}: using virtual_mode sandbox (fine-grained per-subagent permissions not integrated in this project)"
        )

    print(FILESYSTEM_PERMISSION_STATUS)


# Note: Filesystem permissions framework (FilesystemPermission) is available in DeepAgents
# but not yet fully integrated with the create_deep_agent() factory in v0.5.3.
# Fine-grained per-subagent permission controls are planned for a future release.
# For now, use FilesystemBackend with virtual_mode=True for safe sandboxing.




# ===== Initialize Production Features =====

# Setup LangSmith tracing
setup_langsmith_tracing()

# Setup prompt caching
setup_prompt_caching()

# Detect filesystem permission capability by installed DeepAgents version
setup_filesystem_permission_status()

# Global cache for initialized components
_AGENT_RUNTIME_CACHE: dict[str, Any] = {}

def _get_runtime_component(key: str, init_func: callable) -> Any:
    """Helper to implement lazy loading with a cache."""
    if key not in _AGENT_RUNTIME_CACHE:
        _AGENT_RUNTIME_CACHE[key] = init_func()
    return _AGENT_RUNTIME_CACHE[key]

def _init_model() -> Any:
    model_name = get_agent_model("main")
    return create_chat_model(model_name=model_name, temperature=0)

def _init_subagents() -> tuple[dict, list]:
    tool_map = get_specialist_tool_map()
    # Pass pre-built BaseChatModel objects so deepagents.resolve_model bypasses
    # init_chat_model() entirely — needed for custom providers (llama_cpp, huggingface)
    # that LangChain's init_chat_model doesn't recognise by prefix.
    def _build_subagent_model(agent_name: str) -> Any:
        model_string = get_agent_model(agent_name)
        if not model_string:
            model_string = get_agent_model("main")
        return create_chat_model(model_name=model_string, temperature=0)

    specs = build_subagent_specs(model_resolver=_build_subagent_model)
    return tool_map, specs

def get_agent_runtime_components():
    """
    Returns the initialized components (model, subagent_specs, etc.) 
    using lazy loading to ensure fast server startup.
    """
    model = _get_runtime_component("model", _init_model)
    subagent_tool_map, subagent_specs = _get_runtime_component("subagents", _init_subagents)
    return model, subagent_specs, subagent_tool_map

# Module-level flag for persistence mode
_use_platform_persistence = _using_langgraph_api_runtime()


def _get_base_agent_kwargs() -> dict[str, Any]:
    model, subagent_specs, _ = get_agent_runtime_components()
    
    kwargs: dict[str, Any] = {
        "model": model,
        "subagents": subagent_specs,
        "backend": FilesystemBackend(root_dir=PROJECT_DIR, virtual_mode=True),
        "tools": [translate_text],
        "middleware": [InputContentSanitizerMiddleware()],
    }
    if not _use_platform_persistence:
        kwargs["checkpointer"] = LANGGRAPH_CHECKPOINTER
        kwargs["store"] = LANGGRAPH_STORE
    return kwargs

# Load model and create agent at module level for LangGraph deployment
# Note: We use the dynamic kwargs generator to avoid heavy work on import
_BASE_AGENT_KWARGS: dict[str, Any] = _get_base_agent_kwargs()
SUBAGENT_TOOL_MAP = get_specialist_tool_map()

_HEAVY_MAIN_SYSTEM_PROMPT = (
    "You are the main coordinator. Coordinate, delegate, and synthesize. "
    "Do not do specialist work yourself when a matching subagent exists.\n\n"
    "Use `task` for specialist work:\n"
    "- planner: plans, milestones, dependencies\n"
    "- websearch: live/current information\n"
    "- writer: user-facing writing, docs, summaries, blog/social content\n"
    "- coder: code changes, debugging, refactors\n"
    "- reviewer: QA, regressions, critique\n"
    "- presenter: presentations and PPTX output\n"
    "- data_scientist: Python analysis, statistics, plots\n"
    "- ragsub: document-grounded answers, retrieval, citations\n\n"
    "Use `translate_text` directly for translation. If Khmer characters appear, "
    "translate to English first, then continue.\n\n"
    "Use `write_todos` only for clearly multi-step work. Skip it for simple requests.\n"
    "Use `session_search` when prior session context may matter. Use `record_learning` "
    "only for durable reusable lessons.\n\n"
    "Routing rules:\n"
    "- current facts or web lookup -> websearch\n"
    "- document-grounded or citation-grounded requests -> ragsub\n"
    "- code work -> coder\n"
    "- writing deliverables -> writer\n"
    "- planning/roadmaps -> planner\n"
    "- review/testing/critique -> reviewer\n"
    "- data analysis or Python execution -> data_scientist\n"
    "- presentations or PPTX generation -> presenter\n\n"
    "Presentation rule: never create presentation files yourself. Always use presenter. "
    "If the deck must be grounded in documents, do `ragsub` first, then `presenter`.\n\n"
    "Tool rules:\n"
    "- make real tool calls; never describe fake tool usage\n"
    "- do not emit tool calls as plain text\n"
    "- keep orchestration shallow when one specialist is enough\n"
    "- if a specialist fails, retry intelligently or explain the issue\n"
    "- final answer should summarize actual results in plain text\n"
)

_TRIVIAL_MAIN_SYSTEM_PROMPT = (
    "You are a lightweight chat agent for trivial requests. "
    "Answer directly, briefly, and usefully. "
    "Do not call task, write_todos, or session_search. "
    "If the message contains Khmer text, call translate_text first.\n"
)

_STANDARD_MAIN_SYSTEM_PROMPT = (
    "You are a standard chat agent for typical requests. "
    "Answer directly when possible. Delegate only when it materially improves the result.\n\n"
    "Delegate with `task` for: live web search, code work, writing deliverables, "
    "data analysis, document-grounded answers, planning, review, and presentations. "
    "Otherwise answer directly. If Khmer text appears, call translate_text first.\n"
)

_LEAN_MAIN_SYSTEM_PROMPT = (
    "You are the Streamlit chat agent for routine requests. "
    "Answer directly for simple single-step requests. Delegate only when needed.\n\n"
    "Use `task` for: websearch, ragsub, coder, writer, planner, reviewer, presenter, "
    "and data_scientist. Do not default to delegation. "
    "Use `write_todos` only for clearly multi-step work. "
    "Use real tool calls only. If Khmer text appears, call translate_text first.\n"
)


def _build_trivial_agent():
    """Create a minimal agent for trivial requests (no memory, no skills)."""
    return create_deep_agent(
        **{
            **_BASE_AGENT_KWARGS,
            "memory": [],
            "skills": [],
            "tools": [translate_text],
        },
        system_prompt=_TRIVIAL_MAIN_SYSTEM_PROMPT,
    )


def _build_standard_agent():
    """Create a standard agent for typical requests (minimal memory, selective skills)."""
    return create_deep_agent(
        **{
            **_BASE_AGENT_KWARGS,
            "memory": ["./AGENTS.md"],  # Only the main instructions
            "skills": [],  # Skills loaded per-request by caller
            "tools": [session_search, translate_text],
        },
        system_prompt=_STANDARD_MAIN_SYSTEM_PROMPT,
    )


def _build_complex_agent():
    """Create the full coordinator used for complex workflows and LangGraph export."""
    return create_deep_agent(
        **{
            **_BASE_AGENT_KWARGS,
            "memory": [
                "./AGENTS.md",
                "./memories/user/preferences.md",
                "./memories/user/context.md",
                "./memories/repo/agent_learnings.md",
            ],
            "skills": ["./skills/"],
            "tools": [session_search, record_learning, translate_text],
        },
        system_prompt=_HEAVY_MAIN_SYSTEM_PROMPT,
    )


def _build_lean_main_agent():
    """Create a smaller Streamlit-oriented coordinator for routine chat requests."""
    return create_deep_agent(
        **{
            **_BASE_AGENT_KWARGS,
            "memory": [],
            "skills": [],
            "tools": [translate_text],
        },
        system_prompt=_LEAN_MAIN_SYSTEM_PROMPT,
    )

# Lazy initialization for agent variants (to avoid slow import time)
_AGENT_VARIANTS_CACHE: dict[str, Any] = {}

def _get_agent_variant(variant_name: str) -> Any:
    """Lazily build and cache agent variants to avoid slow import time."""
    if variant_name not in _AGENT_VARIANTS_CACHE:
        builder_map = {
            "trivial": _build_trivial_agent,
            "standard": _build_standard_agent,
            "complex": _build_complex_agent,
            "lean": _build_lean_main_agent,
        }
        builder = builder_map.get(variant_name)
        if not builder:
            raise ValueError(f"Unknown agent variant: {variant_name}")
        _AGENT_VARIANTS_CACHE[variant_name] = builder()
    return _AGENT_VARIANTS_CACHE[variant_name]


# Export lazy getter functions for streamlit_app.py
def get_trivial_agent() -> Any:
    """Get or build trivial agent variant."""
    return _get_agent_variant("trivial")

def get_standard_agent() -> Any:
    """Get or build standard agent variant."""
    return _get_agent_variant("standard")

def get_complex_agent() -> Any:
    """Get or build complex agent variant."""
    return _get_agent_variant("complex")

def get_lean_agent() -> Any:
    """Get or build lean agent variant."""
    return _get_agent_variant("lean")


# Build complex agent eagerly for LangGraph export (this is the one deployed)
# This will take a few seconds but is necessary for production deployment
complex_agent = _build_complex_agent()

# Legacy names for backward compatibility
agent = complex_agent
streamlit_agent = get_lean_agent()

# Also cache the already-built variants to avoid rebuilding them
_AGENT_VARIANTS_CACHE["complex"] = complex_agent

# Module initialization messaging (happens only once at import)
print("[OK] Agent module loaded with lazy initialization enabled")
print("  - LangSmith tracing for observability")
print("  - Auto-summarization via LangGraph context management")
if _use_platform_persistence:
    print("  - [INFO] Using LangGraph API platform-managed persistence")
else:
    print(f"  - {LANGGRAPH_PERSISTENCE_STATUS}")
print("  - Sandboxed filesystem access via virtual_mode")
print(f"  - {PROMPT_CACHE_STATUS}")
print(f"  - {FILESYSTEM_PERMISSION_STATUS}")
print("  - Main agent variants: trivial, standard, complex, lean")
print("  - Lazy initialization: trivial, standard, and lean agents build on first request")
print("  - Complex agent is built eagerly for LangGraph export/runtime")


def select_agent_by_complexity(
    user_input: str,
    track: str = "",
    force_complex: bool = False,
    has_attachments: bool = False,
    rag_enabled: bool = False,
) -> tuple[Any, str]:
    """
    Select the appropriate agent based on request complexity.
    
    Args:
        user_input: User's request text
        track: Task track inferred from input (e.g., 'websearch', 'planner', etc.)
        force_complex: Ignored — agent chooses complexity autonomously
        has_attachments: Whether request has file attachments
        rag_enabled: Whether RAG context is enabled for this request
        
    Returns:
        Tuple of (agent, tier_name) where tier_name is 'trivial', 'standard', or 'complex'
    """
    if force_complex:
        return get_complex_agent(), "complex"

    # RAG-backed and attachment-heavy requests benefit from the full coordinator.
    if rag_enabled:
        return get_complex_agent(), "complex"

    text = (user_input or "").strip()
    lowered = text.lower()

    if has_attachments and track in {
        "",
        "general",
        "presenter",
        "ragsub",
        "data_scientist",
        "writer",
    }:
        return get_complex_agent(), "complex"

    # Trivial cases: very short, simple questions or social pleasantries
    if len(text) < 30 and not track:
        # Check for simple patterns
        simple_patterns = [
            r"^(hi|hello|hey|thanks|thank you|ok|got it|cool|nice)[!?]*$",
            r"^what is .{1,20}\?$",
            r"^how do i .{1,20}\?$",
            r"^help$",
        ]
        import re
        for pattern in simple_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return get_trivial_agent(), "trivial"

    complex_markers = [
        "\n",
        " and then ",
        " after that ",
        " step by step ",
        " compare ",
        " evaluate ",
        " critique ",
        " investigate ",
        " with citations ",
        " synthesize ",
        " grounded ",
        " from my uploaded files ",
        " from the documents ",
    ]
    if (
        len(text.split()) > 120
        or track in {"data_scientist", "reviewer"}
        or any(marker in f" {lowered} " for marker in complex_markers)
    ):
        return get_complex_agent(), "complex"

    # Standard for routine single-specialist tasks and general chat.
    return get_standard_agent(), "standard"


def extract_last_ai_text(messages: list[BaseMessage]) -> str:
    """Extract the final AI response text from message history.

    Walks messages in reverse. Skips AIMessages that have empty content but
    contain tool_calls (intermediate delegation steps). Falls back to the
    last ToolMessage content if no AIMessage with text is found.

    Args:
        messages: List of messages from agent invocation

    Returns:
        The text content of the last meaningful message, or empty string
    """
    last_tool_content = ""
    for message in reversed(messages):
        # Collect the most recent ToolMessage as fallback
        if isinstance(message, ToolMessage) and not last_tool_content:
            content = message.content
            if isinstance(content, str) and content.strip():
                last_tool_content = content.strip()

        if isinstance(message, AIMessage):
            content = message.content

            if isinstance(content, str) and content.strip():
                return content

            # Handle content blocks (e.g., structured outputs)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                result = "\n".join(p for p in text_parts if p)
                if result:
                    return result

            # If content is empty but has tool_calls, keep looking
            if hasattr(message, "tool_calls") and message.tool_calls:
                continue
            # Empty content, no tool_calls – stop
            return ""

    # No AIMessage with text found; fall back to last tool output
    return last_tool_content


async def main() -> None:
    """Main entry point for CLI usage (when running directly with Python).

    Reads task from command line arguments and invokes the agent.
    """
    # Get task from command line or use default
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        task = (
            "Create a launch plan for a small SaaS app, including technical architecture, "
            "MVP scope, a two-week build plan, and risk review."
        )

    print(f"Task: {task}\n")
    print("=" * 80)

    try:
        print("Invoking agent...\n")
        result = await agent.ainvoke(
            {"messages": [("user", task)]},
            config=build_agent_config(
                thread_id="lv-combined-agent-thread",
                user_id="cli-user",
            ),
        )

        # Extract and display results
        final_text = extract_last_ai_text(result.get("messages", []))

        print("=" * 80)
        print("\nResult:\n")
        print(final_text or "No response content returned.")

    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
