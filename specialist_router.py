from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from agent_runtime import build_agent_config
from model_config import create_chat_model, get_agent_model
from specialist_catalog import get_specialist_tool_map, get_subagent_config
from translation_tool import translate_text

PROJECT_DIR = Path(__file__).parent


def _build_direct_specialist_prompt(subagent_type: str, base_prompt: str) -> str:
    """Add fast-path execution guidance for single-specialist direct routes."""
    return (
        f"{base_prompt.strip()}\n\n"
        "## Streamlit Fast Path\n"
        f"You were selected as the direct `{subagent_type}` executor for a single request.\n"
        "Work directly instead of delegating through another coordinator.\n"
        "Keep the execution shallow: skip unnecessary planning and avoid multi-pass orchestration "
        "unless correctness clearly requires it.\n"
        "Return a concise final answer grounded in the actual work performed."
    )


@lru_cache(maxsize=8)
def create_specialist_router_agent(subagent_type: str):
    """Create and cache a direct single-specialist agent for Streamlit fast routes."""
    model_name = get_agent_model(subagent_type)
    try:
        model = create_chat_model(model_name=model_name, temperature=0)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize specialist router model '{model_name}': {e}"
        ) from e

    config = get_subagent_config(subagent_type)
    tool_map = get_specialist_tool_map()
    return create_deep_agent(
        model=model,
        memory=[],
        skills=[],
        subagents=[],
        backend=FilesystemBackend(root_dir=PROJECT_DIR, virtual_mode=True),
        tools=[*tool_map.get(subagent_type, []), translate_text],
        system_prompt=_build_direct_specialist_prompt(
            subagent_type,
            config["system_prompt"],
        ),
    )


async def invoke_specialist_router(
    subagent_type: str,
    messages: list,
    thread_id: str,
    user_id: str | None = None,
) -> dict:
    """Invoke the cached direct specialist agent."""
    agent = create_specialist_router_agent(subagent_type)
    return await agent.ainvoke(
        {"messages": messages},
        config=build_agent_config(
            thread_id=f"{thread_id}-{subagent_type}",
            user_id=user_id,
            recursion_limit=24,
            default_user_id=f"{subagent_type}-user",
        ),
    )
