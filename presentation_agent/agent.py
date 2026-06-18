"""Standalone Presenter subagent with dedicated model support."""

import os
from functools import lru_cache
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from agent_runtime import build_agent_config
from model_config import create_chat_model, get_agent_model

from .prompts import get_system_prompt
from .tools import generate_presentation

PROJECT_DIR = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def create_presenter_agent():
    """Create a specialized presenter subagent."""
    presenter_model_name = get_agent_model("presenter")

    try:
        model = create_chat_model(model_name=presenter_model_name, temperature=0)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize presenter model '{presenter_model_name}': {e}"
        ) from e

    agent = create_deep_agent(
        model=model,
        memory=[],
        skills=[],
        subagents=[],
        backend=FilesystemBackend(root_dir=PROJECT_DIR, virtual_mode=True),
        tools=[generate_presentation],
        system_prompt=get_system_prompt(),
    )
    return agent


async def invoke_presenter(
    messages: list,
    thread_id: str = "presenter",
    user_id: str | None = None,
) -> dict:
    """Invoke the presenter subagent."""
    agent = create_presenter_agent()
    return await agent.ainvoke(
        {"messages": messages},
        config=build_agent_config(
            thread_id=thread_id,
            user_id=user_id,
            default_user_id="presenter-user",
        ),
    )
