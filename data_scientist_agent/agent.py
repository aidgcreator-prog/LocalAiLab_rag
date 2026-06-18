"""Standalone Data Scientist Agent with Dedicated Model.

This module creates a specialized agent for data analysis that can use a dedicated model
different from the main orchestrator's model.

Usage:
    from data_scientist_agent.agent import create_data_scientist_agent, invoke_data_scientist
"""

import os
from functools import lru_cache
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from pathlib import Path
from agent_runtime import build_agent_config
from model_config import create_chat_model, get_agent_model

from .tools import execute_python_code, install_package, think_tool, render_quarto_report
from .prompts import get_system_prompt

PROJECT_DIR = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def create_data_scientist_agent():
    """Create a specialized data scientist agent with optional dedicated model.

    Returns:
        Agent instance configured for data analysis tasks
    """
    ds_model_name = get_agent_model("data_scientist")

    try:
        model = create_chat_model(model_name=ds_model_name, temperature=0)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize data scientist model '{ds_model_name}': {e}\n"
            "Check that the model is available and API keys are set."
        ) from e

    # Create the data scientist agent
    agent = create_deep_agent(
        model=model,
        memory=[],  # Don't use shared memory
        skills=[],  # No skill files for data scientist
        subagents=[],  # No nested subagents
        backend=FilesystemBackend(root_dir=PROJECT_DIR, virtual_mode=True),
        tools=[execute_python_code, install_package, think_tool, render_quarto_report],
        system_prompt=get_system_prompt(),
    )

    return agent


async def invoke_data_scientist(
    messages: list,
    thread_id: str = "data-scientist",
    user_id: str | None = None,
) -> dict:
    """Invoke the data scientist agent with messages.

    Args:
        messages: List of (role, content) tuples
        thread_id: Thread ID for conversation context
        user_id: Stable user identity for persistence keying

    Returns:
        Agent result dictionary with response messages
    """
    agent = create_data_scientist_agent()

    result = await agent.ainvoke(
        {"messages": messages},
        config=build_agent_config(
            thread_id=thread_id,
            user_id=user_id,
            recursion_limit=90,
            default_user_id="data-scientist-user",
        ),
    )

    return result
