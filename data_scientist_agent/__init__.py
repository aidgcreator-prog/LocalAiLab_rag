"""Data Scientist Agent Module.

This module provides data analysis tools for Python code execution using conda environment.
Supports data manipulation, visualization, and statistical analysis.
"""

from data_scientist_agent.prompts import DATA_SCIENTIST_INSTRUCTIONS, get_system_prompt
from data_scientist_agent.tools import (
    execute_python_code,
    install_package,
    think_tool,
    render_quarto_report,
)
from data_scientist_agent.agent import create_data_scientist_agent, invoke_data_scientist

__all__ = [
    "execute_python_code",
    "install_package",
    "think_tool",
    "render_quarto_report",
    "DATA_SCIENTIST_INSTRUCTIONS",
    "get_system_prompt",
    "create_data_scientist_agent",
    "invoke_data_scientist",
]
