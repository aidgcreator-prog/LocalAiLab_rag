"""Presentation Agent Module.

This module provides tools for creating professional presentations in PowerPoint format using Quarto.
"""

from presentation_agent.tools import generate_presentation, generate_presentation_quarto
from presentation_agent.prompts import PRESENTER_INSTRUCTIONS, get_system_prompt
from presentation_agent.agent import create_presenter_agent, invoke_presenter

__all__ = [
    "generate_presentation",
    "generate_presentation_quarto",
    "PRESENTER_INSTRUCTIONS",
    "get_system_prompt",
    "create_presenter_agent",
    "invoke_presenter",
]
