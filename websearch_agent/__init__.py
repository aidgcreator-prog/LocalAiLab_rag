"""Web Search Agent Module.

This module provides web search tools for the combined agents framework.
"""

from websearch_agent.prompts import WEBSEARCH_INSTRUCTIONS
from websearch_agent.tools import tavily_search, think_tool

__all__ = [
    "tavily_search",
    "think_tool",
    "WEBSEARCH_INSTRUCTIONS",
]
