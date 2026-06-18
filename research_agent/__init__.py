"""Compatibility shim for the legacy research_agent package."""

from websearch_agent.prompts import WEBSEARCH_INSTRUCTIONS as RESEARCHER_INSTRUCTIONS
from websearch_agent.tools import tavily_search, think_tool

__all__ = ["tavily_search", "think_tool", "RESEARCHER_INSTRUCTIONS"]
