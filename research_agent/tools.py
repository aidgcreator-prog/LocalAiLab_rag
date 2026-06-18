"""Compatibility shim for the legacy research_agent tools module."""

from websearch_agent.tools import (  # noqa: F401
    TAVILY_AVAILABLE,
    fetch_webpage_content,
    tavily_client,
    tavily_search,
    think_tool,
)
