"""Web Search Tools.

This module provides search and content processing utilities for the web search agent,
using Tavily for URL discovery and fetching full webpage content.
"""

import os

import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from typing_extensions import Annotated, Literal

try:
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    TAVILY_AVAILABLE = True
except Exception:
    tavily_client = None
    TAVILY_AVAILABLE = False


def _should_fetch_full_web_content() -> bool:
    """Return whether search should fetch each result page in full."""
    raw = os.getenv("WEBSEARCH_FETCH_FULL_CONTENT", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as exc:
        return f"Error fetching content from {url}: {str(exc)}"


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 2,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs. By default it returns snippets only for
    speed; set `WEBSEARCH_FETCH_FULL_CONTENT=true` to fetch each page body too.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 2)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    if not TAVILY_AVAILABLE:
        return "[WARN] Web search is not available. TAVILY_API_KEY is not set. Please configure it in .env to enable live web search."

    search_results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
    )

    fetch_full_content = _should_fetch_full_web_content()
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        snippet = result.get("content", "") or result.get("snippet", "")
        body = fetch_webpage_content(url) if fetch_full_content else snippet
        result_text = f"""## {title}
**URL:** {url}

{body}

---
"""
        result_texts.append(result_text)

    response = f"""[SEARCH] Found {len(result_texts)} result(s) for '{query}':

{chr(10).join(result_texts)}"""
    return response


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on web search progress and decision-making.

    Args:
        reflection: Your detailed reflection on search progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
