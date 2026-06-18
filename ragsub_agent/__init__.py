"""RAG Subagent module."""

from ragsub_agent.agent import create_ragsub_agent, invoke_ragsub
from ragsub_agent.prompts import RAG_SUB_INSTRUCTIONS, get_system_prompt
from ragsub_agent.tools import (
    ingest_rag_documents,
    ingest_web_search_results,
    list_rag_documents,
    clear_rag_documents,
    rag_retrieve,
    rag_think_tool,
    ingest_rag_paths,
    get_rag_index_summary,
    get_rag_projects,
    get_rag_available_themes,
    delete_rag_documents,
    get_last_rag_query_diagnostics,
    get_multimodal_vision_status,
)

__all__ = [
    "create_ragsub_agent",
    "invoke_ragsub",
    "RAG_SUB_INSTRUCTIONS",
    "get_system_prompt",
    "ingest_rag_documents",
    "ingest_web_search_results",
    "list_rag_documents",
    "clear_rag_documents",
    "rag_retrieve",
    "rag_think_tool",
    "ingest_rag_paths",
    "get_rag_index_summary",
    "get_rag_projects",
    "get_rag_available_themes",
    "delete_rag_documents",
    "get_last_rag_query_diagnostics",
    "get_multimodal_vision_status",
]
