"""Standalone RAG subagent with dedicated model support."""

import re
from functools import lru_cache
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from agent_runtime import build_agent_config
from model_config import create_chat_model, get_agent_model, resolve_provider_and_model
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import get_system_prompt
from .tools import (
    clear_rag_documents,
    ingest_rag_documents,
    list_rag_documents,
    rag_retrieve,
    rag_think_tool,
)

try:
    from literature_review.tools import LITERATURE_REVIEW_TOOLS
except Exception:
    LITERATURE_REVIEW_TOOLS = []

PROJECT_DIR = Path(__file__).parent.parent


def _latest_user_text(messages: list) -> str:
    for role, content in reversed(messages):
        if role == "user":
            return str(content or "")
    return ""


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"^\[{re.escape(tag)}:\s*(.*?)\]\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _strip_rag_tags(text: str) -> str:
    cleaned = re.sub(r"^\[RAG [^\]]+\]\s*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n\n+", "\n\n", cleaned).strip()


def _parse_rag_request(text: str) -> dict[str, str | int]:
    def _int_tag(tag: str, default: int) -> int:
        raw = _extract_tag(text, tag)
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    return {
        "query": _strip_rag_tags(text),
        "project": _extract_tag(text, "RAG PROJECT"),
        "themes": _extract_tag(text, "RAG THEMES"),
        "mode": _extract_tag(text, "RAG MODE") or "Top-K Globally",
        "top_k": _int_tag("RAG TOP_K", 5),
        "fetch_k": _int_tag("RAG FETCH_K", 100),
        "max_files": _int_tag("RAG MAX_FILES", 5),
        "uploaded_files": _extract_tag(text, "RAG UPLOADED FILES"),
    }


def _should_fallback_to_direct_rag(exc: Exception) -> bool:
    message = str(exc).lower()
    provider, _ = resolve_provider_and_model(get_agent_model("ragsub"))
    if provider == "llama_cpp":
        return "exceed context window" in message
    if provider == "llama_server":
        return (
            "exceed_context_size_error" in message
            or "exceeds the available context size" in message
            or "available context size" in message
        )
    return False


def _invoke_direct_rag(messages: list) -> dict:
    latest_user_text = _latest_user_text(messages)
    request = _parse_rag_request(latest_user_text)

    uploaded_files = str(request["uploaded_files"] or "").strip()
    if uploaded_files:
        ingest_rag_documents(
            file_paths=uploaded_files,
            project=str(request["project"] or "Default") or "Default",
            theme=str(request["themes"] or ""),
        )

    retrieved_context = rag_retrieve(
        query=str(request["query"] or latest_user_text),
        top_k=int(request["top_k"]),
        mode=str(request["mode"]),
        max_files=int(request["max_files"]),
        fetch_k=int(request["fetch_k"]),
        project=str(request["project"]),
        themes=str(request["themes"]),
    )

    llm = create_chat_model(model_name=get_agent_model("ragsub"), temperature=0)
    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a compact RAG answerer. Answer only from the provided retrieved context. "
                "If the context is insufficient, say so. Preserve citation tokens exactly as given."
            )
        ),
        HumanMessage(
            content=(
                f"User question:\n{request['query'] or latest_user_text}\n\n"
                f"Retrieved context:\n{retrieved_context}"
            )
        ),
    ])
    return {"messages": [response], "output": response.content}


@lru_cache(maxsize=1)
def create_ragsub_agent():
    """Create a specialized RAG subagent."""
    rag_model_name = get_agent_model("ragsub")

    try:
        model = create_chat_model(model_name=rag_model_name, temperature=0)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize RAG subagent model '{rag_model_name}': {e}"
        ) from e

    agent = create_deep_agent(
        model=model,
        memory=[],
        skills=[],
        subagents=[],
        backend=FilesystemBackend(root_dir=PROJECT_DIR, virtual_mode=True),
        tools=[
            ingest_rag_documents,
            list_rag_documents,
            clear_rag_documents,
            rag_retrieve,
            rag_think_tool,
            *LITERATURE_REVIEW_TOOLS,
        ],
        system_prompt=get_system_prompt(),
    )
    return agent


async def invoke_ragsub(
    messages: list,
    thread_id: str = "ragsub",
    user_id: str | None = None,
) -> dict:
    """Invoke the RAG subagent."""
    agent = create_ragsub_agent()
    try:
        return await agent.ainvoke(
            {"messages": messages},
            config=build_agent_config(
                thread_id=thread_id,
                user_id=user_id,
                recursion_limit=90,
                default_user_id="ragsub-user",
            ),
        )
    except Exception as exc:
        if _should_fallback_to_direct_rag(exc):
            return _invoke_direct_rag(messages)
        raise
