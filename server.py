"""Async Subagent Server — Agent Protocol over FastAPI.

A self-hosted Agent Protocol server that exposes the lv_combined_agents
orchestrator as an async subagent. Any DeepAgents supervisor can connect
to this server using the AsyncSubAgent configuration.

Implements the Agent Protocol endpoints:

    POST /threads                              create a thread
    POST /threads/{thread_id}/runs             start (or interrupt+restart) a run
    GET  /threads/{thread_id}/runs/{run_id}    poll run status
    GET  /threads/{thread_id}                  fetch thread (values.messages)
    POST /threads/{thread_id}/runs/{run_id}/cancel  cancel a run
    GET  /ok                                   health check

Persistence uses a disk-backed SQLite database (agent_protocol.db, no setup required).

Run:
    uvicorn server:app --port 2024

Then point a DeepAgents supervisor at:
    http://localhost:2024
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from agent_runtime import build_agent_config, resolve_user_id
from conversation_memory import get_conversation_memory_store
from message_sanitizer import coerce_message_content_to_text

load_dotenv(Path(__file__).parent / ".env")

# ── Database ───────────────────────────────────────────────────────────────

_db_path = Path(__file__).parent / "agent_protocol.db"
_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
_conn.row_factory = sqlite3.Row
conversation_memory = get_conversation_memory_store()


def _init_db() -> None:
    """Create the threads and runs tables."""
    _conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id  TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            messages   TEXT NOT NULL DEFAULT '[]',
            values_    TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            thread_id    TEXT NOT NULL REFERENCES threads(thread_id),
            assistant_id TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            error        TEXT
        );
    """)
    _conn.commit()


# ── DB helpers ────────────────────────────────────────────────────────────────


def _get_thread(thread_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT thread_id, created_at, messages, values_ FROM threads WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "thread_id": row["thread_id"],
        "created_at": row["created_at"],
        "messages": json.loads(row["messages"]),
        "values": json.loads(row["values_"]),
    }


def _get_run(run_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT run_id, thread_id, assistant_id, status, created_at, error FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


# ── Agent ─────────────────────────────────────────────────────────────────────

# Lazy-load to avoid circular imports and heavy init at import time
_agent = None


def _get_agent():
    """Lazy-load the orchestrator agent."""
    global _agent
    if _agent is None:
        from agent import agent
        _agent = agent
    return _agent


# ── Run executor ──────────────────────────────────────────────────────────────

async def _execute_run(
    run_id: str,
    thread_id: str,
    user_message: str,
    user_id: str,
) -> None:
    """Invoke the agent and persist the result."""
    _conn.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,))
    _conn.commit()
    try:
        from langchain_core.messages import HumanMessage

        agent = _get_agent()
        result = await agent.ainvoke(
            {"messages": [HumanMessage(user_message)]},
            config=build_agent_config(
                thread_id=thread_id,
                user_id=user_id,
                default_user_id="agent-protocol-user",
            ),
        )
        last = result["messages"][-1]
        output = last.content if isinstance(last.content, str) else json.dumps(last.content)
        assistant_msg = {"role": "assistant", "content": output}

        row = _conn.execute(
            "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        msgs = json.loads(row[0]) if row else []
        msgs.append(assistant_msg)
        serialized = json.dumps(msgs)
        _conn.execute(
            "UPDATE threads SET messages = ?, values_ = ? WHERE thread_id = ?",
            (serialized, json.dumps({"messages": msgs}), thread_id),
        )
        _conn.execute("UPDATE runs SET status = 'success' WHERE run_id = ?", (run_id,))
        _conn.commit()
        conversation_memory.persist_session(
            session_id=thread_id,
            messages=msgs,
            model=os.getenv("DEEPAGENT_MODEL", ""),
            created=_get_thread(thread_id).get("created_at", "") if _get_thread(thread_id) else None,
        )
    except Exception as exc:
        detailed_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _conn.execute(
            "UPDATE runs SET status = 'error', error = ? WHERE run_id = ?",
            (detailed_error, run_id),
        )
        _conn.commit()


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _init_db()
    print("[OK] Agent Protocol server ready")
    yield


app = FastAPI(
    title="LV Combined Agents - Agent Protocol Server",
    description="Exposes the multi-agent orchestrator via Agent Protocol endpoints",
    lifespan=_lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ok")
async def health() -> dict[str, bool]:
    """Health check."""
    return {"ok": True}


@app.post("/threads")
async def create_thread() -> dict[str, Any]:
    """Create a thread."""
    thread_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    _conn.execute(
        "INSERT INTO threads (thread_id, created_at) VALUES (?, ?)",
        (thread_id, now),
    )
    _conn.commit()
    return {"thread_id": thread_id, "created_at": now, "messages": [], "values": {}}


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request) -> dict[str, Any]:
    """Create a run on an existing thread.

    Supports multitask_strategy='interrupt' to cancel running tasks and restart.
    """
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    body = await request.json()
    multitask_strategy = body.get("multitask_strategy")

    if multitask_strategy == "interrupt":
        _conn.execute(
            "UPDATE runs SET status = 'cancelled' WHERE thread_id = ? AND status = 'running'",
            (thread_id,),
        )
        _conn.execute(
            "UPDATE threads SET values_ = '{}' WHERE thread_id = ?",
            (thread_id,),
        )
        _conn.commit()

    messages = (body.get("input") or {}).get("messages") or []
    user_messages = [
        coerce_message_content_to_text(m.get("content"))
        for m in messages
        if m.get("role") == "user"
    ]
    user_message = user_messages[-1] if user_messages else ""
    requested_user_id = (
        body.get("user_id")
        or (body.get("metadata") or {}).get("user_id")
    )
    user_id = resolve_user_id(
        requested_user_id,
        default_user_id="agent-protocol-user",
    )

    if user_message:
        existing = json.loads(
            _conn.execute(
                "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()[0]
        )
        existing.append({"role": "user", "content": user_message})
        _conn.execute(
            "UPDATE threads SET messages = ? WHERE thread_id = ?",
            (json.dumps(existing), thread_id),
        )
        _conn.commit()
        conversation_memory.persist_session(
            session_id=thread_id,
            messages=existing,
            model=os.getenv("DEEPAGENT_MODEL", ""),
            created=thread.get("created_at", ""),
        )

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    assistant_id = body.get("assistant_id") or "orchestrator"
    _conn.execute(
        "INSERT INTO runs (run_id, thread_id, assistant_id, created_at) VALUES (?, ?, ?, ?)",
        (run_id, thread_id, assistant_id, now),
    )
    _conn.commit()

    # Fire and forget
    asyncio.ensure_future(_execute_run(run_id, thread_id, user_message, user_id))

    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": "pending",
        "created_at": now,
        "error": None,
    }


@app.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """Get run status."""
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """Get thread state including messages."""
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """Cancel a run."""
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    _conn.execute("UPDATE runs SET status = 'cancelled' WHERE run_id = ?", (run_id,))
    _conn.commit()
    return {**run, "status": "cancelled"}
