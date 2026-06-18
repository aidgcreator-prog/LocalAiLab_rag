"""Runtime helpers for DeepAgents orchestration and LangGraph persistence.

Checkpointer  – EventLoopAwareCheckpointer
               Solves the "Lock bound to a different event loop" problem that
               occurs when AsyncSqliteSaver is used with DeepAgents/LangGraph,
               which calls asyncio.run() per invocation (each call creates and
               destroys a temporary event loop).

               Root cause: asyncio.Lock stores self._loop = running_loop on
               first await. If the saver is created/used inside asyncio.run(),
               the Lock references that temporary loop. The next asyncio.run()
               creates a different loop → RuntimeError: Lock bound to different
               event loop.

               Fix: run ONE persistent event loop in a daemon background thread.
               AsyncSqliteSaver is created once in this persistent loop and
               stays bound to it forever.  Each async method uses
               asyncio.wrap_future() to bridge results back to whatever event
               loop is calling (the DeepAgents/LangGraph invocation loop).
               The daemon thread never blocks process exit.

Store         – InMemoryStore (no durable async store backend available yet)

The durable checkpoint DB lives at:  memory_store/langgraph_checkpoints.db
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
from pathlib import Path
from typing import Any

_STORE_DIR = Path(__file__).parent / "memory_store"
_STORE_DIR.mkdir(exist_ok=True)
_CHECKPOINT_DB = _STORE_DIR / "langgraph_checkpoints.db"

LANGGRAPH_CHECKPOINTER = None
LANGGRAPH_STORE = None
LANGGRAPH_PERSISTENCE_STATUS = "[WARN] LangGraph persistence unavailable"

try:
    import aiosqlite  # noqa: F401 – availability check; used lazily in class below
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.memory import InMemoryStore

    class EventLoopAwareCheckpointer(BaseCheckpointSaver):
        """AsyncSqliteSaver proxy backed by a persistent daemon event loop.

        A single event loop runs forever in a background daemon thread.  The
        AsyncSqliteSaver (and its asyncio.Lock) are created once inside this
        loop and stay bound to it permanently.

        Every async method submits the real work to the persistent loop via
        ``asyncio.run_coroutine_threadsafe()`` and bridges the result back to
        the caller's event loop via ``asyncio.wrap_future()``.

        Because the background thread is a daemon thread, it never prevents the
        process from exiting.
        """

        def __init__(self, db_path: Path) -> None:
            super().__init__()
            self._db_path = db_path

            # Start a persistent daemon event loop
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever,
                name="LangGraphCheckpointLoop",
                daemon=True,
            )
            self._loop_thread.start()

            # Create the AsyncSqliteSaver inside the persistent loop
            init_future = asyncio.run_coroutine_threadsafe(
                self._init_saver(), self._loop
            )
            self._saver: AsyncSqliteSaver = init_future.result(timeout=30)

        async def _init_saver(self) -> AsyncSqliteSaver:
            conn = await aiosqlite.connect(str(self._db_path))
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            return saver

        def _submit(self, coro):
            """Submit a coroutine to the persistent loop; return a concurrent.futures.Future."""
            return asyncio.run_coroutine_threadsafe(coro, self._loop)

        async def _run(self, coro):
            """Submit to the persistent loop and await the result in the calling loop."""
            return await asyncio.wrap_future(self._submit(coro))

        # ── async methods (LangGraph graph execution path) ───────────────────

        async def aget(self, config):
            return await self._run(self._saver.aget(config))

        async def aget_tuple(self, config):
            return await self._run(self._saver.aget_tuple(config))

        async def alist(self, config, *, filter=None, before=None, limit=None):
            # Collect all items in the persistent loop, then yield in the caller
            async def _collect():
                items = []
                async for item in self._saver.alist(
                    config, filter=filter, before=before, limit=limit
                ):
                    items.append(item)
                return items

            for item in await self._run(_collect()):
                yield item

        async def aput(self, config, checkpoint, metadata, new_versions):
            return await self._run(
                self._saver.aput(config, checkpoint, metadata, new_versions)
            )

        async def aput_writes(self, config, writes, task_id, task_path=""):
            return await self._run(
                self._saver.aput_writes(config, writes, task_id, task_path)
            )

        async def adelete_thread(self, thread_id):
            return await self._run(self._saver.adelete_thread(thread_id))

        async def acopy_thread(self, source_thread_id, target_thread_id):
            return await self._run(
                self._saver.acopy_thread(source_thread_id, target_thread_id)
            )

        async def adelete_for_runs(self, run_ids):
            return await self._run(self._saver.adelete_for_runs(run_ids))

        async def aprune(self, thread_ids, *, strategy="keep_latest"):
            return await self._run(self._saver.aprune(thread_ids, strategy=strategy))

        # ── sync methods (not used by DeepAgents' async path) ────────────────

        def get(self, config):
            raise NotImplementedError("EventLoopAwareCheckpointer is async-only; use aget().")

        def get_tuple(self, config):
            raise NotImplementedError("Use aget_tuple().")

        def list(self, config, *, filter=None, before=None, limit=None):
            raise NotImplementedError("Use alist().")

        def put(self, config, checkpoint, metadata, new_versions):
            raise NotImplementedError("Use aput().")

        def put_writes(self, config, writes, task_id, task_path=""):
            raise NotImplementedError("Use aput_writes().")

        def delete_thread(self, thread_id):
            raise NotImplementedError("Use adelete_thread().")

        def copy_thread(self, source_thread_id, target_thread_id):
            raise NotImplementedError("Use acopy_thread().")

        def delete_for_runs(self, run_ids):
            raise NotImplementedError("Use adelete_for_runs().")

        def prune(self, thread_ids, *, strategy="keep_latest"):
            raise NotImplementedError("Use aprune().")

        def get_next_version(self, current, channel):
            """Compute the next checkpoint version (pure arithmetic, no DB access)."""
            if current is None:
                current_v = 0
            else:
                current_v = int(str(current).split(".")[0])
            next_v = current_v + 1
            next_h = random.random()
            return f"{next_v:032}.{next_h:016}"

    LANGGRAPH_CHECKPOINTER = EventLoopAwareCheckpointer(_CHECKPOINT_DB)
    LANGGRAPH_STORE = InMemoryStore()
    LANGGRAPH_PERSISTENCE_STATUS = (
        "[OK] LangGraph persistence enabled "
        f"(EventLoopAwareCheckpointer → {_CHECKPOINT_DB.name} | InMemoryStore)"
    )

except Exception as exc:  # pragma: no cover
    LANGGRAPH_PERSISTENCE_STATUS = f"[WARN] LangGraph persistence unavailable: {exc}"


def resolve_user_id(
    explicit_user_id: str | None = None,
    *,
    default_user_id: str = "local-user",
) -> str:
    """Resolve a stable user identity for LangGraph persistence keying."""
    if explicit_user_id and explicit_user_id.strip():
        return explicit_user_id.strip()

    env_user_id = os.getenv("DEEPAGENT_USER_ID", "").strip()
    if env_user_id:
        return env_user_id

    return default_user_id


def build_agent_config(
    thread_id: str,
    *,
    user_id: str | None = None,
    recursion_limit: int | None = None,
    default_user_id: str = "local-user",
    extra_configurable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent LangGraph/DeepAgents invocation config."""
    configurable = {
        "thread_id": thread_id,
        "user_id": resolve_user_id(user_id, default_user_id=default_user_id),
    }
    if extra_configurable:
        configurable.update(extra_configurable)

    config: dict[str, Any] = {"configurable": configurable}
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config
