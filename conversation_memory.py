"""Conversation search and self-improvement helpers.

This module is the search and learning layer. It is NOT the primary persistence
path — LangGraph's SqliteSaver (agent_runtime.py) owns conversation state
persistence. This module's distinct responsibilities are:

- FTS full-text search across past conversation transcripts
- Human-readable JSON transcript export (also serves as the FTS input source)
- A compact self-improvement learning log the agent can update at runtime
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


@dataclass(slots=True)
class SearchHit:
	"""A single recalled message from a previous session."""

	session_id: str
	session_name: str
	created: str
	role: str
	snippet: str
	score: float


class ConversationMemoryStore:
	"""Transcript search index, JSON export, and self-improvement learning journal.

	Primary state persistence is handled by LangGraph's SqliteSaver (see
	agent_runtime.py). This class adds full-text search over human-readable
	transcript exports and a durable learning log.
	"""

	def __init__(self, base_dir: Path | None = None):
		self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
		self.chat_history_dir = self.base_dir / "chat_history"
		self.store_dir = self.base_dir / "memory_store"
		self.repo_memory_dir = self.base_dir / "memories" / "repo"
		self.db_path = self.store_dir / "conversation_memory.db"
		self.learning_file = self.repo_memory_dir / "agent_learnings.md"

		self.chat_history_dir.mkdir(exist_ok=True)
		self.store_dir.mkdir(exist_ok=True)
		self.repo_memory_dir.mkdir(parents=True, exist_ok=True)
		self._ensure_learning_file()
		self._init_db()

	def _connect(self) -> sqlite3.Connection:
		conn = sqlite3.connect(self.db_path)
		conn.row_factory = sqlite3.Row
		return conn

	def _init_db(self) -> None:
		with self._connect() as conn:
			conn.executescript(
				"""
				CREATE TABLE IF NOT EXISTS sessions (
					session_id TEXT PRIMARY KEY,
					name TEXT NOT NULL,
					created TEXT NOT NULL,
					model TEXT NOT NULL DEFAULT '',
					source_mtime REAL NOT NULL DEFAULT 0,
					updated_at TEXT NOT NULL
				);

				CREATE TABLE IF NOT EXISTS messages (
					session_id TEXT NOT NULL,
					message_index INTEGER NOT NULL,
					role TEXT NOT NULL,
					content TEXT NOT NULL,
					PRIMARY KEY (session_id, message_index)
				);

				CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
					session_id,
					role,
					content,
					tokenize='unicode61'
				);
				"""
			)

	def _ensure_learning_file(self) -> None:
		if self.learning_file.exists():
			return

		self.learning_file.write_text(
			"# Agent Learnings\n\n"
			"Compact reusable notes discovered during real work.\n"
			"Store durable patterns here, not task-specific logs.\n\n"
			"## Workflow\n"
			"- (none yet)\n\n"
			"## Tooling\n"
			"- (none yet)\n\n"
			"## Quality\n"
			"- (none yet)\n",
			encoding="utf-8",
		)

	@staticmethod
	def build_session_name(messages: list[dict[str, str]], fallback_session_id: str) -> str:
		"""Build a stable human-readable session name from the first user turn."""
		first_user = next(
			(message.get("content", "")[:60] for message in messages if message.get("role") == "user"),
			"Untitled",
		)
		name = re.sub(r"\s+", " ", first_user).strip() or fallback_session_id[:20]
		if len(name) > 55:
			name = name[:55] + "..."
		return name

	def persist_session(
		self,
		session_id: str,
		messages: list[dict[str, str]],
		model: str = "",
		created: str | None = None,
		name: str | None = None,
		update_index: bool = True,
	) -> dict[str, Any]:
		"""Export a session transcript to JSON and refresh the FTS search index.

		NOTE: This is the FTS/export layer. LangGraph SqliteSaver independently
		persists the conversation state checkpoint for resumability.
		"""
		clean_messages = [
			{
				"role": str(message.get("role", "assistant")),
				"content": str(message.get("content", "")),
			}
			for message in messages
			if message.get("content")
		]
		if not clean_messages:
			return {}

		session_name = name or self.build_session_name(clean_messages, session_id)
		created_at = created or datetime.now().strftime("%Y-%m-%d %H:%M")
		session_data = {
			"name": session_name,
			"messages": clean_messages,
			"created": created_at,
			"model": model,
		}

		file_path = self.chat_history_dir / f"{session_id}.json"
		file_path.write_text(
			json.dumps(session_data, ensure_ascii=False, indent=1),
			encoding="utf-8",
		)
		if update_index:
			self._index_session(session_id, session_data, file_path.stat().st_mtime)
		return session_data

	def sync_chat_history_index(self) -> None:
		"""Sync the SQLite search index with saved chat history files."""
		with self._connect() as conn:
			known = {
				row["session_id"]: row["source_mtime"]
				for row in conn.execute("SELECT session_id, source_mtime FROM sessions")
			}

		for file_path in self.chat_history_dir.glob("*.json"):
			session_id = file_path.stem
			current_mtime = file_path.stat().st_mtime
			if known.get(session_id, -1) >= current_mtime:
				continue

			try:
				session_data = json.loads(file_path.read_text(encoding="utf-8"))
			except Exception:
				continue

			self._index_session(session_id, session_data, current_mtime)

	def _index_session(self, session_id: str, session_data: dict[str, Any], source_mtime: float) -> None:
		messages = session_data.get("messages", [])
		now = datetime.now().isoformat()
		with self._connect() as conn:
			conn.execute(
				"""
				INSERT INTO sessions(session_id, name, created, model, source_mtime, updated_at)
				VALUES (?, ?, ?, ?, ?, ?)
				ON CONFLICT(session_id) DO UPDATE SET
					name = excluded.name,
					created = excluded.created,
					model = excluded.model,
					source_mtime = excluded.source_mtime,
					updated_at = excluded.updated_at
				""",
				(
					session_id,
					session_data.get("name", session_id),
					session_data.get("created", ""),
					session_data.get("model", ""),
					source_mtime,
					now,
				),
			)
			conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
			conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
			conn.executemany(
				"INSERT INTO messages(session_id, message_index, role, content) VALUES (?, ?, ?, ?)",
				[
					(
						session_id,
						index,
						str(message.get("role", "assistant")),
						str(message.get("content", "")),
					)
					for index, message in enumerate(messages)
					if message.get("content")
				],
			)
			conn.executemany(
				"INSERT INTO messages_fts(session_id, role, content) VALUES (?, ?, ?)",
				[
					(
						session_id,
						str(message.get("role", "assistant")),
						str(message.get("content", "")),
					)
					for message in messages
					if message.get("content")
				],
			)

	@staticmethod
	def _normalize_search_query(query: str) -> str:
		tokens = re.findall(r"[A-Za-z0-9_]{2,}", query.lower())
		if not tokens:
			return '"' + query.replace('"', ' ') + '"'
		return " OR ".join(f'"{token}"' for token in tokens[:8])

	def search_sessions(
		self,
		query: str,
		max_results: int = 5,
		exclude_session_id: str | None = None,
	) -> list[SearchHit]:
		"""Search prior session transcripts and return ranked hits."""
		self.sync_chat_history_index()
		search_query = self._normalize_search_query(query)
		sql = (
			"""
			SELECT
				messages_fts.session_id,
				sessions.name,
				sessions.created,
				messages_fts.role,
				snippet(messages_fts, 2, '[', ']', ' ... ', 20) AS snippet,
				bm25(messages_fts) AS score
			FROM messages_fts
			JOIN sessions ON sessions.session_id = messages_fts.session_id
			WHERE messages_fts MATCH ?
			"""
		)
		params: list[Any] = [search_query]
		if exclude_session_id:
			sql += " AND messages_fts.session_id != ?"
			params.append(exclude_session_id)
		sql += " ORDER BY bm25(messages_fts), sessions.updated_at DESC LIMIT ?"
		params.append(max(1, min(max_results, 10)))

		with self._connect() as conn:
			try:
				rows = conn.execute(sql, params).fetchall()
			except sqlite3.OperationalError:
				return []

		return [
			SearchHit(
				session_id=row["session_id"],
				session_name=row["name"],
				created=row["created"],
				role=row["role"],
				snippet=row["snippet"],
				score=float(row["score"]),
			)
			for row in rows
		]

	def record_feedback(
		self,
		rating: str,
		user_prompt: str,
		agent_response: str,
		reason: str = "",
	) -> str:
		"""Persist a user feedback rating into the agent learnings file.

		Args:
			rating: One of 'good', 'very_good', or 'bad'.
			user_prompt: The user message that triggered the response.
			agent_response: The AI response being rated (truncated for brevity).
			reason: Optional free-text explanation (used for 'bad' ratings).
		"""
		if not _self_learning_enabled():
			return "Self-learning is disabled."

		date = datetime.now().date().isoformat()
		prompt_preview = user_prompt.strip()[:120].replace("\n", " ")
		response_preview = agent_response.strip()[:200].replace("\n", " ")

		if rating == "very_good":
			bullet = (
				f"- {date} [❤️ VERY GOOD] Prompt: \"{prompt_preview}\" → "
				f"Response excerpt: \"{response_preview}\""
			)
			category = "quality"
		elif rating == "good":
			bullet = (
				f"- {date} [👍 GOOD] Prompt: \"{prompt_preview}\" — "
				f"user found this response helpful."
			)
			category = "quality"
		else:  # bad
			extra = f" Reason: \"{reason.strip()}\"" if reason.strip() else ""
			bullet = (
				f"- {date} [👎 BAD] Prompt: \"{prompt_preview}\" → "
				f"Response excerpt: \"{response_preview}\".{extra} "
				f"Avoid similar approaches for this type of request."
			)
			category = "quality"

		return self.record_learning(learning=bullet, category=category)

	def record_learning(self, learning: str, category: str = "workflow") -> str:
		"""Append a compact reusable note to the agent learning file."""
		if not _self_learning_enabled():
			return "Self-learning is disabled."

		normalized = learning.strip()
		if not normalized:
			return "No learning recorded."

		heading = {
			"workflow": "Workflow",
			"tooling": "Tooling",
			"quality": "Quality",
		}.get(category.lower(), "Workflow")
		bullet = f"- {datetime.now().date().isoformat()}: {normalized}"

		content = self.learning_file.read_text(encoding="utf-8")
		if normalized.lower() in content.lower():
			return "Learning already recorded."

		placeholder = f"## {heading}\n- (none yet)"
		if placeholder in content:
			content = content.replace(placeholder, f"## {heading}\n{bullet}")
		else:
			marker = f"## {heading}\n"
			if marker in content:
				content = content.replace(marker, f"{marker}{bullet}\n", 1)
			else:
				content = content.rstrip() + f"\n\n## {heading}\n{bullet}\n"

		self.learning_file.write_text(content, encoding="utf-8")
		return f"Recorded learning under {heading}."


_memory_store: ConversationMemoryStore | None = None


def _self_learning_enabled() -> bool:
	"""Return whether self-learning writes are enabled for this process."""
	return os.getenv("SELF_LEARNING_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def get_conversation_memory_store() -> ConversationMemoryStore:
	"""Return the singleton conversation memory store."""
	global _memory_store
	if _memory_store is None:
		_memory_store = ConversationMemoryStore()
	return _memory_store


@tool(parse_docstring=True)
def session_search(query: str, max_results: int = 5) -> str:
	"""Search saved conversation history for prior relevant context.

	Use this before asking the user to repeat prior decisions, facts, or outputs.
	Search results span saved Streamlit chats and API-server threads persisted by this app.

	Args:
		query: The concept, decision, topic, or phrase to search for across past sessions.
		max_results: Maximum number of recalled snippets to return. Keep this small.

	Returns:
		A formatted summary of the most relevant prior conversation snippets.
	"""
	hits = get_conversation_memory_store().search_sessions(query=query, max_results=max_results)
	if not hits:
		return f"No prior conversation context found for '{query}'."

	lines = [f"Found {len(hits)} prior conversation match(es) for '{query}':"]
	for hit in hits:
		lines.append(
			f"- Session: {hit.session_name} ({hit.session_id}, {hit.created}) [{hit.role}]\n"
			f"  Snippet: {hit.snippet}"
		)
	return "\n".join(lines)


@tool(parse_docstring=True)
def record_learning(learning: str, category: str = "workflow") -> str:
	"""Record a compact reusable learning discovered during work.

	Use this after solving a tricky issue or discovering a stable workflow that will matter later.
	Do not use it for temporary progress logs, completed TODOs, or one-off outcomes.

	Args:
		learning: One concise, reusable lesson or pattern.
		category: Learning bucket. Prefer 'workflow', 'tooling', or 'quality'.

	Returns:
		Confirmation that the learning note was persisted.
	"""
	return get_conversation_memory_store().record_learning(learning=learning, category=category)
