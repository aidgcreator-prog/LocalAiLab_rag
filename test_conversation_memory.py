from conversation_memory import ConversationMemoryStore


def test_session_search_indexes_persisted_sessions(tmp_path):
    store = ConversationMemoryStore(base_dir=tmp_path)
    store.persist_session(
        session_id="chat-1",
        messages=[
            {"role": "user", "content": "Use Tavily for live market research"},
            {"role": "assistant", "content": "I used Tavily and summarized the findings."},
        ],
        model="test-model",
        created="2026-04-18 10:00",
        name="Research Session",
    )

    hits = store.search_sessions("Tavily market research", max_results=3)

    assert hits
    assert hits[0].session_id == "chat-1"
    assert "Tavily" in hits[0].snippet or "tavily" in hits[0].snippet.lower()


def test_record_learning_updates_markdown_file(tmp_path):
    store = ConversationMemoryStore(base_dir=tmp_path)

    result = store.record_learning(
        "Use session_search before asking the user to restate earlier architecture decisions.",
        category="workflow",
    )

    learning_text = store.learning_file.read_text(encoding="utf-8")
    assert "Recorded learning" in result
    assert "session_search before asking the user" in learning_text
