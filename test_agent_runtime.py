from agent_runtime import (
    LANGGRAPH_CHECKPOINTER,
    LANGGRAPH_PERSISTENCE_STATUS,
    LANGGRAPH_STORE,
    build_agent_config,
    resolve_user_id,
)


def test_build_agent_config_includes_thread_and_user():
    config = build_agent_config(
        thread_id="thread-1",
        user_id="user-1",
        recursion_limit=150,
    )

    assert config["configurable"]["thread_id"] == "thread-1"
    assert config["configurable"]["user_id"] == "user-1"
    assert config["recursion_limit"] == 150


def test_resolve_user_id_uses_default_when_not_provided(monkeypatch):
    monkeypatch.delenv("DEEPAGENT_USER_ID", raising=False)

    user_id = resolve_user_id(default_user_id="fallback-user")

    assert user_id == "fallback-user"


def test_langgraph_runtime_objects_are_initialized_when_available():
    assert isinstance(LANGGRAPH_PERSISTENCE_STATUS, str)
    assert LANGGRAPH_CHECKPOINTER is not None
    assert LANGGRAPH_STORE is not None
