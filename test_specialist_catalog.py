import agent
from agent_registry import AgentType, get_registry
from specialist_catalog import build_subagent_specs, get_specialist_tool_names


def test_shared_specialist_catalog_keeps_ragsub_out_of_presentation_ownership():
    tool_names = get_specialist_tool_names()

    assert "generate_presentation" not in tool_names["ragsub"]
    assert any(name.startswith("generate_presentation") for name in tool_names["presenter"])


def test_built_ragsub_spec_uses_shared_tool_contract():
    ragsub_spec = build_subagent_specs(names=["ragsub"])[0]
    tool_names = {tool.name for tool in ragsub_spec["tools"]}

    assert "generate_presentation" not in tool_names
    assert "presenter rather than render the final deck" in ragsub_spec["system_prompt"]


def test_agent_registry_ragsub_metadata_matches_shared_contract():
    registry = get_registry()
    metadata = registry.get_metadata(AgentType.RAG_SUB)

    assert "generate_presentation" not in metadata["tools"]
    assert "grounded slide outlines" in metadata["description"]


def test_main_agent_uses_shared_ragsub_tool_contract():
    tool_names = {tool.name for tool in agent.SUBAGENT_TOOL_MAP["ragsub"]}

    assert "generate_presentation" not in tool_names
