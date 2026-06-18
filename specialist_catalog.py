from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from data_scientist_agent.tools import (
    execute_python_code,
    install_package,
    render_quarto_report,
    think_tool as ds_think_tool,
)
from presentation_agent.tools import generate_presentation
from ragsub_agent.tools import (
    clear_rag_documents,
    ingest_rag_documents,
    ingest_web_search_results,
    list_rag_documents,
    rag_retrieve,
    rag_think_tool,
)
from websearch_agent.tools import tavily_search, think_tool as websearch_think_tool

try:
    from literature_review.tools import LITERATURE_REVIEW_TOOLS as _lit_tools
except Exception:
    _lit_tools = []

PROJECT_DIR = Path(__file__).parent
SUBAGENT_CONFIG_PATH = PROJECT_DIR / "subagents.yaml"


def load_subagent_configs(config_path: Path | None = None) -> list[dict[str, Any]]:
    """Load subagent definitions from YAML configuration."""
    resolved_path = config_path or SUBAGENT_CONFIG_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Subagent config not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    subagents: list[dict[str, Any]] = []
    for name, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid subagent spec for '{name}': must be a dict")

        required_keys = {"description", "system_prompt"}
        missing_keys = required_keys - set(spec.keys())
        if missing_keys:
            raise ValueError(
                f"Subagent '{name}' missing required keys: {missing_keys}"
            )

        subagents.append(
            {
                "name": name,
                "description": spec["description"],
                "system_prompt": spec["system_prompt"],
            }
        )

    return subagents


def get_subagent_config(name: str, config_path: Path | None = None) -> dict[str, Any]:
    """Return one subagent config from YAML by name."""
    for config in load_subagent_configs(config_path):
        if config["name"] == name:
            return config
    raise ValueError(f"Unknown specialist subagent: {name}")


def get_specialist_tool_map() -> dict[str, list[Any]]:
    """Central tool ownership map for specialists."""
    return {
        "data_scientist": [
            execute_python_code,
            install_package,
            ds_think_tool,
            render_quarto_report,
        ],
        "websearch": [tavily_search, websearch_think_tool],
        "presenter": [generate_presentation],
        "ragsub": [
            ingest_rag_documents,
            ingest_web_search_results,
            list_rag_documents,
            clear_rag_documents,
            rag_retrieve,
            rag_think_tool,
            *_lit_tools,
        ],
    }


def get_specialist_tool_names() -> dict[str, list[str]]:
    """Tool names for UI and registry metadata."""
    return {
        name: [getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in tools]
        for name, tools in get_specialist_tool_map().items()
    }


def build_subagent_specs(
    *,
    names: list[str] | None = None,
    model_resolver: Callable[[str], Any] | None = None,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build DeepAgents-compatible subagent specs from shared config and tools."""
    selected = set(names) if names else None
    tool_map = get_specialist_tool_map()
    specs: list[dict[str, Any]] = []
    for config in load_subagent_configs(config_path):
        if selected and config["name"] not in selected:
            continue
        spec: dict[str, Any] = {
            "name": config["name"],
            "description": config["description"],
            "system_prompt": config["system_prompt"],
        }
        if config["name"] in tool_map:
            spec["tools"] = tool_map[config["name"]]
        if model_resolver:
            spec["model"] = model_resolver(config["name"])
        specs.append(spec)
    return specs
