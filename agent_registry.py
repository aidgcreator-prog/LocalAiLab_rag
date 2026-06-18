"""SubAgent registry backed by shared specialist definitions."""

from typing import Dict, Callable, Any
from enum import Enum
from model_config import get_agent_model_override
from specialist_catalog import get_specialist_tool_names, load_subagent_configs


class AgentType(Enum):
    """Available agent types."""
    AUTO = "auto"
    WEBSEARCH = "websearch"
    RESEARCHER = "websearch"
    DATA_SCIENTIST = "data_scientist"
    RAG_SUB = "ragsub"
    WRITER = "writer"
    CODER = "coder"
    REVIEWER = "reviewer"
    PRESENTER = "presenter"
    PLANNER = "planner"


class SubAgentRegistry:
    """Registry for managing specialized sub-agents with consistent interface."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[AgentType, Dict[str, Any]] = {}
        self._loaders: Dict[AgentType, Callable] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default agent metadata and loaders."""
        config_by_name = {item["name"]: item for item in load_subagent_configs()}
        tool_names = get_specialist_tool_names()
        
        # Data Scientist Agent
        self.register(
            AgentType.DATA_SCIENTIST,
            {
                "name": "[CHART] Data Scientist (Data Analysis & EDA)",
                "description": config_by_name["data_scientist"]["description"],
                "emoji": "[CHART]",
                "model_override": get_agent_model_override("data_scientist"),
                "tools": tool_names.get("data_scientist", []),
            },
            self._load_data_scientist_agent
        )
        
        # Web Search Agent
        self.register(
            AgentType.WEBSEARCH,
            {
                "name": "[SEARCH] Web Search (Live Web Search)",
                "description": config_by_name["websearch"]["description"],
                "emoji": "[SEARCH]",
                "tools": tool_names.get("websearch", []),
            },
            self._load_websearch_agent
        )

        # RAG SubAgent
        self.register(
            AgentType.RAG_SUB,
            {
                "name": "[BOOKS] RAG SubAgent (Document Retrieval + Rerank)",
                "description": config_by_name["ragsub"]["description"],
                "emoji": "[BOOKS]",
                "model_override": get_agent_model_override("ragsub"),
                "tools": tool_names.get("ragsub", []),
            },
            self._load_ragsub_agent,
        )
        
        # Writer Agent
        self.register(
            AgentType.WRITER,
            {
                "name": "[WRITE] Writer (Documentation & Content)",
                "description": config_by_name["writer"]["description"],
                "emoji": "[WRITE]",
                "tools": tool_names.get("writer", []),
            },
            self._load_writer_agent
        )
        
        # Coder Agent
        self.register(
            AgentType.CODER,
            {
                "name": "[CODE] Coder (Code Implementation)",
                "description": config_by_name["coder"]["description"],
                "emoji": "[CODE]",
                "tools": tool_names.get("coder", []),
            },
            self._load_coder_agent
        )
        
        # Reviewer Agent
        self.register(
            AgentType.REVIEWER,
            {
                "name": "[REVIEW] Reviewer (Quality & Testing)",
                "description": config_by_name["reviewer"]["description"],
                "emoji": "[REVIEW]",
                "tools": tool_names.get("reviewer", []),
            },
            self._load_reviewer_agent
        )
        
        # Presenter Agent
        self.register(
            AgentType.PRESENTER,
            {
                "name": "[PRESENT] Presenter (PowerPoint Slides)",
                "description": config_by_name["presenter"]["description"],
                "emoji": "[PRESENT]",
                "tools": tool_names.get("presenter", []),
            },
            self._load_presenter_agent
        )
        
        # Planner Agent
        self.register(
            AgentType.PLANNER,
            {
                "name": "[PLAN] Planner (Planning & Roadmaps)",
                "description": config_by_name["planner"]["description"],
                "emoji": "[PLAN]",
                "tools": tool_names.get("planner", []),
            },
            self._load_planner_agent
        )
    
    def register(self, agent_type: AgentType, metadata: Dict[str, Any], loader: Callable):
        """Register an agent with metadata and loader.
        
        Args:
            agent_type: AgentType enum
            metadata: Agent metadata (name, description, etc.)
            loader: Callable that loads the agent
        """
        self._agents[agent_type] = metadata
        self._loaders[agent_type] = loader
    
    def get_metadata(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get agent metadata.
        
        Args:
            agent_type: AgentType enum
            
        Returns:
            Agent metadata dictionary
        """
        return self._agents.get(agent_type, {})
    
    def get_all_agents(self) -> Dict[str, str]:
        """Get all available agents as display name -> type mapping.
        
        Returns:
            Dictionary mapping display names to agent types
        """
        return {
            "Auto (Main Agent Decides)": AgentType.AUTO.value,
            **{meta["name"]: agent_type.value for agent_type, meta in self._agents.items()}
        }
    
    def load_agent(self, agent_type: AgentType):
        """Load and return an agent instance.
        
        Args:
            agent_type: AgentType enum
            
        Returns:
            Loaded agent
        """
        if agent_type not in self._loaders:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        loader = self._loaders[agent_type]
        return loader()
    
    # Lazy loaders for each agent type
    @staticmethod
    def _load_data_scientist_agent():
        """Load data scientist agent."""
        from data_scientist_agent import create_data_scientist_agent
        return create_data_scientist_agent()
    
    @staticmethod
    def _load_websearch_agent():
        """Load web search agent."""
        from specialist_router import create_specialist_router_agent
        return create_specialist_router_agent("websearch")

    @staticmethod
    def _load_ragsub_agent():
        """Load RAG subagent."""
        from ragsub_agent import create_ragsub_agent
        return create_ragsub_agent()
    
    @staticmethod
    def _load_writer_agent():
        """Load writer agent."""
        from specialist_router import create_specialist_router_agent
        return create_specialist_router_agent("writer")
    
    @staticmethod
    def _load_coder_agent():
        """Load coder agent."""
        from specialist_router import create_specialist_router_agent
        return create_specialist_router_agent("coder")
    
    @staticmethod
    def _load_reviewer_agent():
        """Load reviewer agent."""
        from specialist_router import create_specialist_router_agent
        return create_specialist_router_agent("reviewer")
    
    @staticmethod
    def _load_presenter_agent():
        """Load presenter agent."""
        from presentation_agent import create_presenter_agent
        return create_presenter_agent()
    
    @staticmethod
    def _load_planner_agent():
        """Load planner agent."""
        from specialist_router import create_specialist_router_agent
        return create_specialist_router_agent("planner")


# Global registry instance
_registry = SubAgentRegistry()


def get_registry() -> SubAgentRegistry:
    """Get the global agent registry.
    
    Returns:
        SubAgentRegistry instance
    """
    return _registry
