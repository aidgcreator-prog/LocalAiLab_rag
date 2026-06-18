"""File-Based State Management for Context Offloading.

Store analysis results and metadata in files instead of context.
Reduces token usage and improves scalability.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class StateManager:
    """Manages file-based state for agents."""
    
    def __init__(self, base_dir: Path = None):
        """Initialize state manager.
        
        Args:
            base_dir: Base directory for state files (default: project root)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent
        
        self.base_dir = base_dir
        self.state_dir = base_dir / "agent_state"
        self.state_dir.mkdir(exist_ok=True)
    
    def save_analysis_state(self, agent_name: str, analysis_data: Dict[str, Any], session_id: str = "default"):
        """Save analysis results to file.
        
        Args:
            agent_name: Name of the agent
            analysis_data: Dictionary with analysis results
            session_id: Session identifier
        """
        session_dir = self.state_dir / session_id / agent_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        analysis_data["_timestamp"] = datetime.now().isoformat()
        
        # Save to JSON
        file_path = session_dir / "analysis.json"
        with open(file_path, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        return file_path
    
    def load_analysis_state(self, agent_name: str, session_id: str = "default") -> Optional[Dict[str, Any]]:
        """Load analysis results from file.
        
        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            
        Returns:
            Analysis data dictionary or None if not found
        """
        file_path = self.state_dir / session_id / agent_name / "analysis.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, "r") as f:
            return json.load(f)
    
    def save_todo_state(self, agent_name: str, todos: List[Dict[str, Any]], session_id: str = "default"):
        """Save TODO list to file.
        
        Args:
            agent_name: Name of the agent
            todos: List of TODO items
            session_id: Session identifier
        """
        session_dir = self.state_dir / session_id / agent_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = session_dir / "todos.json"
        with open(file_path, "w") as f:
            json.dump(todos, f, indent=2)
        
        return file_path
    
    def load_todo_state(self, agent_name: str, session_id: str = "default") -> List[Dict[str, Any]]:
        """Load TODO list from file.
        
        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            
        Returns:
            List of TODO items or empty list
        """
        file_path = self.state_dir / session_id / agent_name / "todos.json"
        
        if not file_path.exists():
            return []
        
        with open(file_path, "r") as f:
            return json.load(f)
    
    def save_context_file(self, agent_name: str, filename: str, content: str, session_id: str = "default"):
        """Save arbitrary context to file.
        
        Args:
            agent_name: Name of the agent
            filename: Name of the file
            content: File content
            session_id: Session identifier
        """
        session_dir = self.state_dir / session_id / agent_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = session_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
        
        return file_path
    
    def get_context_summary(self, agent_name: str, session_id: str = "default") -> str:
        """Generate summary of agent's state files.
        
        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            
        Returns:
            Markdown formatted summary
        """
        session_dir = self.state_dir / session_id / agent_name
        
        if not session_dir.exists():
            return f"No state found for {agent_name}"
        
        summary = f"## {agent_name} State Summary\n\n"
        
        # List all files
        files = list(session_dir.glob("*"))
        if files:
            summary += "**Files:**\n"
            for file_path in sorted(files):
                size = file_path.stat().st_size if file_path.is_file() else 0
                summary += f"- `{file_path.name}` ({size} bytes)\n"
        
        return summary


# Global state manager instance
_state_manager = None


def get_state_manager() -> StateManager:
    """Get global state manager instance.
    
    Returns:
        StateManager instance
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
