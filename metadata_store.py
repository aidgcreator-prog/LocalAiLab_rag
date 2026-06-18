"""Metadata Store for Context Offloading.

Track generated plots, analysis results, and other artifacts separately
from the main context. Reduces token usage significantly.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class MetadataStore:
    """Store and manage metadata for generated artifacts."""
    
    def __init__(self, base_dir: Path = None):
        """Initialize metadata store.
        
        Args:
            base_dir: Base directory for metadata (default: project root)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent
        
        self.base_dir = base_dir
        self.metadata_dir = base_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self._plots_metadata_file = self.metadata_dir / "plots.json"
        self._results_metadata_file = self.metadata_dir / "results.json"
    
    def register_plot(self, plot_name: str, plot_path: str, agent_name: str, 
                      description: str = "", tags: List[str] = None, session_id: str = "default") -> Dict[str, Any]:
        """Register a generated plot.
        
        Args:
            plot_name: Name of the plot
            plot_path: Path to the plot file
            agent_name: Agent that generated it
            description: Plot description
            tags: Optional tags (e.g., ["distribution", "eda"])
            session_id: Session identifier
            
        Returns:
            Plot metadata entry
        """
        metadata = self._load_metadata(self._plots_metadata_file)
        
        if session_id not in metadata:
            metadata[session_id] = []
        
        plot_entry = {
            "name": plot_name,
            "path": plot_path,
            "agent": agent_name,
            "description": description,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
            "exists": Path(plot_path).exists(),
        }
        
        metadata[session_id].append(plot_entry)
        self._save_metadata(self._plots_metadata_file, metadata)
        
        return plot_entry
    
    def get_plots_summary(self, session_id: str = "default", agent_name: str = None) -> str:
        """Get summary of all plots.
        
        Args:
            session_id: Session identifier
            agent_name: Optional filter by agent
            
        Returns:
            Markdown formatted summary
        """
        metadata = self._load_metadata(self._plots_metadata_file)
        plots = metadata.get(session_id, [])
        
        if not plots:
            return "No plots generated yet."
        
        if agent_name:
            plots = [p for p in plots if p["agent"] == agent_name]
        
        summary = "## Generated Plots\n\n"
        
        for plot in sorted(plots, key=lambda x: x["timestamp"]):
            exists_icon = "[OK]" if plot["exists"] else "[ERR]"
            summary += f"{exists_icon} **{plot['name']}**\n"
            summary += f"  - Path: `{plot['path']}`\n"
            summary += f"  - Agent: {plot['agent']}\n"
            if plot["description"]:
                summary += f"  - Description: {plot['description']}\n"
            if plot["tags"]:
                summary += f"  - Tags: {', '.join(plot['tags'])}\n"
            summary += "\n"
        
        return summary
    
    def register_analysis_result(self, result_name: str, result_data: Dict[str, Any], 
                                agent_name: str, session_id: str = "default") -> Dict[str, Any]:
        """Register analysis result metadata.
        
        Args:
            result_name: Name of the result
            result_data: Result data dictionary
            agent_name: Agent that generated it
            session_id: Session identifier
            
        Returns:
            Result metadata entry
        """
        metadata = self._load_metadata(self._results_metadata_file)
        
        if session_id not in metadata:
            metadata[session_id] = {}
        
        result_entry = {
            "name": result_name,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "data_keys": list(result_data.keys()) if isinstance(result_data, dict) else [],
            "data_size": len(str(result_data)),
        }
        
        if session_id not in metadata:
            metadata[session_id] = {}
        
        metadata[session_id][result_name] = result_entry
        self._save_metadata(self._results_metadata_file, metadata)
        
        return result_entry
    
    def get_results_summary(self, session_id: str = "default", agent_name: str = None) -> str:
        """Get summary of all analysis results.
        
        Args:
            session_id: Session identifier
            agent_name: Optional filter by agent
            
        Returns:
            Markdown formatted summary
        """
        metadata = self._load_metadata(self._results_metadata_file)
        results = metadata.get(session_id, {})
        
        if not results:
            return "No analysis results stored yet."
        
        if agent_name:
            results = {k: v for k, v in results.items() if v.get("agent") == agent_name}
        
        summary = "## Analysis Results\n\n"
        
        for result_name, result_data in sorted(results.items(), key=lambda x: x[1].get("timestamp", "")):
            summary += f"[CHART] **{result_name}**\n"
            summary += f"  - Agent: {result_data.get('agent')}\n"
            summary += f"  - Size: {result_data.get('data_size', 0)} bytes\n"
            summary += f"  - Timestamp: {result_data.get('timestamp')}\n"
            if result_data.get("data_keys"):
                summary += f"  - Keys: {', '.join(result_data['data_keys'][:5])}"
                if len(result_data["data_keys"]) > 5:
                    summary += f" ... and {len(result_data['data_keys']) - 5} more"
                summary += "\n"
            summary += "\n"
        
        return summary
    
    def cleanup_session(self, session_id: str):
        """Remove all metadata for a session.
        
        Args:
            session_id: Session identifier
        """
        for metadata_file in [self._plots_metadata_file, self._results_metadata_file]:
            metadata = self._load_metadata(metadata_file)
            if session_id in metadata:
                del metadata[session_id]
                self._save_metadata(metadata_file, metadata)
    
    @staticmethod
    def _load_metadata(file_path: Path) -> Dict[str, Any]:
        """Load metadata from file.
        
        Args:
            file_path: Path to metadata file
            
        Returns:
            Metadata dictionary
        """
        if not file_path.exists():
            return {}
        
        with open(file_path, "r") as f:
            return json.load(f)
    
    @staticmethod
    def _save_metadata(file_path: Path, metadata: Dict[str, Any]):
        """Save metadata to file.
        
        Args:
            file_path: Path to metadata file
            metadata: Metadata dictionary
        """
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


# Global metadata store instance
_metadata_store = None


def get_metadata_store() -> MetadataStore:
    """Get global metadata store instance.
    
    Returns:
        MetadataStore instance
    """
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store
