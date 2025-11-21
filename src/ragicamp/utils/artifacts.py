"""Simple artifact management for saving/loading agents and retrievers."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class ArtifactManager:
    """Simple manager for saving/loading artifacts (indices, configs, etc.)."""

    def __init__(self, base_dir: str = "artifacts"):
        """Initialize artifact manager.

        Args:
            base_dir: Base directory for all artifacts
        """
        self.base_dir = Path(base_dir)
        self.retrievers_dir = self.base_dir / "retrievers"
        self.agents_dir = self.base_dir / "agents"

        # Create directories if they don't exist
        self.retrievers_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    def get_retriever_path(self, name: str) -> Path:
        """Get path for a retriever artifact.

        Args:
            name: Retriever artifact name (e.g., 'wikipedia_nq_v1')

        Returns:
            Path to retriever directory
        """
        path = self.retrievers_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_agent_path(self, name: str) -> Path:
        """Get path for an agent artifact.

        Args:
            name: Agent artifact name (e.g., 'fixed_rag_wikipedia_v1')

        Returns:
            Path to agent directory
        """
        path = self.agents_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save dictionary as JSON.

        Args:
            data: Dictionary to save
            path: Path to save to
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file.

        Args:
            path: Path to load from

        Returns:
            Loaded dictionary
        """
        with open(path, "r") as f:
            return json.load(f)

    def save_pickle(self, obj: Any, path: Path) -> None:
        """Save object as pickle.

        Args:
            obj: Object to save
            path: Path to save to
        """
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, path: Path) -> Any:
        """Load pickle file.

        Args:
            path: Path to load from

        Returns:
            Loaded object
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_retrievers(self) -> list:
        """List all saved retrievers.

        Returns:
            List of retriever names
        """
        return [d.name for d in self.retrievers_dir.iterdir() if d.is_dir()]

    def list_agents(self) -> list:
        """List all saved agents.

        Returns:
            List of agent names
        """
        return [d.name for d in self.agents_dir.iterdir() if d.is_dir()]


# Global artifact manager instance
_artifact_manager: Optional[ArtifactManager] = None


def get_artifact_manager(base_dir: str = "artifacts") -> ArtifactManager:
    """Get or create global artifact manager.

    Args:
        base_dir: Base directory for artifacts

    Returns:
        ArtifactManager instance
    """
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager(base_dir)
    return _artifact_manager
