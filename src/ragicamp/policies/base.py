"""Base class for decision policies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Policy(ABC):
    """Base class for decision policies.

    Policies select actions (e.g., RAG parameters, retrieval strategies)
    and can be updated based on rewards (for learning).
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the policy.

        Args:
            name: Policy identifier
            **kwargs: Policy-specific configuration
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def select_action(self, **context: Any) -> Dict[str, Any]:
        """Select an action based on context.

        Args:
            **context: Context information (query, state, etc.)

        Returns:
            Selected action as a dictionary
        """
        pass

    @abstractmethod
    def update(self, **feedback: Any) -> None:
        """Update policy based on feedback.

        Args:
            **feedback: Feedback information (reward, trajectory, etc.)
        """
        pass

    def save(self, path: str) -> None:
        """Save policy to disk.

        Args:
            path: Path to save policy
        """
        pass

    def load(self, path: str) -> None:
        """Load policy from disk.

        Args:
            path: Path to load policy from
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
