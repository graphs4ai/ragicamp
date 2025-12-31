"""Base classes for RAG agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ragicamp.retrievers.base import Document
    from ragicamp.core.schemas import RAGResponseMeta


@dataclass
class RAGContext:
    """Context information for a RAG query.

    Attributes:
        query: The input question/query
        retrieved_docs: Documents retrieved during the process
        intermediate_steps: Log of actions taken (for MDP agents)
        metadata: Additional context information
    """

    query: str
    retrieved_docs: List["Document"] = field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Response from a RAG agent.

    Attributes:
        answer: The generated answer
        context: The RAG context used
        prompt: The full prompt sent to the model (for debugging/analysis)
        confidence: Optional confidence score
        metadata: Response metadata - can be Dict or typed RAGResponseMeta.
                  Use metadata_dict property for consistent dict access.
    """

    answer: str
    context: RAGContext
    prompt: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Union[Dict[str, Any], "RAGResponseMeta"] = field(default_factory=dict)

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as a dict (for backwards compatibility).
        
        Works whether metadata is a dict or RAGResponseMeta.
        """
        if isinstance(self.metadata, dict):
            return self.metadata
        # RAGResponseMeta has to_dict()
        return self.metadata.to_dict() if hasattr(self.metadata, "to_dict") else {}


class RAGAgent(ABC):
    """Base class for all RAG agents.

    A RAG agent takes a query and produces an answer, potentially using
    retrieval and other intermediate steps. Different agents implement
    different strategies (direct LLM, fixed RAG, bandit-based, MDP-based).
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the agent.

        Args:
            name: Agent identifier
            **kwargs: Agent-specific configuration
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer for the given query.

        Args:
            query: The input question
            **kwargs: Additional arguments specific to the agent

        Returns:
            RAGResponse containing the answer and context
        """
        pass

    def batch_answer(self, queries: List[str], **kwargs: Any) -> List[RAGResponse]:
        """Generate answers for multiple queries (batch processing).

        Default implementation: loops through queries one by one.
        Subclasses can override for true parallel/batch processing.

        Args:
            queries: List of input questions
            **kwargs: Additional arguments specific to the agent

        Returns:
            List of RAGResponse objects, one per query
        """
        return [self.answer(query, **kwargs) for query in queries]

    def reset(self) -> None:
        """Reset agent state (useful for stateful agents)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
