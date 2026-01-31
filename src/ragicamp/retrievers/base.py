"""Base class for retrievers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Document:
    """Represents a document in the retrieval system.

    Attributes:
        id: Unique document identifier
        text: Document text content
        metadata: Additional document metadata (source, title, etc.)
        score: Retrieval score (relevance to query)
    """

    id: str
    text: str
    metadata: dict[str, Any]
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "text": self.text,
            "content": self.text,  # Alias for compatibility
            "metadata": self.metadata,
            "score": self.score,
        }


class Retriever(ABC):
    """Base class for all document retrievers.

    A retriever takes a query and returns relevant documents from a corpus.
    Different implementations can use sparse (BM25), dense (embeddings),
    or hybrid retrieval methods.
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the retriever.

        Args:
            name: Retriever identifier
            **kwargs: Retriever-specific configuration
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query
            top_k: Number of documents to retrieve
            **kwargs: Additional retrieval parameters

        Returns:
            List of retrieved Document objects
        """
        pass

    @abstractmethod
    def index_documents(self, documents: list[Document]) -> None:
        """Index a collection of documents for retrieval.

        Args:
            documents: List of documents to index
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
