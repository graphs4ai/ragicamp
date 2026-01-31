"""Base class for indexes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ragicamp.retrievers.base import Document


class Index(ABC):
    """Base class for all document indexes.

    An index stores documents and their representations (embeddings, etc.)
    for efficient retrieval. Indexes are expensive to build but reusable
    across different retriever strategies.

    Indexes are responsible for:
    - Storing documents and their embeddings
    - Building search structures (FAISS, BM25, etc.)
    - Saving/loading to disk

    Retrievers are responsible for:
    - Search strategy (dense, hybrid, hierarchical)
    - Query processing
    - Result ranking
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the index.

        Args:
            name: Index identifier (used for saving/loading)
            **kwargs: Index-specific configuration
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def build(self, documents: list[Document]) -> None:
        """Build the index from documents.

        Args:
            documents: List of documents to index
        """
        pass

    @abstractmethod
    def search(self, query_embedding: Any, top_k: int = 10) -> list[tuple]:
        """Search the index with a query embedding.

        Args:
            query_embedding: Query vector/embedding
            top_k: Number of results to return

        Returns:
            List of (document_idx, score) tuples
        """
        pass

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> Path:
        """Save the index to disk.

        Args:
            path: Optional custom path, otherwise uses default artifact location

        Returns:
            Path where index was saved
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, name: str, path: Optional[Path] = None) -> "Index":
        """Load an index from disk.

        Args:
            name: Index name
            path: Optional custom path, otherwise uses default artifact location

        Returns:
            Loaded index instance
        """
        pass

    def get_document(self, idx: int) -> Optional[Document]:
        """Get a document by index.

        Args:
            idx: Document index

        Returns:
            Document if found, None otherwise
        """
        raise NotImplementedError("Subclass must implement get_document")

    def __len__(self) -> int:
        """Return number of indexed documents."""
        raise NotImplementedError("Subclass must implement __len__")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
