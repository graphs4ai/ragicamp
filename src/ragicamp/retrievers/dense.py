"""Dense retriever using vector similarity."""

from typing import Any, Optional

from ragicamp.core.logging import get_logger
from ragicamp.indexes.embedding import EmbeddingIndex
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class DenseRetriever(Retriever):
    """Dense retriever using neural embeddings and vector similarity.

    This is a thin strategy wrapper around an EmbeddingIndex.
    The index stores the documents and embeddings; the retriever
    just implements the search strategy.
    """

    def __init__(
        self,
        name: str,
        index: Optional[EmbeddingIndex] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ):
        """Initialize dense retriever.

        Args:
            name: Retriever identifier
            index: Pre-built EmbeddingIndex to use (preferred)
            embedding_model: Model name (used if index not provided)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.index = index
        self.embedding_model_name = embedding_model

    def index_documents(self, documents: list[Document]) -> None:
        """Build index from documents.

        Note: Prefer building the index separately and passing it to __init__.
        This method exists for backward compatibility.
        """
        if self.index is None:
            self.index = EmbeddingIndex(
                name=self.name,
                embedding_model=self.embedding_model_name,
            )
        self.index.build(documents)

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[Document]:
        """Retrieve documents using dense similarity search."""
        if self.index is None or len(self.index) == 0:
            return []

        # Encode and search
        query_embedding = self.index.encode_query(query)
        results = self.index.search(query_embedding, top_k=top_k)

        # Build result documents
        docs = []
        for idx, score in results:
            doc = self.index.get_document(idx)
            if doc:
                # Create copy with score
                result = Document(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata.copy(),
                    score=score,
                )
                docs.append(result)

        return docs

    def batch_retrieve(
        self, queries: list[str], top_k: int = 5, **kwargs: Any
    ) -> list[list[Document]]:
        """Retrieve documents for multiple queries using batched encoding and search.

        This is significantly faster than calling retrieve() for each query:
        - Batch encodes all queries at once
        - Single FAISS search call for all queries

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query

        Returns:
            List of document lists, one per query
        """
        if self.index is None or len(self.index) == 0:
            return [[] for _ in queries]

        # Batch encode all queries at once (major speedup)
        query_embeddings = self.index.batch_encode_queries(queries)

        # Batch search (FAISS handles this efficiently)
        all_results = self.index.batch_search(query_embeddings, top_k=top_k)

        # Build result documents for each query
        all_docs = []
        for results in all_results:
            docs = []
            for idx, score in results:
                doc = self.index.get_document(idx)
                if doc:
                    result = Document(
                        id=doc.id,
                        text=doc.text,
                        metadata=doc.metadata.copy(),
                        score=score,
                    )
                    docs.append(result)
            all_docs.append(docs)

        return all_docs

    @property
    def documents(self) -> list[Document]:
        """Get documents from index (for backward compatibility)."""
        if self.index is None:
            return []
        return self.index.documents

    def save(self, artifact_name: str) -> str:
        """Save retriever config (index should be saved separately).

        Args:
            artifact_name: Name for this retriever artifact

        Returns:
            Path where saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Save config only - references the index
        config = {
            "name": self.name,
            "type": "dense",
            "embedding_index": self.index.name if self.index else None,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.index.embedding_dim if self.index else None,
            "num_documents": len(self.index) if self.index else 0,
            "index_type": self.index.index_type if self.index else "flat",
        }
        manager.save_json(config, artifact_path / "config.json")

        logger.info("Saved retriever config to: %s", artifact_path)
        return str(artifact_path)

    save_index = save  # Alias

    @classmethod
    def load(
        cls,
        artifact_name: str,
        index: Optional[EmbeddingIndex] = None,
        embedding_model: Optional[str] = None,
    ) -> "DenseRetriever":
        """Load a retriever.

        Args:
            artifact_name: Name of the retriever artifact
            index: Optional pre-loaded index (avoids reloading)
            embedding_model: Optional override for embedding model

        Returns:
            Loaded DenseRetriever
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Load index if not provided
        if index is None:
            index_name = config.get("embedding_index")
            if index_name:
                logger.info("Loading index: %s", index_name)
                index = EmbeddingIndex.load(index_name)
            else:
                # Legacy: standalone retriever with files in retriever path
                logger.info("Loading legacy standalone index")
                index = EmbeddingIndex.load(artifact_name, path=artifact_path)

        # Use config model if not overridden
        if embedding_model is None:
            embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")

        retriever = cls(
            name=config.get("name", artifact_name),
            index=index,
            embedding_model=embedding_model,
        )

        logger.info("Loaded retriever: %s (%d docs)", artifact_name, len(index))
        return retriever

    load_index = load  # Alias
