"""Hierarchical retriever with parent-child chunk structure.

Hierarchical retrieval addresses a fundamental tradeoff in RAG:
- Small chunks: Better for precise semantic matching
- Large chunks: Better context for the LLM to generate answers

Solution: Search with small "child" chunks, return larger "parent" chunks.

This gives you:
- Precise retrieval (small chunks match better)
- Rich context (large chunks have more information)
"""

from typing import Any, List, Optional

from ragicamp.core.logging import get_logger
from ragicamp.indexes.hierarchical import HierarchicalIndex
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class HierarchicalRetriever(Retriever):
    """Retriever that searches child chunks but returns parent chunks.

    This is a thin strategy wrapper around a HierarchicalIndex.
    The index stores the parent/child documents and embeddings;
    the retriever implements the child-to-parent mapping strategy.
    """

    def __init__(
        self,
        name: str,
        index: Optional[HierarchicalIndex] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize hierarchical retriever.

        Args:
            name: Retriever identifier
            index: Pre-built HierarchicalIndex (preferred)
            embedding_model: Model name (used if building index)
            parent_chunk_size: Size of parent chunks
            child_chunk_size: Size of child chunks
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.index = index
        self.embedding_model_name = embedding_model
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def index_documents(self, documents: List[Document]) -> None:
        """Build index from documents.

        Note: Prefer building the index separately and passing to __init__.
        """
        if self.index is None:
            self.index = HierarchicalIndex(
                name=self.name,
                embedding_model=self.embedding_model_name,
                parent_chunk_size=self.parent_chunk_size,
                child_chunk_size=self.child_chunk_size,
            )
        self.index.build(documents)

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """Retrieve parent chunks by searching child chunks.

        Args:
            query: Search query
            top_k: Number of parent documents to return

        Returns:
            List of parent documents (with richer context)
        """
        if self.index is None or len(self.index) == 0:
            return []

        # Encode and search
        query_embedding = self.index.encode_query(query)
        hits = self.index.search(query_embedding, top_k=top_k)

        # Build result documents
        results = []
        for parent_idx, score, child_idx in hits:
            parent_doc = self.index.get_parent(parent_idx)
            child_doc = self.index.get_child(child_idx)

            if parent_doc:
                result = Document(
                    id=parent_doc.id,
                    text=parent_doc.text,
                    metadata={
                        **parent_doc.metadata,
                        "matched_child": child_doc.id if child_doc else None,
                        "matched_child_text": child_doc.text[:200] if child_doc else None,
                    },
                    score=score,
                )
                results.append(result)

        return results

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5, **kwargs: Any
    ) -> List[List[Document]]:
        """Retrieve parent chunks for multiple queries using batched encoding.

        This is significantly faster than calling retrieve() for each query:
        - Batch encodes all queries at once
        - Single FAISS search call for all queries

        Args:
            queries: List of query strings
            top_k: Number of parent documents to retrieve per query

        Returns:
            List of document lists, one per query
        """
        if self.index is None or len(self.index) == 0:
            return [[] for _ in queries]

        # Batch encode all queries at once (major speedup)
        query_embeddings = self.index.batch_encode_queries(queries)

        # Batch search (FAISS handles this efficiently)
        all_hits = self.index.batch_search(query_embeddings, top_k=top_k)

        # Build result documents for each query
        all_results = []
        for hits in all_hits:
            results = []
            for parent_idx, score, child_idx in hits:
                parent_doc = self.index.get_parent(parent_idx)
                child_doc = self.index.get_child(child_idx)

                if parent_doc:
                    result = Document(
                        id=parent_doc.id,
                        text=parent_doc.text,
                        metadata={
                            **parent_doc.metadata,
                            "matched_child": child_doc.id if child_doc else None,
                            "matched_child_text": child_doc.text[:200] if child_doc else None,
                        },
                        score=score,
                    )
                    results.append(result)

            all_results.append(results)

        return all_results

    @property
    def parent_docs(self) -> List[Document]:
        """Get parent documents (for backward compatibility)."""
        if self.index is None:
            return []
        return self.index.parent_docs

    @property
    def child_docs(self) -> List[Document]:
        """Get child documents (for backward compatibility)."""
        if self.index is None:
            return []
        return self.index.child_docs

    def save(self, artifact_name: str) -> str:
        """Save retriever config (index should be saved separately).

        Args:
            artifact_name: Name for this retriever artifact

        Returns:
            Path where saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        config = {
            "name": self.name,
            "type": "hierarchical",
            "hierarchical_index": self.index.name if self.index else None,
            "embedding_model": self.embedding_model_name,
            "parent_chunk_size": self.parent_chunk_size,
            "child_chunk_size": self.child_chunk_size,
            "num_parents": len(self.index.parent_docs) if self.index else 0,
            "num_children": len(self.index.child_docs) if self.index else 0,
        }
        manager.save_json(config, artifact_path / "config.json")

        logger.info("Saved hierarchical retriever config to: %s", artifact_path)
        return str(artifact_path)

    @classmethod
    def load(
        cls,
        artifact_name: str,
        index: Optional[HierarchicalIndex] = None,
    ) -> "HierarchicalRetriever":
        """Load a hierarchical retriever.

        Args:
            artifact_name: Name of the retriever artifact
            index: Optional pre-loaded index

        Returns:
            Loaded HierarchicalRetriever
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Load index if not provided
        if index is None:
            index_name = config.get("hierarchical_index")
            if index_name:
                logger.info("Loading hierarchical index: %s", index_name)
                index = HierarchicalIndex.load(index_name)
            else:
                # Legacy format - load from retriever path
                logger.info("Loading legacy hierarchical retriever")
                index = HierarchicalIndex.load(artifact_name, path=artifact_path)

        retriever = cls(
            name=config.get("name", artifact_name),
            index=index,
            embedding_model=config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
            parent_chunk_size=config.get("parent_chunk_size", 1024),
            child_chunk_size=config.get("child_chunk_size", 256),
        )

        logger.info(
            "Loaded hierarchical retriever: %s (%d parents, %d children)",
            artifact_name,
            len(index.parent_docs) if index else 0,
            len(index.child_docs) if index else 0,
        )
        return retriever
