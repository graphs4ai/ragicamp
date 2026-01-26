"""Hybrid retriever combining dense and sparse search.

Hybrid retrieval combines the strengths of:
- Dense retrieval (semantic understanding, handles synonyms)
- Sparse retrieval (BM25, exact keyword matching, handles rare terms)

This is especially useful when documents contain:
- Technical jargon
- Part numbers, IDs, codes
- Specific terminology that embeddings might smooth over

The combination is done using Reciprocal Rank Fusion (RRF) which
is robust and doesn't require score calibration.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragicamp.core.logging import get_logger
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.sparse import SparseRetriever
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class HybridRetriever(Retriever):
    """Combines dense (semantic) and sparse (BM25) retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    retrievers. RRF is robust and doesn't require score normalization.

    RRF formula: score(d) = Σ 1/(k + rank(d))
    where k is a constant (typically 60) and rank is 1-indexed.
    """

    def __init__(
        self,
        name: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        alpha: float = 0.5,
        rrf_k: int = 60,
        **kwargs: Any,
    ):
        """Initialize hybrid retriever.

        Args:
            name: Retriever identifier
            embedding_model: Model for dense embeddings
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
                   Note: When using RRF, alpha is less important
            rrf_k: RRF constant (higher = more weight to lower ranks)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.embedding_model = embedding_model
        self.alpha = alpha
        self.rrf_k = rrf_k

        # Initialize component retrievers
        self.dense = DenseRetriever(
            name=f"{name}_dense",
            embedding_model=embedding_model,
        )
        self.sparse = SparseRetriever(name=f"{name}_sparse")

        self.documents: List[Document] = []

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents in both dense and sparse indices.

        Args:
            documents: List of documents to index
        """
        self.documents = documents

        logger.info("Indexing %d documents in hybrid retriever", len(documents))

        # Index in both retrievers
        self.dense.index_documents(documents)
        self.sparse.index_documents(documents)

        logger.info("Hybrid indexing complete")

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """Retrieve documents using hybrid search with RRF fusion.

        Args:
            query: Search query
            top_k: Number of documents to return
            **kwargs: Additional retrieval parameters

        Returns:
            List of top-k documents ranked by RRF score
        """
        if len(self.documents) == 0:
            return []

        # Retrieve more candidates from each retriever
        candidates_per_retriever = top_k * 3

        # Get results from both retrievers
        dense_results = self.dense.retrieve(query, top_k=candidates_per_retriever)
        sparse_results = self.sparse.retrieve(query, top_k=candidates_per_retriever)

        # Compute RRF scores
        rrf_scores = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Sort by RRF score and return top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            # Find the original document
            doc = self._get_doc_by_id(doc_id, dense_results, sparse_results)
            if doc:
                doc.score = score
                results.append(doc)

        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
    ) -> Dict[str, float]:
        """Compute RRF scores for documents from both retrievers.

        RRF formula: score(d) = Σ 1/(k + rank(d))

        Args:
            dense_results: Results from dense retriever (ranked)
            sparse_results: Results from sparse retriever (ranked)

        Returns:
            Dict mapping document ID to RRF score
        """
        rrf_scores: Dict[str, float] = {}

        # Add dense scores (with alpha weight)
        for rank, doc in enumerate(dense_results, start=1):
            score = self.alpha / (self.rrf_k + rank)
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + score

        # Add sparse scores (with 1-alpha weight)
        for rank, doc in enumerate(sparse_results, start=1):
            score = (1 - self.alpha) / (self.rrf_k + rank)
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + score

        return rrf_scores

    def _get_doc_by_id(
        self,
        doc_id: str,
        dense_results: List[Document],
        sparse_results: List[Document],
    ) -> Optional[Document]:
        """Find a document by ID from the result lists.

        Args:
            doc_id: Document ID to find
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results

        Returns:
            Document if found, None otherwise
        """
        # Check dense results first
        for doc in dense_results:
            if doc.id == doc_id:
                return doc

        # Check sparse results
        for doc in sparse_results:
            if doc.id == doc_id:
                return doc

        return None

    def save(self, artifact_name: str) -> str:
        """Save the hybrid retriever index.

        Args:
            artifact_name: Name for this retriever artifact

        Returns:
            Path where the artifact was saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Save component retrievers
        self.dense.save(f"{artifact_name}_dense")
        # Sparse retriever doesn't persist well, save documents instead

        # Save documents and config
        with open(artifact_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        config = {
            "name": self.name,
            "type": "hybrid",
            "embedding_model": self.embedding_model,
            "alpha": self.alpha,
            "rrf_k": self.rrf_k,
            "num_documents": len(self.documents),
        }
        manager.save_json(config, artifact_path / "config.json")

        logger.info("Saved hybrid retriever to: %s", artifact_path)
        return str(artifact_path)

    @classmethod
    def load(cls, artifact_name: str) -> "HybridRetriever":
        """Load a previously saved hybrid retriever.

        Args:
            artifact_name: Name of the retriever artifact to load

        Returns:
            Loaded HybridRetriever instance
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Create retriever
        retriever = cls(
            name=config["name"],
            embedding_model=config["embedding_model"],
            alpha=config["alpha"],
            rrf_k=config.get("rrf_k", 60),
        )

        # Load documents
        with open(artifact_path / "documents.pkl", "rb") as f:
            documents = pickle.load(f)

        # Reindex (sparse index can't be persisted easily)
        retriever.index_documents(documents)

        logger.info("Loaded hybrid retriever: %s (%d docs)", artifact_name, len(documents))
        return retriever
