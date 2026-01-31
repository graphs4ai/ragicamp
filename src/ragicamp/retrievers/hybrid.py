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

from typing import Any, Dict, List, Optional

from ragicamp.core.logging import get_logger
from ragicamp.indexes.embedding import EmbeddingIndex
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.retrievers.sparse import SparseRetriever
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class HybridRetriever(Retriever):
    """Combines dense (semantic) and sparse (BM25) retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    retrievers. RRF is robust and doesn't require score normalization.

    RRF formula: score(d) = Σ 1/(k + rank(d))
    where k is a constant (typically 60) and rank is 1-indexed.

    This is a thin wrapper around an EmbeddingIndex + BM25.
    """

    def __init__(
        self,
        name: str,
        index: Optional[EmbeddingIndex] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        alpha: float = 0.5,
        rrf_k: int = 60,
        **kwargs: Any,
    ):
        """Initialize hybrid retriever.

        Args:
            name: Retriever identifier
            index: Pre-built EmbeddingIndex to use (preferred)
            embedding_model: Model name (used if building index)
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            rrf_k: RRF constant (higher = more weight to lower ranks)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.index = index
        self.embedding_model_name = embedding_model
        self.alpha = alpha
        self.rrf_k = rrf_k

        # Sparse retriever (BM25) - built on demand
        self._sparse: Optional[SparseRetriever] = None

    @property
    def sparse(self) -> SparseRetriever:
        """Get or create sparse retriever."""
        if self._sparse is None:
            self._sparse = SparseRetriever(name=f"{self.name}_sparse")
            # Index with current documents if available
            if self.index and len(self.index) > 0:
                self._sparse.index_documents(self.index.documents)
        return self._sparse

    @property
    def documents(self) -> List[Document]:
        """Get documents from index."""
        if self.index is None:
            return []
        return self.index.documents

    def index_documents(self, documents: List[Document]) -> None:
        """Build index from documents.

        Note: Prefer building the index separately and passing to __init__.
        """
        if self.index is None:
            self.index = EmbeddingIndex(
                name=self.name,
                embedding_model=self.embedding_model_name,
            )
        self.index.build(documents)

        # Reset sparse to rebuild
        self._sparse = None

        logger.info("Hybrid indexing complete: %d documents", len(documents))

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """Retrieve documents using hybrid search with RRF fusion.

        Args:
            query: Search query
            top_k: Number of documents to return
            **kwargs: Additional retrieval parameters

        Returns:
            List of top-k documents ranked by RRF score
        """
        if self.index is None or len(self.index) == 0:
            return []

        # Retrieve more candidates from each method
        candidates = top_k * 3

        # Dense search
        query_embedding = self.index.encode_query(query)
        dense_hits = self.index.search(query_embedding, top_k=candidates)

        # Convert to documents for RRF
        dense_results = []
        for idx, score in dense_hits:
            doc = self.index.get_document(idx)
            if doc:
                result = Document(id=doc.id, text=doc.text, metadata=doc.metadata.copy(), score=score)
                dense_results.append(result)

        # Sparse search
        sparse_results = self.sparse.retrieve(query, top_k=candidates)

        # Compute RRF scores
        rrf_scores = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Sort by RRF score and return top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self._get_doc_by_id(doc_id, dense_results, sparse_results)
            if doc:
                doc.score = score
                results.append(doc)

        return results

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5, **kwargs: Any
    ) -> List[List[Document]]:
        """Retrieve documents for multiple queries using fully batched operations.

        Both dense and sparse retrieval are batched for maximum speed.

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query

        Returns:
            List of document lists, one per query
        """
        if self.index is None or len(self.index) == 0:
            return [[] for _ in queries]

        candidates = top_k * 3

        # Batch encode all queries at once (major speedup)
        query_embeddings = self.index.batch_encode_queries(queries)

        # Batch search for dense results
        all_dense_hits = self.index.batch_search(query_embeddings, top_k=candidates)

        # Batch sparse search (also batched now!)
        all_sparse_results = self.sparse.batch_retrieve(queries, top_k=candidates)

        # Process each query - combine dense and sparse results
        all_results = []
        for i, query in enumerate(queries):
            # Convert dense hits to documents
            dense_results = []
            for idx, score in all_dense_hits[i]:
                doc = self.index.get_document(idx)
                if doc:
                    result = Document(
                        id=doc.id, text=doc.text, metadata=doc.metadata.copy(), score=score
                    )
                    dense_results.append(result)

            sparse_results = all_sparse_results[i]

            # Compute RRF scores
            rrf_scores = self._reciprocal_rank_fusion(dense_results, sparse_results)

            # Sort by RRF score and return top-k
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            results = []
            for doc_id, score in sorted_docs[:top_k]:
                doc = self._get_doc_by_id(doc_id, dense_results, sparse_results)
                if doc:
                    doc.score = score
                    results.append(doc)

            all_results.append(results)

        return all_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
    ) -> Dict[str, float]:
        """Compute RRF scores.

        RRF formula: score(d) = Σ 1/(k + rank(d))
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
        """Find a document by ID."""
        for doc in dense_results:
            if doc.id == doc_id:
                return doc
        for doc in sparse_results:
            if doc.id == doc_id:
                return doc
        return None

    def save(self, artifact_name: str) -> str:
        """Save retriever config and sparse index if built.

        Args:
            artifact_name: Name for this retriever artifact

        Returns:
            Path where saved
        """
        import pickle

        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Check if sparse index is built
        has_sparse = self._sparse is not None and self._sparse.doc_vectors is not None

        config = {
            "name": self.name,
            "type": "hybrid",
            "embedding_index": self.index.name if self.index else None,
            "embedding_model": self.embedding_model_name,
            "alpha": self.alpha,
            "rrf_k": self.rrf_k,
            "num_documents": len(self.index) if self.index else 0,
            "has_sparse": has_sparse,
        }

        # Save sparse index components if built
        if has_sparse:
            logger.info("Saving sparse index components for: %s", artifact_name)
            with open(artifact_path / "sparse_vectorizer.pkl", "wb") as f:
                pickle.dump(self._sparse.vectorizer, f)
            with open(artifact_path / "sparse_matrix.pkl", "wb") as f:
                pickle.dump(self._sparse.doc_vectors, f)

        manager.save_json(config, artifact_path / "config.json")

        logger.info("Saved hybrid retriever config to: %s", artifact_path)
        return str(artifact_path)

    @classmethod
    def load(
        cls,
        artifact_name: str,
        index: Optional[EmbeddingIndex] = None,
    ) -> "HybridRetriever":
        """Load a hybrid retriever.

        Args:
            artifact_name: Name of the retriever artifact
            index: Optional pre-loaded index

        Returns:
            Loaded HybridRetriever
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
                # Legacy format - need to rebuild from documents.pkl
                import pickle
                docs_path = artifact_path / "documents.pkl"
                if docs_path.exists():
                    logger.info("Loading legacy hybrid retriever")
                    with open(docs_path, "rb") as f:
                        documents = pickle.load(f)
                    index = EmbeddingIndex(
                        name=artifact_name,
                        embedding_model=config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                    )
                    index.build(documents)

        retriever = cls(
            name=config["name"],
            index=index,
            embedding_model=config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
            alpha=config.get("alpha", 0.5),
            rrf_k=config.get("rrf_k", 60),
        )

        # Load pre-built sparse index if available
        if config.get("has_sparse"):
            sparse_vectorizer_path = artifact_path / "sparse_vectorizer.pkl"
            sparse_matrix_path = artifact_path / "sparse_matrix.pkl"
            
            if sparse_vectorizer_path.exists() and sparse_matrix_path.exists():
                import pickle
                
                logger.info("Loading pre-built sparse index for: %s", artifact_name)
                sparse = SparseRetriever(name=f"{artifact_name}_sparse")
                
                with open(sparse_vectorizer_path, "rb") as f:
                    sparse.vectorizer = pickle.load(f)
                with open(sparse_matrix_path, "rb") as f:
                    sparse.doc_vectors = pickle.load(f)
                
                # Link documents from the dense index
                if index:
                    sparse.documents = index.documents
                
                retriever._sparse = sparse
                logger.info("Loaded sparse index: %d documents", len(sparse.documents))
            else:
                logger.warning(
                    "Sparse index files not found for %s, will rebuild on first query",
                    artifact_name,
                )

        logger.info("Loaded hybrid retriever: %s (%d docs)", artifact_name, len(index) if index else 0)
        return retriever
