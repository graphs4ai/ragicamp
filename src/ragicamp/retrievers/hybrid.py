"""Hybrid retriever combining dense and sparse search.

Hybrid retrieval combines the strengths of:
- Dense retrieval (semantic understanding, handles synonyms)
- Sparse retrieval (TF-IDF or BM25, exact keyword matching, handles rare terms)

This is especially useful when documents contain:
- Technical jargon
- Part numbers, IDs, codes
- Specific terminology that embeddings might smooth over

The combination is done using Reciprocal Rank Fusion (RRF) which
is robust and doesn't require score calibration.
"""

from typing import Any, Optional

from ragicamp.core.logging import get_logger
from ragicamp.indexes.embedding import EmbeddingIndex
from ragicamp.indexes.sparse import SparseIndex, SparseMethod, get_sparse_index_name
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class HybridRetriever(Retriever):
    """Combines dense (semantic) and sparse (TF-IDF/BM25) retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    retrievers. RRF is robust and doesn't require score normalization.

    RRF formula: score(d) = Σ 1/(k + rank(d))
    where k is a constant (typically 60) and rank is 1-indexed.

    Uses shared SparseIndex for efficiency - multiple hybrid retrievers
    with different alphas share the same sparse index.
    """

    def __init__(
        self,
        name: str,
        index: Optional[EmbeddingIndex] = None,
        sparse_index: Optional[SparseIndex] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        sparse_method: str | SparseMethod = SparseMethod.TFIDF,
        alpha: float = 0.5,
        rrf_k: int = 60,
        **kwargs: Any,
    ):
        """Initialize hybrid retriever.

        Args:
            name: Retriever identifier
            index: Pre-built EmbeddingIndex to use (preferred)
            sparse_index: Pre-built SparseIndex to use (preferred)
            embedding_model: Model name (used if building index)
            sparse_method: Sparse method ('tfidf' or 'bm25')
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            rrf_k: RRF constant (higher = more weight to lower ranks)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.index = index
        self._sparse_index = sparse_index
        self.embedding_model_name = embedding_model
        self.sparse_method = (
            SparseMethod(sparse_method) if isinstance(sparse_method, str) else sparse_method
        )
        self.alpha = alpha
        self.rrf_k = rrf_k

    @property
    def sparse(self) -> SparseIndex:
        """Get or create sparse index."""
        if self._sparse_index is None:
            # Build sparse index on demand if not provided
            sparse_name = get_sparse_index_name(
                self.index.name if self.index else self.name,
                self.sparse_method.value,
            )
            self._sparse_index = SparseIndex(
                name=sparse_name,
                method=self.sparse_method,
            )
            if self.index and len(self.index) > 0:
                logger.info(
                    "Building sparse index (%s) for hybrid retriever: %s",
                    self.sparse_method.value,
                    self.name,
                )
                self._sparse_index.build(self.index.documents)
                logger.info("Sparse index ready for hybrid retriever: %s", self.name)
        return self._sparse_index

    @property
    def documents(self) -> list[Document]:
        """Get documents from index."""
        if self.index is None:
            return []
        return self.index.documents

    def index_documents(self, documents: list[Document]) -> None:
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

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[Document]:
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
                result = Document(
                    id=doc.id, text=doc.text, metadata=doc.metadata.copy(), score=score
                )
                dense_results.append(result)

        # Sparse search (using SparseIndex)
        sparse_hits = self.sparse.search(query, top_k=candidates)
        sparse_results = []
        for idx, score in sparse_hits:
            doc = self.index.get_document(idx)
            if doc:
                result = Document(
                    id=doc.id, text=doc.text, metadata=doc.metadata.copy(), score=score
                )
                sparse_results.append(result)

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
        self, queries: list[str], top_k: int = 5, **kwargs: Any
    ) -> list[list[Document]]:
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

        # Batch sparse search (using SparseIndex)
        all_sparse_hits = self.sparse.batch_search(queries, top_k=candidates)

        # Process each query - combine dense and sparse results
        all_results = []
        for i, _query in enumerate(queries):
            # Convert dense hits to documents
            dense_results = []
            for idx, score in all_dense_hits[i]:
                doc = self.index.get_document(idx)
                if doc:
                    result = Document(
                        id=doc.id, text=doc.text, metadata=doc.metadata.copy(), score=score
                    )
                    dense_results.append(result)

            # Convert sparse hits to documents
            sparse_results = []
            for idx, score in all_sparse_hits[i]:
                doc = self.index.get_document(idx)
                if doc:
                    result = Document(
                        id=doc.id, text=doc.text, metadata=doc.metadata.copy(), score=score
                    )
                    sparse_results.append(result)

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
        dense_results: list[Document],
        sparse_results: list[Document],
    ) -> dict[str, float]:
        """Compute RRF scores.

        RRF formula: score(d) = Σ 1/(k + rank(d))
        """
        rrf_scores: dict[str, float] = {}

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
        dense_results: list[Document],
        sparse_results: list[Document],
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
        """Save retriever config (sparse index is saved separately).

        Args:
            artifact_name: Name for this retriever artifact

        Returns:
            Path where saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Get sparse index name for reference
        sparse_index_name = None
        if self._sparse_index is not None:
            sparse_index_name = self._sparse_index.name
        elif self.index:
            sparse_index_name = get_sparse_index_name(
                self.index.name, self.sparse_method.value
            )

        config = {
            "name": self.name,
            "type": "hybrid",
            "embedding_index": self.index.name if self.index else None,
            "sparse_index": sparse_index_name,
            "embedding_model": self.embedding_model_name,
            "sparse_method": self.sparse_method.value,
            "alpha": self.alpha,
            "rrf_k": self.rrf_k,
            "num_documents": len(self.index) if self.index else 0,
        }

        # Save sparse index if built and not yet saved
        if self._sparse_index is not None:
            sparse_path = manager.get_sparse_index_path(self._sparse_index.name)
            if not (sparse_path / "config.json").exists():
                logger.info("Saving shared sparse index: %s", self._sparse_index.name)
                self._sparse_index.save()

        manager.save_json(config, artifact_path / "config.json")

        logger.info("Saved hybrid retriever config to: %s", artifact_path)
        return str(artifact_path)

    @classmethod
    def load(
        cls,
        artifact_name: str,
        index: Optional[EmbeddingIndex] = None,
        sparse_index: Optional[SparseIndex] = None,
    ) -> "HybridRetriever":
        """Load a hybrid retriever.

        Args:
            artifact_name: Name of the retriever artifact
            index: Optional pre-loaded embedding index
            sparse_index: Optional pre-loaded sparse index (for sharing)

        Returns:
            Loaded HybridRetriever
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Load embedding index if not provided
        if index is None:
            index_name = config.get("embedding_index")
            if index_name:
                logger.info("Loading embedding index: %s", index_name)
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

        sparse_method = config.get("sparse_method", "tfidf")

        # Load sparse index if not provided
        if sparse_index is None:
            sparse_index_name = config.get("sparse_index")
            if sparse_index_name and manager.sparse_index_exists(sparse_index_name):
                logger.info("Loading shared sparse index: %s", sparse_index_name)
                sparse_index = SparseIndex.load(
                    sparse_index_name,
                    documents=index.documents if index else None,
                )
            # Check for legacy format (sparse stored with retriever)
            elif (artifact_path / "sparse_vectorizer.pkl").exists():
                import pickle

                logger.info("Loading legacy sparse index from retriever artifact")
                sparse_index = SparseIndex(
                    name=f"{artifact_name}_sparse",
                    method="tfidf",
                )
                with open(artifact_path / "sparse_vectorizer.pkl", "rb") as f:
                    sparse_index._vectorizer = pickle.load(f)
                with open(artifact_path / "sparse_matrix.pkl", "rb") as f:
                    sparse_index._doc_vectors = pickle.load(f)
                if index:
                    sparse_index.documents = index.documents

        retriever = cls(
            name=config["name"],
            index=index,
            sparse_index=sparse_index,
            embedding_model=config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
            sparse_method=sparse_method,
            alpha=config.get("alpha", 0.5),
            rrf_k=config.get("rrf_k", 60),
        )

        logger.info(
            "Loaded hybrid retriever: %s (%s, alpha=%.2f, %d docs)",
            artifact_name,
            sparse_method,
            retriever.alpha,
            len(index) if index else 0,
        )
        return retriever
