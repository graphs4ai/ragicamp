"""Cross-encoder reranker for high-accuracy document ranking.

Cross-encoders process (query, document) pairs jointly through a transformer,
enabling rich interaction between query and document tokens. This is more
accurate than bi-encoders but slower (can't pre-compute document embeddings).

Use cross-encoders as a second stage after initial retrieval:
1. Retrieve 20-50 candidates with fast bi-encoder/BM25
2. Rerank top candidates with cross-encoder
3. Return top-k reranked results
"""

from typing import TYPE_CHECKING, Optional

from ragicamp.core.logging import get_logger
from ragicamp.rag.rerankers.base import Reranker

if TYPE_CHECKING:
    from ragicamp.retrievers.base import Document

logger = get_logger(__name__)


class CrossEncoderReranker(Reranker):
    """Rerank documents using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders because they
    process query and document together, allowing full attention
    between all tokens. The tradeoff is speed - they can't use
    pre-computed embeddings.

    Supported models:
    - "bge": BAAI/bge-reranker-large (recommended, best accuracy)
    - "bge-base": BAAI/bge-reranker-base (faster, good accuracy)
    - "ms-marco": cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, English)
    - "ms-marco-large": cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
    """

    MODELS = {
        "bge": "BAAI/bge-reranker-large",
        "bge-base": "BAAI/bge-reranker-base",
        "bge-v2": "BAAI/bge-reranker-v2-m3",
        "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-large": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    }

    def __init__(
        self,
        model_name: str = "bge",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: Model identifier (key from MODELS dict or HuggingFace path)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for scoring
        """
        import torch
        from sentence_transformers import CrossEncoder

        # Resolve model name
        self.model_id = self.MODELS.get(model_name, model_name)
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.batch_size = batch_size

        logger.info("Loading cross-encoder: %s on %s", self.model_id, self.device)
        self.model = CrossEncoder(
            self.model_id,
            device=self.device,
            trust_remote_code=True,
            automodel_args={"attn_implementation": "flash_attention_2"},
        )

        # Apply torch.compile() for additional speedup (PyTorch 2.0+)
        try:
            if hasattr(torch, "compile") and torch.cuda.is_available():
                self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
                logger.info("Applied torch.compile() to cross-encoder model")
        except Exception as e:
            logger.debug("torch.compile() not applied: %s", e)

    def rerank(
        self,
        query: str,
        documents: list["Document"],
        top_k: int,
    ) -> list["Document"]:
        """Rerank documents using cross-encoder scores.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            Top-k documents sorted by cross-encoder relevance score
        """
        if not documents:
            return []

        # Create (query, document) pairs for scoring
        pairs = [(query, doc.text) for doc in documents]

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        # Sort by score (descending) and return top_k
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        return sorted_docs[:top_k]

    def batch_rerank(
        self,
        queries: list[str],
        documents_list: list[list["Document"]],
        top_k: int,
    ) -> list[list["Document"]]:
        """Rerank documents for multiple queries more efficiently.

        Batches all (query, document) pairs together for faster scoring.

        Args:
            queries: List of search queries
            documents_list: List of document lists, one per query
            top_k: Number of top documents to return per query

        Returns:
            List of top-k reranked document lists
        """
        if not queries:
            return []

        # Build all pairs at once with indices to track which query each belongs to
        all_pairs = []
        pair_indices = []  # (query_idx, doc_idx_within_query)

        for q_idx, (query, docs) in enumerate(zip(queries, documents_list)):
            for d_idx, doc in enumerate(docs):
                all_pairs.append((query, doc.text))
                pair_indices.append((q_idx, d_idx))

        if not all_pairs:
            return [[] for _ in queries]

        # Score all pairs in one batch
        scores = self.model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Assign scores back to documents
        for (q_idx, d_idx), score in zip(pair_indices, scores):
            documents_list[q_idx][d_idx].score = float(score)

        # Sort and return top_k for each query
        results = []
        for docs in documents_list:
            sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
            results.append(sorted_docs[:top_k])

        return results

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model='{self.model_name}', device='{self.device}')"
