"""Cross-encoder reranker for high-accuracy document ranking.

Cross-encoders process (query, document) pairs jointly through a transformer,
enabling rich interaction between query and document tokens. This is more
accurate than bi-encoders but slower (can't pre-compute document embeddings).

Use cross-encoders as a second stage after initial retrieval:
1. Retrieve 20-50 candidates with fast bi-encoder/BM25
2. Rerank top candidates with cross-encoder
3. Return top-k reranked results
"""

from typing import TYPE_CHECKING, List, Optional

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
        from sentence_transformers import CrossEncoder
        import torch

        # Resolve model name
        self.model_id = self.MODELS.get(model_name, model_name)
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.batch_size = batch_size

        logger.info("Loading cross-encoder: %s on %s", self.model_id, self.device)
        self.model = CrossEncoder(self.model_id, device=self.device)

    def rerank(
        self,
        query: str,
        documents: List["Document"],
        top_k: int,
    ) -> List["Document"]:
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

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model='{self.model_name}', device='{self.device}')"
