"""Reranker provider with lazy loading and lifecycle management."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from ragicamp.core.logging import get_logger
from ragicamp.utils.resource_manager import ResourceManager

from .base import ModelProvider

logger = get_logger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for reranker."""

    model_name: str = "bge"  # bge, bge-base, ms-marco, or HF path
    batch_size: int = 32


class RerankerProvider(ModelProvider):
    """Provides reranker with lazy loading and proper cleanup.

    Usage:
        provider = RerankerProvider(RerankerConfig("bge"))

        with provider.load() as reranker:
            reranked = reranker.rerank(query, documents, top_k=5)
        # Reranker unloaded, GPU memory freed
    """

    MODELS = {
        "bge": "BAAI/bge-reranker-large",
        "bge-base": "BAAI/bge-reranker-base",
        "bge-v2": "BAAI/bge-reranker-v2-m3",
        "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-large": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    }

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._reranker = None

    @property
    def model_name(self) -> str:
        return self.MODELS.get(self.config.model_name, self.config.model_name)

    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator["RerankerWrapper"]:
        """Load reranker, yield it, then unload."""
        import torch
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker: %s", self.model_name)

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = CrossEncoder(
                self.model_name,
                device=device,
                trust_remote_code=True,
            )

            self._reranker = RerankerWrapper(model, self.config.batch_size)
            yield self._reranker

        finally:
            self._unload()

    def _unload(self):
        """Unload reranker and free GPU memory."""
        if self._reranker is not None:
            self._reranker.unload()
            self._reranker = None

        ResourceManager.clear_gpu_memory()
        logger.info("Reranker unloaded: %s", self.model_name)


class RerankerWrapper:
    """Wrapper around CrossEncoder implementing reranker interface."""

    def __init__(self, model, batch_size: int = 32):
        self._model = model
        self._batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int,
    ) -> list:
        """Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of Document objects
            top_k: Number to return

        Returns:
            Top-k documents sorted by reranker score
        """
        if not documents:
            return []

        # Create pairs
        pairs = [(query, doc.text) for doc in documents]

        # Score
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        return sorted_docs[:top_k]

    def batch_rerank(
        self,
        queries: list[str],
        documents_list: list[list],
        top_k: int,
    ) -> list[list]:
        """Batch rerank for multiple queries.

        Args:
            queries: List of search queries
            documents_list: List of document lists
            top_k: Number to return per query

        Returns:
            List of top-k document lists
        """
        if not queries:
            return []

        # Build all pairs
        all_pairs = []
        pair_indices = []

        for q_idx, (query, docs) in enumerate(zip(queries, documents_list)):
            for d_idx, doc in enumerate(docs):
                all_pairs.append((query, doc.text))
                pair_indices.append((q_idx, d_idx))

        if not all_pairs:
            return [[] for _ in queries]

        # Score all at once
        scores = self._model.predict(
            all_pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Assign scores
        for (q_idx, d_idx), score in zip(pair_indices, scores):
            documents_list[q_idx][d_idx].score = float(score)

        # Sort and return
        results = []
        for docs in documents_list:
            sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
            results.append(sorted_docs[:top_k])

        return results

    def unload(self):
        """Unload model."""
        import gc
        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
