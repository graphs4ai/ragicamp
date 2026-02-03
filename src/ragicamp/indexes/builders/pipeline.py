"""Pipelined processing with GPU/CPU overlap.

Overlaps GPU embedding with CPU post-processing (normalize, index, save)
to improve throughput by ~30-40%.

Usage:
    pipeline = EmbeddingPipeline(encoder, on_batch_ready)
    for texts, chunks in batches:
        pipeline.submit(texts, chunks)
    pipeline.finish()
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


class Encoder(Protocol):
    """Protocol for embedding encoders (SentenceTransformer or VLLMEmbedder)."""

    def encode(
        self,
        sentences: list[str],
        show_progress_bar: bool = True,
        batch_size: int = 256,
        **kwargs,
    ) -> np.ndarray: ...


@dataclass
class EmbeddingResult:
    """Result of an embedding batch."""

    embeddings: np.ndarray
    chunks: list[Any]
    batch_num: int


class EmbeddingPipeline:
    """Overlaps GPU embedding with CPU post-processing.

    While GPU encodes batch N, CPU processes results from batch N-1.
    This hides the CPU latency (normalize, index.add, save) behind GPU work.
    """

    def __init__(
        self,
        encoder: Encoder,
        process_fn: Callable[[EmbeddingResult], None],
        embedding_batch_size: int = 4096,
        normalize: bool = True,
    ):
        """Initialize pipeline.

        Args:
            encoder: Embedding encoder (VLLMEmbedder or SentenceTransformer)
            process_fn: Callback to process each batch result (receives EmbeddingResult)
            embedding_batch_size: Batch size for encoder
            normalize: Whether to L2-normalize embeddings
        """
        self.encoder = encoder
        self.process_fn = process_fn
        self.embedding_batch_size = embedding_batch_size
        self.normalize = normalize

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
        self._pending: tuple[Future, list[Any], int] | None = None
        self._batch_num = 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings (runs in background thread)."""
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.embedding_batch_size,
        )

        if self.normalize:
            # In-place normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.divide(embeddings, norms, out=embeddings)

        return embeddings

    def submit(self, texts: list[str], chunks: list[Any]) -> None:
        """Submit a batch for embedding.

        If there's a pending batch, processes it while the new batch encodes.

        Args:
            texts: List of text strings to embed
            chunks: Corresponding chunk objects (passed through to process_fn)
        """
        self._batch_num += 1

        # Submit current batch to GPU (background thread)
        future = self._executor.submit(self._encode, texts)

        # While GPU works on current batch, process previous batch
        if self._pending is not None:
            prev_future, prev_chunks, prev_batch_num = self._pending
            embeddings = prev_future.result()  # Wait for previous GPU work

            result = EmbeddingResult(
                embeddings=embeddings,
                chunks=prev_chunks,
                batch_num=prev_batch_num,
            )
            self.process_fn(result)

        # Store current as pending
        self._pending = (future, chunks, self._batch_num)

    def finish(self) -> None:
        """Process any remaining pending batch and shutdown."""
        if self._pending is not None:
            future, chunks, batch_num = self._pending
            embeddings = future.result()

            result = EmbeddingResult(
                embeddings=embeddings,
                chunks=chunks,
                batch_num=batch_num,
            )
            self.process_fn(result)
            self._pending = None

        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False
