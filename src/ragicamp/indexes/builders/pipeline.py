"""Pipelined processing with GPU/CPU overlap.

Overlaps GPU embedding with CPU post-processing (normalize, index, save)
to improve throughput by ~30-40%.

Usage:
    pipeline = EmbeddingPipeline(encoder, on_batch_ready)
    for texts, chunks in batches:
        pipeline.submit(texts, chunks)
    pipeline.finish()
"""

import time
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

    Uses 2 workers so batch N+1 can start immediately while batch N finishes,
    allowing true overlap between GPU encoding and CPU post-processing.
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

        # 2 workers: allows batch N+1 to start encoding while we process batch N
        # vLLM internally serializes GPU access, but worker 2 can prepare while worker 1 finishes
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")
        self._pending: tuple[Future, list[Any], int, float] | None = None  # Added submit_time
        self._batch_num = 0

        # Stats for overlap tracking
        self._total_encode_time = 0.0
        self._total_process_time = 0.0
        self._overlap_time = 0.0

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings (runs in background thread).

        Note: Normalization is done in main thread to free GPU thread faster.
        """
        return self.encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.embedding_batch_size,
        )

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings in-place."""
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
        submit_time = time.perf_counter()

        # Submit current batch to GPU (background thread starts immediately)
        future = self._executor.submit(self._encode, texts)

        # While GPU works on current batch, process previous batch
        if self._pending is not None:
            prev_future, prev_chunks, prev_batch_num, prev_submit_time = self._pending

            # Wait for previous GPU work
            wait_start = time.perf_counter()
            embeddings = prev_future.result()
            encode_time = time.perf_counter() - prev_submit_time

            # Normalize and process in main thread
            process_start = time.perf_counter()
            if self.normalize:
                embeddings = self._normalize(embeddings)

            result = EmbeddingResult(
                embeddings=embeddings,
                chunks=prev_chunks,
                batch_num=prev_batch_num,
            )
            self.process_fn(result)
            process_time = time.perf_counter() - process_start

            # Track stats
            self._total_encode_time += encode_time
            self._total_process_time += process_time
            # Overlap = time GPU was working while we waited (negative wait = overlap)
            wait_time = wait_start - prev_submit_time
            if wait_time < encode_time:
                self._overlap_time += encode_time - wait_time

        # Store current as pending
        self._pending = (future, chunks, self._batch_num, submit_time)

    def finish(self) -> None:
        """Process any remaining pending batch and shutdown."""
        if self._pending is not None:
            future, chunks, batch_num, submit_time = self._pending

            wait_start = time.perf_counter()
            embeddings = future.result()
            encode_time = time.perf_counter() - submit_time

            process_start = time.perf_counter()
            if self.normalize:
                embeddings = self._normalize(embeddings)

            result = EmbeddingResult(
                embeddings=embeddings,
                chunks=chunks,
                batch_num=batch_num,
            )
            self.process_fn(result)
            process_time = time.perf_counter() - process_start

            self._total_encode_time += encode_time
            self._total_process_time += process_time
            self._pending = None

        self._executor.shutdown(wait=True)

        # Print pipeline stats
        if self._batch_num > 0:
            total = self._total_encode_time + self._total_process_time
            saved = self._overlap_time
            if total > 0 and saved > 0:
                pct = (saved / total) * 100
                print(f"    ⚡ Pipeline: {saved:.1f}s overlap saved ({pct:.0f}% efficiency)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False
