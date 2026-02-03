"""Pipelined processing with GPU/CPU overlap.

Overlaps GPU embedding with CPU post-processing (normalize, index, save).

Supports two modes:
- Sync mode (ThreadPoolExecutor): For VLLMEmbedder/SentenceTransformer
- Async mode (asyncio): For VLLMServerEmbedder with true overlap

Usage:
    pipeline = EmbeddingPipeline(encoder, on_batch_ready)
    for texts, chunks in batches:
        pipeline.submit(texts, chunks)
    pipeline.finish()
"""

import asyncio
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

import numpy as np

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


def _timestamp() -> str:
    """Return current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def _has_async_encode(encoder) -> bool:
    """Check if encoder supports async encoding."""
    return hasattr(encoder, "encode_async") and asyncio.iscoroutinefunction(encoder.encode_async)


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
    """Batched embedding with GPU/CPU overlap.

    Supports two modes:
    - Async mode (VLLMServerEmbedder): True overlap via asyncio
    - Sync mode (VLLMEmbedder/SentenceTransformer): Limited overlap via threads
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
            encoder: Embedding encoder (VLLMEmbedder, VLLMServerEmbedder, or SentenceTransformer)
            process_fn: Callback to process each batch result (receives EmbeddingResult)
            embedding_batch_size: Batch size for encoder
            normalize: Whether to L2-normalize embeddings
        """
        self.encoder = encoder
        self.process_fn = process_fn
        self.embedding_batch_size = embedding_batch_size
        self.normalize = normalize

        self._use_async = _has_async_encode(encoder)
        self._batch_num = 0

        # Stats for overlap tracking
        self._total_encode_time = 0.0
        self._total_process_time = 0.0
        self._overlap_time = 0.0

        if self._use_async:
            print(f"    [{_timestamp()}] Using async pipeline (true overlap)")
            self._loop = asyncio.new_event_loop()
            self._pending_task: asyncio.Task | None = None
            self._pending_data: tuple[list[Any], int, float] | None = None
        else:
            print(f"    [{_timestamp()}] Using sync pipeline (limited overlap)")
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
            self._pending: tuple[Future, list[Any], int, float] | None = None

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings in-place."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.divide(embeddings, norms, out=embeddings)
        return embeddings

    def _process_embeddings(self, embeddings: np.ndarray, chunks: list[Any], batch_num: int) -> float:
        """Process embeddings: normalize and call process_fn. Returns process time."""
        process_start = time.perf_counter()

        if self.normalize:
            print(f"    [{_timestamp()}] Normalizing {len(embeddings)} embeddings...")
            norm_start = time.perf_counter()
            embeddings = self._normalize(embeddings)
            norm_time = time.perf_counter() - norm_start
            print(f"    [{_timestamp()}] Normalized in {norm_time:.1f}s")

        print(f"    [{_timestamp()}] Saving batch {batch_num} to index...")
        result = EmbeddingResult(embeddings=embeddings, chunks=chunks, batch_num=batch_num)
        self.process_fn(result)

        return time.perf_counter() - process_start

    def submit(self, texts: list[str], chunks: list[Any]) -> None:
        """Submit a batch for embedding."""
        self._batch_num += 1
        submit_time = time.perf_counter()

        if self._use_async:
            self._submit_async(texts, chunks, submit_time)
        else:
            self._submit_sync(texts, chunks, submit_time)

    def _submit_async(self, texts: list[str], chunks: list[Any], submit_time: float) -> None:
        """Async submit using asyncio."""
        # Start encoding current batch
        async def encode():
            return await self.encoder.encode_async(texts)

        new_task = self._loop.create_task(encode())

        # Process previous batch while new one encodes
        if self._pending_task is not None:
            prev_chunks, prev_batch_num, prev_submit_time = self._pending_data

            # Wait for previous task
            embeddings = self._loop.run_until_complete(self._pending_task)
            encode_time = time.perf_counter() - prev_submit_time

            # Process (CPU work - new task running in background)
            process_time = self._process_embeddings(embeddings, prev_chunks, prev_batch_num)

            self._total_encode_time += encode_time
            self._total_process_time += process_time
            self._overlap_time += min(process_time, encode_time)

        self._pending_task = new_task
        self._pending_data = (chunks, self._batch_num, submit_time)

    def _submit_sync(self, texts: list[str], chunks: list[Any], submit_time: float) -> None:
        """Sync submit using ThreadPoolExecutor."""
        def encode():
            return self.encoder.encode(texts, show_progress_bar=True, batch_size=self.embedding_batch_size)

        future = self._executor.submit(encode)

        if self._pending is not None:
            prev_future, prev_chunks, prev_batch_num, prev_submit_time = self._pending

            wait_start = time.perf_counter()
            embeddings = prev_future.result()
            encode_time = time.perf_counter() - prev_submit_time

            process_time = self._process_embeddings(embeddings, prev_chunks, prev_batch_num)

            self._total_encode_time += encode_time
            self._total_process_time += process_time
            wait_time = wait_start - prev_submit_time
            if wait_time < encode_time:
                self._overlap_time += encode_time - wait_time

        self._pending = (future, chunks, self._batch_num, submit_time)

    def finish(self) -> None:
        """Process any remaining pending batch and shutdown."""
        if self._use_async:
            self._finish_async()
        else:
            self._finish_sync()

        # Print pipeline stats
        if self._batch_num > 0:
            total = self._total_encode_time + self._total_process_time
            saved = self._overlap_time
            print(f"    [{_timestamp()}] ⚡ Pipeline complete: encode={self._total_encode_time:.1f}s, process={self._total_process_time:.1f}s")
            if total > 0 and saved > 0:
                pct = (saved / total) * 100
                print(f"    [{_timestamp()}] ⚡ Overlap saved: {saved:.1f}s ({pct:.0f}% efficiency)")

    def _finish_async(self) -> None:
        """Finish async pipeline."""
        if self._pending_task is not None:
            chunks, batch_num, submit_time = self._pending_data
            embeddings = self._loop.run_until_complete(self._pending_task)
            encode_time = time.perf_counter() - submit_time

            process_time = self._process_embeddings(embeddings, chunks, batch_num)

            self._total_encode_time += encode_time
            self._total_process_time += process_time
            self._pending_task = None
            self._pending_data = None

        self._loop.close()

    def _finish_sync(self) -> None:
        """Finish sync pipeline."""
        if self._pending is not None:
            future, chunks, batch_num, submit_time = self._pending
            embeddings = future.result()
            encode_time = time.perf_counter() - submit_time

            process_time = self._process_embeddings(embeddings, chunks, batch_num)

            self._total_encode_time += encode_time
            self._total_process_time += process_time
            self._pending = None

        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False
