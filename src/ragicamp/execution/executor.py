"""Resilient executor for agent calls with auto batch size reduction.

This module handles the execution-layer concerns:
- Batch processing with automatic size reduction on errors
- GPU memory management between batches
- Error classification (recoverable vs fatal)
- Progress tracking

The executor is agnostic to what it's executing - it just needs something
with a `batch_answer` or `answer` method.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from tqdm import tqdm

from ragicamp.core.logging import get_logger
from ragicamp.utils.resource_manager import ResourceManager

logger = get_logger(__name__)


# Errors that indicate batch size should be reduced (recoverable errors)
# These patterns match errors from GPU/CUDA operations that can be resolved
# by reducing batch size or retrying with different configuration.
# See also: ragicamp.core.exceptions.RecoverableError
REDUCIBLE_ERROR_PATTERNS = (
    "CUDA",  # General CUDA errors
    "out of memory",  # OOM errors
    "OOM",  # OOM abbreviation
    "invalid configuration argument",  # bitsandbytes CUDA kernel errors (8-bit quant)
    "cuBLAS",  # CUDA BLAS library errors
    "CUDNN",  # cuDNN errors
    "RuntimeError",  # Torch runtime errors (often GPU-related)
    "NCCL",  # Multi-GPU communication errors
    "allocat",  # Catches "allocation", "allocate failed", etc.
    "device-side assert",  # CUDA assertions
)


class Answerable(Protocol):
    """Protocol for anything that can answer questions."""

    def answer(self, query: str, **kwargs: Any) -> Any:
        """Answer a single query."""
        ...


class BatchAnswerable(Answerable, Protocol):
    """Protocol for anything that can answer questions in batches."""

    def batch_answer(self, queries: list[str], **kwargs: Any) -> list[Any]:
        """Answer multiple queries in a batch."""
        ...


@dataclass
class ExecutionConfig:
    """Configuration for resilient execution."""

    batch_size: int = 32
    min_batch_size: int = 1
    checkpoint_callback: Optional[Callable[[int], None]] = None
    progress_bar: bool = True

    def __post_init__(self):
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")
        if self.batch_size < self.min_batch_size:
            raise ValueError("batch_size must be >= min_batch_size")


@dataclass
class BatchResult:
    """Result of processing a batch."""

    items: list[dict[str, Any]]
    batch_size_used: int
    errors: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for item in self.items if not item.get("error"))

    @property
    def error_count(self) -> int:
        return len(self.items) - self.success_count


class ResilientExecutor:
    """Executes agent calls with automatic batch size reduction on errors.

    When a batch fails with a CUDA/OOM error, the batch size is automatically
    halved and the batch is retried. This continues until min_batch_size is
    reached, at which point items are processed sequentially.

    Example:
        executor = ResilientExecutor(agent, batch_size=32)
        results = executor.execute(queries, expected_answers)

        for result in results:
            print(f"Q: {result['query']}")
            print(f"A: {result['prediction']}")
            if result.get('error'):
                print(f"Error: {result['error']}")
    """

    def __init__(
        self,
        agent: Answerable,
        batch_size: int = 32,
        min_batch_size: int = 1,
    ):
        """Initialize the executor.

        Args:
            agent: The agent to execute (must have answer() or batch_answer())
            batch_size: Initial batch size
            min_batch_size: Minimum batch size before falling back to sequential
        """
        self.agent = agent
        self.initial_batch_size = batch_size
        self.current_batch_size = batch_size
        self.min_batch_size = min_batch_size
        self._supports_batch = hasattr(agent, "batch_answer")

    def execute(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool = True,
        checkpoint_every: int = 0,
        checkpoint_callback: Optional[Callable[[list[dict]], None]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute queries with resilient batch processing.

        Args:
            queries: List of (index, question, expected_answers) tuples
            progress: Show progress bar
            checkpoint_every: Save checkpoint every N items (0 = disabled)
            checkpoint_callback: Called with results after each checkpoint
            **kwargs: Additional arguments passed to agent.answer()

        Returns:
            List of result dicts with keys: idx, query, prediction, expected, prompt, error
        """
        if not queries:
            return []

        results: list[dict[str, Any]] = []
        pending = list(queries)

        if self._supports_batch and self.current_batch_size > 1:
            results = self._execute_batched(
                pending, progress, checkpoint_every, checkpoint_callback, **kwargs
            )
        else:
            results = self._execute_sequential(
                pending, progress, checkpoint_every, checkpoint_callback, **kwargs
            )

        # Log final batch size if it was reduced
        if self.current_batch_size < self.initial_batch_size:
            logger.info(
                "Execution completed with reduced batch size: %d (started at %d)",
                self.current_batch_size,
                self.initial_batch_size,
            )

        return results

    def _execute_batched(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool,
        checkpoint_every: int,
        checkpoint_callback: Optional[Callable[[list[dict]], None]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute with batch processing and auto-reduction."""
        import torch

        results: list[dict[str, Any]] = []
        idx = 0
        pbar = tqdm(total=len(queries), desc="Generating", disable=not progress)
        consecutive_failures = 0
        max_consecutive_failures = 5  # Fail entire execution after this many consecutive errors

        while idx < len(queries):
            batch = queries[idx : idx + self.current_batch_size]
            batch_queries = [q for _, q, _ in batch]

            try:
                responses = self.agent.batch_answer(batch_queries, **kwargs)

                for (orig_idx, query, expected), resp in zip(batch, responses):
                    result_item = {
                        "idx": orig_idx,
                        "query": query,
                        "prediction": resp.answer,
                        "expected": expected,
                        "prompt": getattr(resp, "prompt", None),
                        "error": None,
                    }
                    # Include metadata for analysis (RAG steps, iterations, decisions)
                    metadata = (
                        getattr(resp, "metadata_dict", None) or getattr(resp, "metadata", {}) or {}
                    )
                    if metadata:
                        result_item["metadata"] = metadata
                    # Include intermediate steps (iterations, query refinements, etc.)
                    context = getattr(resp, "context", None)
                    if (
                        context
                        and hasattr(context, "intermediate_steps")
                        and context.intermediate_steps
                    ):
                        result_item["intermediate_steps"] = context.intermediate_steps
                    results.append(result_item)

                pbar.update(len(batch))
                idx += self.current_batch_size
                consecutive_failures = 0  # Reset on success

            except Exception as e:
                error_str = str(e)
                consecutive_failures += 1

                # Check if we've hit too many consecutive failures (model may be broken)
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Hit %d consecutive failures - model may be broken. Aborting execution.",
                        consecutive_failures,
                    )
                    # Record remaining items as failed
                    for i in range(idx, len(queries)):
                        orig_idx, query, expected = queries[i]
                        results.append(
                            {
                                "idx": orig_idx,
                                "query": query,
                                "prediction": f"[ABORTED: {consecutive_failures} consecutive failures]",
                                "expected": expected,
                                "prompt": None,
                                "error": f"Aborted after {consecutive_failures} failures: {error_str}",
                            }
                        )
                    pbar.update(len(queries) - idx)
                    break  # Exit the while loop

                if self._is_reducible_error(error_str):
                    if self.current_batch_size > self.min_batch_size:
                        # Halve batch size and retry
                        old_size = self.current_batch_size
                        self.current_batch_size = max(
                            self.min_batch_size, self.current_batch_size // 2
                        )
                        logger.warning(
                            "Batch failed with CUDA error, reducing: %d → %d (attempt %d)",
                            old_size,
                            self.current_batch_size,
                            consecutive_failures,
                        )
                        self._clear_gpu()
                        continue  # Retry same batch with smaller size
                    else:
                        # At min batch size, fall back to sequential for this batch
                        logger.warning(
                            "Batch size at minimum (%d), processing sequentially (attempt %d)",
                            self.min_batch_size,
                            consecutive_failures,
                        )
                        try:
                            seq_results = self._execute_sequential_batch(batch, **kwargs)
                            results.extend(seq_results)
                            pbar.update(len(batch))
                            idx += len(batch)
                            consecutive_failures = 0  # Reset on success
                        except Exception as seq_error:
                            # Even sequential failed - mark all as errors and move on
                            logger.error(
                                "Sequential processing also failed: %s", str(seq_error)[:100]
                            )
                            for orig_idx, query, expected in batch:
                                results.append(
                                    {
                                        "idx": orig_idx,
                                        "query": query,
                                        "prediction": f"[ERROR: {str(seq_error)[:50]}]",
                                        "expected": expected,
                                        "prompt": None,
                                        "error": str(seq_error),
                                    }
                                )
                            pbar.update(len(batch))
                            idx += len(batch)
                else:
                    # Non-reducible error - record and continue
                    logger.warning(
                        "Batch failed (non-recoverable, attempt %d): %s",
                        consecutive_failures,
                        error_str[:80],
                    )
                    for orig_idx, query, expected in batch:
                        results.append(
                            {
                                "idx": orig_idx,
                                "query": query,
                                "prediction": f"[ERROR: {error_str[:50]}]",
                                "expected": expected,
                                "prompt": None,
                                "error": error_str,
                            }
                        )
                    pbar.update(len(batch))
                    idx += len(batch)

            # Checkpoint
            if checkpoint_every and len(results) % checkpoint_every == 0:
                if checkpoint_callback:
                    checkpoint_callback(results)

            # Clear GPU between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()
        return results

    def _execute_sequential(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool,
        checkpoint_every: int,
        checkpoint_callback: Optional[Callable[[list[dict]], None]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute queries one at a time."""
        import torch

        results: list[dict[str, Any]] = []

        for orig_idx, query, expected in tqdm(queries, desc="Generating", disable=not progress):
            try:
                resp = self.agent.answer(query, **kwargs)
                result_item = {
                    "idx": orig_idx,
                    "query": query,
                    "prediction": resp.answer,
                    "expected": expected,
                    "prompt": getattr(resp, "prompt", None),
                    "error": None,
                }
                # Include metadata for analysis (RAG steps, iterations, decisions)
                metadata = (
                    getattr(resp, "metadata_dict", None) or getattr(resp, "metadata", {}) or {}
                )
                if metadata:
                    result_item["metadata"] = metadata
                # Include intermediate steps (iterations, query refinements, etc.)
                context = getattr(resp, "context", None)
                if (
                    context
                    and hasattr(context, "intermediate_steps")
                    and context.intermediate_steps
                ):
                    result_item["intermediate_steps"] = context.intermediate_steps
                results.append(result_item)
            except Exception as e:
                results.append(
                    {
                        "idx": orig_idx,
                        "query": query,
                        "prediction": f"[ERROR: {str(e)[:50]}]",
                        "expected": expected,
                        "prompt": None,
                        "error": str(e),
                    }
                )

            # Checkpoint
            if checkpoint_every and len(results) % checkpoint_every == 0:
                if checkpoint_callback:
                    checkpoint_callback(results)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _execute_sequential_batch(
        self,
        batch: list[tuple[int, str, list[str]]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Process a batch sequentially (fallback when batching fails)."""
        import torch

        results = []
        for orig_idx, query, expected in batch:
            try:
                resp = self.agent.answer(query, **kwargs)
                result_item = {
                    "idx": orig_idx,
                    "query": query,
                    "prediction": resp.answer,
                    "expected": expected,
                    "prompt": getattr(resp, "prompt", None),
                    "error": None,
                }
                # Include metadata for analysis (RAG steps, iterations, decisions)
                metadata = (
                    getattr(resp, "metadata_dict", None) or getattr(resp, "metadata", {}) or {}
                )
                if metadata:
                    result_item["metadata"] = metadata
                # Include intermediate steps (iterations, query refinements, etc.)
                context = getattr(resp, "context", None)
                if (
                    context
                    and hasattr(context, "intermediate_steps")
                    and context.intermediate_steps
                ):
                    result_item["intermediate_steps"] = context.intermediate_steps
                results.append(result_item)
            except Exception as e:
                results.append(
                    {
                        "idx": orig_idx,
                        "query": query,
                        "prediction": f"[ERROR: {str(e)[:50]}]",
                        "expected": expected,
                        "prompt": None,
                        "error": str(e),
                    }
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _is_reducible_error(self, error_str: str) -> bool:
        """Check if error indicates batch size should be reduced."""
        error_lower = error_str.lower()
        return any(pattern.lower() in error_lower for pattern in REDUCIBLE_ERROR_PATTERNS)

    def _clear_gpu(self) -> None:
        """Clear GPU memory for retry."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        ResourceManager.clear_gpu_memory()
