"""Resilient executor for agent calls.

This module handles the execution-layer concerns:
- Batch processing for throughput
- GPU memory management between batches
- Error handling and progress tracking

The executor is agnostic to what it's executing - it just needs something
with a `batch_answer` or `answer` method.

Note: With vLLM as the backend, batch size just controls how many prompts
are queued at once. vLLM handles its own GPU memory management internally.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from tqdm import tqdm

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


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
    """Configuration for execution."""

    batch_size: int = 32
    checkpoint_callback: Optional[Callable[[int], None]] = None
    progress_bar: bool = True

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


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
    """Executes agent calls with batch processing.

    With vLLM as the backend, batch size just controls how many prompts
    are queued at once. vLLM handles its own GPU memory management.

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
    ):
        """Initialize the executor.

        Args:
            agent: The agent to execute (must have answer() or batch_answer())
            batch_size: Number of prompts to process per batch
        """
        self.agent = agent
        self.batch_size = batch_size
        self._supports_batch = hasattr(agent, "batch_answer")

    def execute(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool = True,
        checkpoint_every: int = 0,
        checkpoint_callback: Optional[Callable[[list[dict]], None]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute queries with batch processing.

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

        pending = list(queries)

        if self._supports_batch and self.batch_size > 1:
            return self._execute_batched(
                pending, progress, checkpoint_every, checkpoint_callback, **kwargs
            )
        else:
            return self._execute_sequential(
                pending, progress, checkpoint_every, checkpoint_callback, **kwargs
            )

    def _execute_batched(
        self,
        queries: list[tuple[int, str, list[str]]],
        progress: bool,
        checkpoint_every: int,
        checkpoint_callback: Optional[Callable[[list[dict]], None]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute with batch processing."""
        import torch

        results: list[dict[str, Any]] = []
        idx = 0
        pbar = tqdm(total=len(queries), desc="Generating", disable=not progress)
        consecutive_failures = 0
        max_consecutive_failures = 5  # Fail entire execution after this many consecutive errors

        while idx < len(queries):
            batch = queries[idx : idx + self.batch_size]
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
                idx += self.batch_size
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

                # Log error and record failed items
                logger.warning(
                    "Batch failed (attempt %d/%d): %s",
                    consecutive_failures,
                    max_consecutive_failures,
                    error_str[:100],
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

