"""Base class for async API-based metrics.

This module provides a base class for metrics that call external APIs
(like OpenAI for LLM-as-a-judge), with support for:
- Async parallel execution
- Rate limiting via semaphore
- Progress tracking
- Backward-compatible sync interface

Example:
    >>> class MyLLMMetric(AsyncAPIMetric):
    ...     async def acompute_single(self, prediction, reference, question=None, **kwargs):
    ...         # Call OpenAI API
    ...         response = await self.client.chat.completions.create(...)
    ...         return self._parse_score(response)
    >>>
    >>> metric = MyLLMMetric(max_concurrent=10)
    >>> scores = metric.compute(predictions, references)  # Sync interface
    >>> # Or use async directly:
    >>> scores = await metric.acompute(predictions, references)
"""

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from tqdm.asyncio import tqdm as atqdm

from ragicamp.metrics.base import Metric


class AsyncAPIMetric(Metric):
    """Base class for metrics that call external APIs asynchronously.

    Subclasses should implement `acompute_single()` which handles a single
    prediction-reference pair. The base class handles parallelization,
    rate limiting, and progress tracking.

    Attributes:
        max_concurrent: Maximum number of concurrent API calls
        show_progress: Whether to show progress bar
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        show_progress: bool = True,
        **kwargs: Any,
    ):
        """Initialize async API metric.

        Args:
            name: Metric identifier
            max_concurrent: Maximum concurrent API calls (default: 10)
            show_progress: Show progress bar during computation
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.max_concurrent = max_concurrent
        self.show_progress = show_progress

    @abstractmethod
    async def acompute_single(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        question: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute metric for a single prediction-reference pair (async).

        Subclasses must implement this method.

        Args:
            prediction: Predicted answer
            reference: Reference answer(s)
            question: Optional question for context
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores for this single item
        """
        pass

    async def acompute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        questions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute metric for all predictions using async parallel execution.

        Args:
            predictions: List of predicted answers
            references: List of reference answers
            questions: Optional list of questions for context
            **kwargs: Additional parameters

        Returns:
            Dict of aggregated metric scores
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def rate_limited_compute(
            idx: int,
            pred: str,
            ref: Union[str, List[str]],
            q: Optional[str],
        ) -> Dict[str, float]:
            """Compute with rate limiting."""
            async with semaphore:
                try:
                    return await self.acompute_single(pred, ref, q, **kwargs)
                except Exception as e:
                    # Return error score but don't crash the batch
                    print(f"⚠️  Error computing metric for item {idx}: {e}")
                    return {self.name: 0.0, f"{self.name}_error": 1.0}

        # Build tasks
        tasks = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            q = questions[i] if questions and i < len(questions) else None
            tasks.append(rate_limited_compute(i, pred, ref, q))

        # Execute with progress bar
        if self.show_progress:
            results = await atqdm.gather(
                *tasks,
                desc=f"Computing {self.name}",
                total=len(tasks),
            )
        else:
            results = await asyncio.gather(*tasks)

        # Store per-item results for detailed analysis
        self._last_results = results

        # Aggregate results
        return self._aggregate_results(results)

    def get_per_item_scores(self) -> List[float]:
        """Get per-item main scores from last compute() call."""
        results = getattr(self, "_last_results", [])
        return [r.get(self.name, 0.0) for r in results]

    def get_per_item_results(self) -> List[Dict[str, float]]:
        """Get full per-item result dicts from last compute() call."""
        return getattr(self, "_last_results", [])

    def _aggregate_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate individual results into final metrics.

        Args:
            results: List of per-item result dictionaries

        Returns:
            Aggregated metrics dictionary
        """
        if not results:
            return {self.name: 0.0}

        # Collect all metric keys
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())

        # Compute mean for each metric
        aggregated = {}
        for key in all_keys:
            values = [r.get(key, 0.0) for r in results if key in r]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        questions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute metric synchronously (wrapper around async).

        This provides backward compatibility with the sync Metric interface.

        Args:
            predictions: List of predicted answers
            references: List of reference answers
            questions: Optional list of questions
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use asyncio.run()
            # Create a new thread to run the async code
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.acompute(predictions, references, questions, **kwargs)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self.acompute(predictions, references, questions, **kwargs))

    def compute_single(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        question: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute metric for a single item (sync wrapper).

        Args:
            prediction: Predicted answer
            reference: Reference answer(s)
            question: Optional question for context
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores
        """
        return self.compute(
            [prediction],
            [reference],
            [question] if question else None,
            **kwargs,
        )
