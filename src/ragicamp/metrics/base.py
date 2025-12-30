"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class MetricResult:
    """Result of computing a metric.

    Attributes:
        name: Metric name (e.g., "f1", "exact_match")
        aggregate: Aggregate score across all examples
        per_item: Per-item scores (one per prediction)
        metadata: Additional metric-specific data
    """

    name: str
    aggregate: float
    per_item: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "aggregate": self.aggregate,
            "per_item": self.per_item,
            "metadata": self.metadata,
        }


class Metric(ABC):
    """Base class for all evaluation metrics.

    Metrics evaluate the quality of generated answers against
    reference answers. All metrics return both aggregate and per-item scores.
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the metric.

        Args:
            name: Metric identifier
            **kwargs: Metric-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._last_per_item: List[float] = []

    @abstractmethod
    def compute(
        self, predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute the metric.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (can be list of lists for multiple refs)
            **kwargs: Additional computation parameters

        Returns:
            Dict of metric scores (e.g., {"exact_match": 0.85})

        Note:
            Implementations should also populate self._last_per_item with per-item scores.
        """
        pass

    def compute_with_details(
        self, predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs: Any
    ) -> MetricResult:
        """Compute the metric and return detailed results.

        Args:
            predictions: List of predicted answers
            references: List of reference answers
            **kwargs: Additional computation parameters

        Returns:
            MetricResult with aggregate and per-item scores
        """
        result = self.compute(predictions, references, **kwargs)
        aggregate = result.get(self.name, 0.0)
        return MetricResult(
            name=self.name,
            aggregate=aggregate,
            per_item=self._last_per_item.copy(),
        )

    def get_per_item_scores(self) -> List[float]:
        """Get per-item scores from the last compute() call.

        Returns:
            List of scores, one per prediction
        """
        return self._last_per_item.copy()

    def compute_single(
        self, prediction: str, reference: Union[str, List[str]], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute metric for a single prediction-reference pair.

        Args:
            prediction: Predicted answer
            reference: Reference answer(s)
            **kwargs: Additional parameters

        Returns:
            Dict of metric scores
        """
        predictions = [prediction]
        references = [reference] if isinstance(reference, str) else [reference]
        return self.compute(predictions, references, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
