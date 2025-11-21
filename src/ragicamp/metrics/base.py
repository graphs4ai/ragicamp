"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class Metric(ABC):
    """Base class for all evaluation metrics.

    Metrics evaluate the quality of generated answers against
    reference answers.
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize the metric.

        Args:
            name: Metric identifier
            **kwargs: Metric-specific configuration
        """
        self.name = name
        self.config = kwargs

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
            Dict of metric scores (e.g., {"exact_match": 0.85, "f1": 0.92})
            All metrics should return a dictionary for consistency
        """
        pass

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
