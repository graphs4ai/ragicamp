"""Adapter for Ragas metrics to work with RAGiCamp interface.

This module provides a unified interface between Ragas metrics and RAGiCamp's
Metric base class. This allows seamless swapping between custom and Ragas
implementations without changing evaluation code.
"""

from typing import Any, Dict, List, Optional, Union

from ragicamp.metrics.base import Metric


class RagasMetricAdapter(Metric):
    """Adapter that wraps Ragas metrics to RAGiCamp's interface.

    This adapter converts RAGiCamp's evaluation format to Ragas format,
    calls the Ragas metric, and converts results back.

    Example:
        >>> from ragas.metrics import faithfulness
        >>> metric = RagasMetricAdapter(faithfulness, name="faithfulness")
        >>> scores = metric.compute(
        ...     predictions=["Paris is the capital"],
        ...     references=[["Paris"]],
        ...     contexts=[["Paris is France's capital city"]],
        ...     questions=["What is France's capital?"]
        ... )
    """

    def __init__(self, ragas_metric, name: Optional[str] = None, **kwargs):
        """Initialize adapter.

        Args:
            ragas_metric: The Ragas metric instance
            name: Optional override for metric name
            **kwargs: Additional parameters passed to base Metric
        """
        self.ragas_metric = ragas_metric
        metric_name = name or getattr(ragas_metric, "name", "ragas_metric")
        super().__init__(name=metric_name, **kwargs)

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        contexts: Optional[List[List[str]]] = None,
        questions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute metric using Ragas.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (ground truth)
            contexts: List of retrieved contexts (list of docs per question)
            questions: List of questions
            **kwargs: Additional parameters

        Returns:
            Dict with metric scores
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "datasets library required for Ragas metrics. " "Install with: pip install datasets"
            )

        # Prepare data in Ragas format
        num_examples = len(predictions)

        # Normalize references to list of lists
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Use first reference for each question (Ragas expects single ground truth)
        ground_truths = [refs[0] if refs else "" for refs in references]

        # Handle missing optional fields
        if contexts is None:
            contexts = [[] for _ in range(num_examples)]
        if questions is None:
            questions = [""] * num_examples

        # Create Ragas dataset
        data_dict = {
            "question": questions,
            "answer": predictions,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(data_dict)

        # Compute metric using Ragas
        try:
            # Ragas v0.1+ uses score() method
            if hasattr(self.ragas_metric, "score"):
                result = self.ragas_metric.score(dataset)
            # Fallback for older versions
            elif hasattr(self.ragas_metric, "calculate"):
                result = self.ragas_metric.calculate(dataset)
            else:
                raise AttributeError(
                    f"Ragas metric {self.ragas_metric} has no score() or calculate() method"
                )

            # Convert result to our format
            if isinstance(result, dict):
                # Already a dict, just ensure our metric name is present
                if self.name not in result:
                    # Try to find a reasonable score
                    if "score" in result:
                        return {self.name: result["score"]}
                    elif len(result) == 1:
                        return {self.name: list(result.values())[0]}
                return result
            else:
                # Single value returned
                return {self.name: float(result)}

        except Exception as e:
            print(f"⚠️  Ragas metric {self.name} failed: {e}")
            # Return None to indicate failure
            return {self.name: None}

    def __repr__(self) -> str:
        return f"RagasMetricAdapter(metric='{self.name}', ragas_metric={self.ragas_metric})"


def create_ragas_metric(metric_name: str, **kwargs) -> RagasMetricAdapter:
    """Factory function to create Ragas metrics.

    Args:
        metric_name: Name of the Ragas metric
        **kwargs: Parameters for the metric

    Returns:
        RagasMetricAdapter wrapping the Ragas metric

    Example:
        >>> metric = create_ragas_metric("faithfulness")
        >>> metric = create_ragas_metric("context_precision", top_k=5)
    """
    try:
        from ragas import metrics as ragas_metrics
    except ImportError:
        raise ImportError("ragas library required. Install with: pip install ragas")

    # Map metric names to Ragas metrics
    metric_map = {
        "faithfulness": ragas_metrics.faithfulness,
        "answer_relevancy": ragas_metrics.answer_relevancy,
        "context_precision": ragas_metrics.context_precision,
        "context_recall": ragas_metrics.context_recall,
        "context_relevancy": ragas_metrics.context_relevancy,
        "answer_similarity": ragas_metrics.answer_similarity,
        "answer_correctness": ragas_metrics.answer_correctness,
    }

    if metric_name not in metric_map:
        available = ", ".join(sorted(metric_map.keys()))
        raise ValueError(f"Unknown Ragas metric: {metric_name}. Available: {available}")

    ragas_metric = metric_map[metric_name]
    return RagasMetricAdapter(ragas_metric, name=metric_name, **kwargs)
