"""Evaluation utilities for computing metrics on predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


def compute_metrics_from_file(
    predictions_path: str,
    metrics: List[Metric],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute metrics from saved predictions file.

    This is useful for post-hoc evaluation when you have predictions
    but want to compute additional metrics.

    Args:
        predictions_path: Path to predictions JSON file
        metrics: List of metrics to compute
        output_path: Optional path to save results

    Returns:
        Dictionary with metric scores (aggregate and per-item)

    Example:
        >>> from ragicamp.metrics import F1Metric, ExactMatchMetric
        >>> metrics = [F1Metric(), ExactMatchMetric()]
        >>> results = compute_metrics_from_file("predictions.json", metrics)
        >>> print(results["aggregate"]["f1"])
    """
    logger.info("Loading predictions from: %s", predictions_path)

    with open(predictions_path) as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    predictions = [p["prediction"] for p in preds]
    questions = [p["question"] for p in preds]
    references = [p.get("expected", "") for p in preds]

    # Compute metrics
    aggregate_results: Dict[str, float] = {}
    per_item_results: Dict[str, List[float]] = {}

    for metric in metrics:
        try:
            logger.debug("Computing %s...", metric.name)
            if metric.name in ("llm_judge", "llm_judge_qa"):
                scores = metric.compute(
                    predictions=predictions, references=references, questions=questions
                )
            else:
                scores = metric.compute(predictions=predictions, references=references)
            aggregate_results.update(scores)

            # Get per-item scores
            if hasattr(metric, "get_per_item_scores"):
                per_item = metric.get_per_item_scores()
                if per_item:
                    per_item_results[metric.name] = per_item
        except Exception as e:
            logger.warning("%s failed: %s", metric.name, e)

    results = {
        "aggregate": aggregate_results,
        "per_item": per_item_results,
        "num_examples": len(predictions),
    }

    # Update predictions file with new metrics if requested
    if output_path:
        # Update per-prediction metrics
        for i, pred in enumerate(preds):
            if "metrics" not in pred:
                pred["metrics"] = {}
            for metric_name, scores in per_item_results.items():
                if i < len(scores):
                    pred["metrics"][metric_name] = scores[i]

        # Save updated predictions
        data["aggregate_metrics"] = aggregate_results
        data["predictions"] = preds

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved updated predictions to: %s", output_path)

    return results


# For backwards compatibility
class Evaluator:
    """Deprecated: Use compute_metrics_from_file() or Experiment.run() instead."""

    @staticmethod
    def compute_metrics_from_file(
        predictions_path: str,
        metrics: List[Metric],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute metrics from saved predictions file."""
        return compute_metrics_from_file(predictions_path, metrics, output_path)


__all__ = ["compute_metrics_from_file", "Evaluator"]
