"""Evaluation utilities for computing metrics on predictions."""

import json
from pathlib import Path
from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric
from ragicamp.utils.experiment_io import atomic_write_json

logger = get_logger(__name__)


def compute_metrics_from_file(
    predictions_path: str,
    metrics: list[Metric],
    output_path: str | None = None,
) -> dict[str, Any]:
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
    aggregate_results: dict[str, float] = {}
    per_item_results: dict[str, list[float]] = {}

    for metric in metrics:
        try:
            logger.debug("Computing %s...", metric.name)
            if metric.name in ("llm_judge", "llm_judge_qa"):
                result = metric.compute_with_details(
                    predictions=predictions, references=references, questions=questions
                )
            else:
                result = metric.compute_with_details(predictions=predictions, references=references)
            aggregate_results[result.name] = result.aggregate
            if result.per_item:
                per_item_results[result.name] = result.per_item
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

        # Save updated predictions atomically
        data["aggregate_metrics"] = aggregate_results
        data["predictions"] = preds

        atomic_write_json(data, Path(output_path))
        logger.info("Saved updated predictions to: %s", output_path)

    return results


__all__ = ["compute_metrics_from_file"]
