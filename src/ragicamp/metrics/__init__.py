"""Evaluation metrics for RAG systems."""

from typing import Any, Optional, Union

from ragicamp.core.logging import get_logger
from ragicamp.metrics.async_base import AsyncAPIMetric
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


def _expand_for_multi_reference(
    predictions: list[str],
    references: list[Union[str, list[str]]],
    questions: Optional[list[str]] = None,
) -> tuple[list[str], list[str], Optional[list[str]], list[int], list[int]]:
    """Expand predictions and references for multi-reference evaluation.

    For each prediction with multiple references, creates one (pred, ref) pair
    per reference. Returns mapping indices to aggregate results later.

    Args:
        predictions: List of model predictions
        references: List of references (can be str or List[str] per item)
        questions: Optional list of questions

    Returns:
        Tuple of:
        - expanded_predictions: Flattened predictions (repeated for each ref)
        - expanded_references: Flattened references (one per prediction)
        - expanded_questions: Flattened questions (or None)
        - pred_indices: Original prediction index for each expanded item
        - ref_indices: Reference index within original list for each expanded item
    """
    expanded_preds = []
    expanded_refs = []
    expanded_questions = [] if questions else None
    pred_indices = []
    ref_indices = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Normalize to list of references
        refs_list = ref if isinstance(ref, list) else [ref]

        for j, single_ref in enumerate(refs_list):
            expanded_preds.append(pred)
            expanded_refs.append(single_ref)
            pred_indices.append(i)
            ref_indices.append(j)
            if questions:
                expanded_questions.append(questions[i])

    return expanded_preds, expanded_refs, expanded_questions, pred_indices, ref_indices


def _aggregate_multi_reference_scores(
    scores: list[float],
    pred_indices: list[int],
    num_predictions: int,
) -> list[float]:
    """Aggregate expanded scores back to original predictions using max.

    For each original prediction, takes the maximum score across all its
    reference comparisons.

    Args:
        scores: Per-item scores from expanded computation
        pred_indices: Original prediction index for each score
        num_predictions: Number of original predictions

    Returns:
        Aggregated scores (one per original prediction)
    """
    # Initialize with -inf so any score will be higher
    best_scores = [float("-inf")] * num_predictions

    for score, idx in zip(scores, pred_indices):
        if score > best_scores[idx]:
            best_scores[idx] = score

    # Replace any remaining -inf (predictions with no references) with 0.0
    return [s if s != float("-inf") else 0.0 for s in best_scores]


def _has_multi_references(references: list[Any]) -> bool:
    """Check if references contain multi-reference items."""
    return any(isinstance(ref, list) for ref in references)


def compute_metrics_batched(
    metrics: list[Metric],
    predictions: list[str],
    references: list[Any],
    questions: Optional[list[str]] = None,
    already_computed: Optional[list[str]] = None,
    on_metric_complete: Optional[callable] = None,
) -> tuple[dict[str, float], dict[str, list[float]], list[str], list[str]]:
    """Compute metrics with proper GPU memory management and multi-reference support.

    This is the shared implementation used by both Experiment._phase_compute_metrics()
    and run_metrics_only() to avoid code duplication.

    Multi-reference handling:
        When references contain lists (multiple correct answers), this function
        expands the computation to evaluate against all references and aggregates
        using max strategy (best score across all references per prediction).

    Args:
        metrics: List of Metric objects to compute
        predictions: List of model predictions
        references: List of reference answers (can be str or List[str] per item)
        questions: Optional list of questions (needed for llm_judge)
        already_computed: List of metric names already computed (to skip)
        on_metric_complete: Callback(metric_name) called after each metric

    Returns:
        Tuple of:
        - aggregate_results: Dict of aggregated metric scores
        - per_item_metrics: Dict of per-item scores for each metric
        - computed: List of successfully computed metric names
        - failed: List of failed metric names
    """
    from ragicamp.utils.resource_manager import ResourceManager

    already_computed = already_computed or []
    aggregate_results: dict[str, float] = {}
    per_item_metrics: dict[str, list[float]] = {}
    computed: list[str] = []
    failed: list[str] = []

    # Check if we need multi-reference handling
    has_multi = _has_multi_references(references)

    if has_multi:
        # Expand for multi-reference evaluation
        expanded_preds, expanded_refs, expanded_questions, pred_indices, _ = (
            _expand_for_multi_reference(predictions, references, questions)
        )
        logger.info(
            "Multi-reference detected: %d predictions -> %d pairs",
            len(predictions), len(expanded_preds),
        )
    else:
        # Simple case: 1-to-1
        expanded_preds = predictions
        expanded_refs = references
        expanded_questions = questions
        pred_indices = list(range(len(predictions)))

    for metric in metrics:
        if metric.name in already_computed:
            logger.info("%s already computed", metric.name)
            continue

        try:
            logger.info("Computing %s...", metric.name)
            ResourceManager.clear_gpu_memory()

            # Call metric with expanded inputs
            if metric.name in ("llm_judge", "llm_judge_qa"):
                if expanded_questions is None:
                    logger.warning("%s requires questions, skipping", metric.name)
                    failed.append(metric.name)
                    continue
                scores = metric.compute(
                    predictions=expanded_preds,
                    references=expanded_refs,
                    questions=expanded_questions,
                )
            else:
                scores = metric.compute(predictions=expanded_preds, references=expanded_refs)

            # Get per-item scores
            if hasattr(metric, "get_per_item_scores"):
                expanded_per_item = metric.get_per_item_scores()
            else:
                expanded_per_item = []

            # Aggregate if multi-reference
            if has_multi and expanded_per_item:
                aggregated_per_item = _aggregate_multi_reference_scores(
                    expanded_per_item, pred_indices, len(predictions)
                )
                per_item_metrics[metric.name] = aggregated_per_item

                # Recompute aggregate from aggregated per-item scores
                avg_score = sum(aggregated_per_item) / len(aggregated_per_item)
                # Update the main metric score with aggregated value
                for key in scores:
                    if metric.name in key:
                        scores[key] = avg_score
                        break
            elif expanded_per_item:
                per_item_metrics[metric.name] = expanded_per_item

            aggregate_results.update(scores)
            computed.append(metric.name)

            if on_metric_complete:
                on_metric_complete(metric.name)

            ResourceManager.clear_gpu_memory()

        except Exception as e:
            logger.warning("%s failed: %s", metric.name, e)
            failed.append(metric.name)
            ResourceManager.clear_gpu_memory()

    return aggregate_results, per_item_metrics, computed, failed


# Import specific metrics (but handle import errors gracefully)
try:
    from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric  # noqa: F401

    _has_exact_match = True
except ImportError:
    _has_exact_match = False

try:
    from ragicamp.metrics.bertscore import BERTScoreMetric  # noqa: F401

    _has_bertscore = True
except ImportError:
    _has_bertscore = False

try:
    from ragicamp.metrics.bleurt import BLEURTMetric  # noqa: F401

    _has_bleurt = True
except ImportError:
    _has_bleurt = False

try:
    from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric  # noqa: F401

    _has_llm_judge = True
except ImportError:
    _has_llm_judge = False

try:
    from ragicamp.metrics.faithfulness import FaithfulnessMetric  # noqa: F401

    _has_faithfulness = True
except ImportError:
    _has_faithfulness = False

try:
    from ragicamp.metrics.hallucination import HallucinationMetric  # noqa: F401

    _has_hallucination = True
except ImportError:
    _has_hallucination = False

__all__ = ["Metric", "AsyncAPIMetric", "compute_metrics_batched"]

# Add available metrics to __all__
if _has_exact_match:
    __all__.extend(["ExactMatchMetric", "F1Metric"])
if _has_bertscore:
    __all__.append("BERTScoreMetric")
if _has_bleurt:
    __all__.append("BLEURTMetric")
if _has_llm_judge:
    __all__.append("LLMJudgeQAMetric")
if _has_faithfulness:
    __all__.append("FaithfulnessMetric")
if _has_hallucination:
    __all__.append("HallucinationMetric")
