"""Evaluation metrics for RAG systems."""

from typing import Any, Dict, List, Optional, Tuple

from ragicamp.metrics.async_base import AsyncAPIMetric
from ragicamp.metrics.base import Metric


def compute_metrics_batched(
    metrics: List[Metric],
    predictions: List[str],
    references: List[Any],
    questions: Optional[List[str]] = None,
    already_computed: Optional[List[str]] = None,
    on_metric_complete: Optional[callable] = None,
) -> Tuple[Dict[str, float], Dict[str, List[float]], List[str], List[str]]:
    """Compute metrics with proper GPU memory management.
    
    This is the shared implementation used by both Experiment._phase_compute_metrics()
    and run_metrics_only() to avoid code duplication.
    
    Args:
        metrics: List of Metric objects to compute
        predictions: List of model predictions
        references: List of reference answers
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
    aggregate_results: Dict[str, float] = {}
    per_item_metrics: Dict[str, List[float]] = {}
    computed: List[str] = []
    failed: List[str] = []
    
    for metric in metrics:
        if metric.name in already_computed:
            print(f"  ✓ {metric.name} (already computed)")
            continue
        
        try:
            print(f"  Computing {metric.name}...")
            ResourceManager.clear_gpu_memory()
            
            # Call metric with appropriate arguments
            if metric.name in ("llm_judge", "llm_judge_qa"):
                if questions is None:
                    print(f"  ⚠ {metric.name} requires questions, skipping")
                    failed.append(metric.name)
                    continue
                scores = metric.compute(
                    predictions=predictions,
                    references=references,
                    questions=questions,
                )
            else:
                scores = metric.compute(predictions=predictions, references=references)
            
            aggregate_results.update(scores)
            
            # Get per-item scores if available
            if hasattr(metric, "get_per_item_scores"):
                per_item = metric.get_per_item_scores()
                if per_item:
                    per_item_metrics[metric.name] = per_item
            
            computed.append(metric.name)
            
            if on_metric_complete:
                on_metric_complete(metric.name)
            
            ResourceManager.clear_gpu_memory()
            
        except Exception as e:
            print(f"  ⚠ {metric.name} failed: {e}")
            failed.append(metric.name)
            ResourceManager.clear_gpu_memory()
    
    return aggregate_results, per_item_metrics, computed, failed

# Import specific metrics (but handle import errors gracefully)
try:
    from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

    _has_exact_match = True
except ImportError:
    _has_exact_match = False

try:
    from ragicamp.metrics.bertscore import BERTScoreMetric

    _has_bertscore = True
except ImportError:
    _has_bertscore = False

try:
    from ragicamp.metrics.bleurt import BLEURTMetric

    _has_bleurt = True
except ImportError:
    _has_bleurt = False

try:
    from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric

    _has_llm_judge = True
except ImportError:
    _has_llm_judge = False

try:
    from ragicamp.metrics.faithfulness import FaithfulnessMetric

    _has_faithfulness = True
except ImportError:
    _has_faithfulness = False

try:
    from ragicamp.metrics.hallucination import HallucinationMetric

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
