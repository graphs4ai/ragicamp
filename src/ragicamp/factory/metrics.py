"""Metric factory for creating evaluation metrics from configuration."""

from typing import Any, Dict, List, Optional, Union

from ragicamp.core.logging import get_logger
from ragicamp.metrics import Metric
from ragicamp.models.base import LanguageModel

logger = get_logger(__name__)


class MetricFactory:
    """Factory for creating evaluation metrics from configuration."""

    # Custom metric registry
    _custom_metrics: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Register a custom metric type."""
        def decorator(metric_class: type) -> type:
            cls._custom_metrics[name] = metric_class
            return metric_class
        return decorator

    @classmethod
    def create(
        cls,
        config: List[Union[str, Dict[str, Any]]],
        judge_model: Optional[LanguageModel] = None,
    ) -> List[Metric]:
        """Create metrics from configuration.

        Args:
            config: List of metric names or dicts with name and params
            judge_model: Optional judge model for LLM-based metrics

        Returns:
            List of instantiated Metric objects

        Example:
            >>> config = ["exact_match", "f1", {"name": "bertscore", "params": {...}}]
            >>> metrics = MetricFactory.create(config)
        """
        from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

        # Import optional metrics with guards
        try:
            from ragicamp.metrics.bertscore import BERTScoreMetric
            BERTSCORE_AVAILABLE = True
        except ImportError:
            BERTSCORE_AVAILABLE = False

        try:
            from ragicamp.metrics.bleurt import BLEURTMetric
            BLEURT_AVAILABLE = True
        except ImportError:
            BLEURT_AVAILABLE = False

        try:
            from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric
            LLM_JUDGE_AVAILABLE = True
        except ImportError:
            LLM_JUDGE_AVAILABLE = False

        try:
            from ragicamp.metrics.faithfulness import FaithfulnessMetric
            FAITHFULNESS_AVAILABLE = True
        except ImportError:
            FAITHFULNESS_AVAILABLE = False

        try:
            from ragicamp.metrics.hallucination import HallucinationMetric
            HALLUCINATION_AVAILABLE = True
        except ImportError:
            HALLUCINATION_AVAILABLE = False

        metrics = []

        for metric_config in config:
            # Handle both string and dict formats
            if isinstance(metric_config, str):
                metric_name = metric_config
                metric_params = {}
            else:
                metric_name = metric_config["name"]
                metric_params = metric_config.get("params", {})

            # Create metric based on name
            if metric_name == "exact_match":
                metrics.append(ExactMatchMetric())

            elif metric_name == "f1":
                metrics.append(F1Metric())

            elif metric_name == "bertscore":
                if not BERTSCORE_AVAILABLE:
                    logger.warning("Skipping BERTScore (not installed)")
                    continue
                metrics.append(BERTScoreMetric(**metric_params))

            elif metric_name == "bleurt":
                if not BLEURT_AVAILABLE:
                    logger.warning("Skipping BLEURT (not installed)")
                    continue
                metrics.append(BLEURTMetric(**metric_params))

            elif metric_name in ("llm_judge_qa", "llm_judge"):
                if not LLM_JUDGE_AVAILABLE:
                    logger.warning("Skipping LLM Judge (not available)")
                    continue
                if not judge_model:
                    logger.warning("Skipping LLM Judge (judge_model not configured)")
                    continue
                judgment_type = metric_params.get("judgment_type", "binary")
                max_concurrent = metric_params.get(
                    "max_concurrent", metric_params.get("batch_size", 20)
                )
                metrics.append(
                    LLMJudgeQAMetric(
                        judge_model=judge_model,
                        judgment_type=judgment_type,
                        max_concurrent=max_concurrent,
                    )
                )

            elif metric_name == "faithfulness":
                if not FAITHFULNESS_AVAILABLE:
                    logger.warning("Skipping Faithfulness (not installed)")
                    continue
                metrics.append(FaithfulnessMetric(**metric_params))

            elif metric_name == "hallucination":
                if not HALLUCINATION_AVAILABLE:
                    logger.warning("Skipping Hallucination (not installed)")
                    continue
                metrics.append(HallucinationMetric(**metric_params))

            else:
                logger.warning("Unknown metric: %s, skipping", metric_name)

        return metrics
