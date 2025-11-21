"""BERTScore metric implementation."""

import os
from typing import Any, Dict, List, Union

# Fix matplotlib backend BEFORE any imports that might use it
# This prevents errors when running in non-interactive environments (scripts vs notebooks)
if "MPLBACKEND" in os.environ:
    # Change to non-interactive backend for scripts
    os.environ["MPLBACKEND"] = "Agg"

from ragicamp.metrics.base import Metric


class BERTScoreMetric(Metric):
    """BERTScore for semantic similarity evaluation."""

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", **kwargs: Any):
        """Initialize BERTScore metric.

        Args:
            model_type: Model to use for computing BERTScore
            **kwargs: Additional configuration
        """
        super().__init__(name="bertscore", **kwargs)
        self.model_type = model_type

        # Lazy import to avoid requiring bert-score unless used
        try:
            from bert_score import BERTScorer

            self.scorer = BERTScorer(model_type=model_type, lang="en")
        except ImportError:
            raise ImportError(
                "bert-score is required for BERTScoreMetric. "
                "Install with: pip install bert-score"
            )

    def compute(
        self, predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute BERTScore.

        Returns:
            Dict with precision, recall, and F1 scores
        """
        # Handle multiple references - take first one for now
        # TODO: Extend to handle multiple references properly
        refs = []
        for ref in references:
            if isinstance(ref, list):
                refs.append(ref[0])
            else:
                refs.append(ref)

        # Compute BERTScore
        P, R, F1 = self.scorer.score(predictions, refs)

        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
