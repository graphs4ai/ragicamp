"""BERTScore metric implementation with lazy loading and proper GPU cleanup."""

import gc
import os
from typing import Any, Dict, List, Optional, Union

# Fix matplotlib backend BEFORE any imports that might use it
# This prevents errors when running in non-interactive environments (scripts vs notebooks)
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

from ragicamp.metrics.base import Metric


class BERTScoreMetric(Metric):
    """BERTScore for semantic similarity evaluation.

    Uses lazy loading - model is only loaded when compute() is called,
    and unloaded immediately after to free GPU memory.
    """

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", **kwargs: Any):
        """Initialize BERTScore metric.

        Args:
            model_type: Model to use for computing BERTScore
            **kwargs: Additional configuration
        """
        super().__init__(name="bertscore", **kwargs)
        self.model_type = model_type
        self._scorer: Optional[Any] = None  # Lazy loaded

    def _load_scorer(self) -> None:
        """Load the BERTScore model (lazy loading)."""
        if self._scorer is not None:
            return

        try:
            from bert_score import BERTScorer

            print(f"  📥 Loading BERTScore model: {self.model_type}")
            self._scorer = BERTScorer(model_type=self.model_type, lang="en")
        except ImportError:
            raise ImportError(
                "bert-score is required for BERTScoreMetric. "
                "Install with: pip install bert-score"
            )

    def _unload_scorer(self) -> None:
        """Unload the BERTScore model to free GPU memory."""
        if self._scorer is None:
            return

        import torch

        # Delete the scorer and its model
        if hasattr(self._scorer, "model"):
            del self._scorer.model
        del self._scorer
        self._scorer = None

        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  🗑️  BERTScore model unloaded")

    def compute(
        self, predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute BERTScore.

        Loads model, computes scores, then unloads to free GPU memory.

        Returns:
            Dict with precision, recall, and F1 scores
        """
        # Handle multiple references - take first one for now
        refs = []
        for ref in references:
            if isinstance(ref, list):
                refs.append(ref[0] if ref else "")
            else:
                refs.append(ref)

        try:
            # Load model (lazy)
            self._load_scorer()

            # Compute BERTScore
            P, R, F1 = self._scorer.score(predictions, refs)

            # Store per-item scores for detailed analysis
            self._last_scores = F1.tolist()

            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item(),
            }
        finally:
            # ALWAYS unload after computation to free GPU
            self._unload_scorer()

    def get_per_item_scores(self) -> List[float]:
        """Get per-item F1 scores from last compute() call."""
        return getattr(self, "_last_scores", [])
