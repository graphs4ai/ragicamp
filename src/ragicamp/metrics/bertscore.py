"""BERTScore metric implementation with lazy loading and proper GPU cleanup."""

import gc
import os
from typing import Any, Optional

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
        except ImportError as e:
            raise ImportError(
                "bert-score is required for BERTScoreMetric. Install with: pip install bert-score"
            ) from e

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
        print("  🗑️  BERTScore model unloaded")

    def compute(
        self, predictions: list[str], references: list[str], **kwargs: Any
    ) -> dict[str, float]:
        """Compute BERTScore with automatic batching for memory efficiency (1-to-1).

        Loads model, computes scores in batches, then unloads to free GPU memory.

        Args:
            predictions: List of predicted answers
            references: List of reference answers (one per prediction)

        Returns:
            Dict with precision, recall, and F1 scores
        """
        import torch

        try:
            # Load model (lazy)
            self._load_scorer()

            # Compute BERTScore in batches to avoid OOM
            batch_size = self._estimate_batch_size(predictions)

            all_P, all_R, all_F1 = [], [], []

            for i in range(0, len(predictions), batch_size):
                batch_preds = predictions[i : i + batch_size]
                batch_refs = references[i : i + batch_size]

                # Clear cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                P, R, F1 = self._scorer.score(batch_preds, batch_refs)
                all_P.append(P)
                all_R.append(R)
                all_F1.append(F1)

            # Concatenate results
            P = torch.cat(all_P)
            R = torch.cat(all_R)
            F1 = torch.cat(all_F1)

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

    def _estimate_batch_size(self, predictions: list[str]) -> int:
        """Estimate safe batch size based on text lengths and GPU memory.

        BERTScore memory scales with sequence length squared.
        For deberta-xlarge-mnli:
        - Model base: ~2.5 GB
        - Each batch item with long text can use 0.5-1 GB
        """

        if not predictions:
            return 1

        # Calculate average and max text lengths
        avg_len = sum(len(p) for p in predictions) / len(predictions)
        max_len = max(len(p) for p in predictions)

        # Very conservative batch sizing based on text length
        # Long texts (like RAG predictions with context) need very small batches
        if max_len > 3000 or avg_len > 1000:
            batch_size = 2  # Very long texts - minimal batch
        elif max_len > 1500 or avg_len > 500:
            batch_size = 4  # Long texts
        elif max_len > 500 or avg_len > 200:
            batch_size = 8  # Medium texts
        else:
            batch_size = 16  # Short texts

        # Cap at total predictions
        batch_size = min(batch_size, len(predictions))

        if len(predictions) > batch_size:
            print(f"    Processing {len(predictions)} items in batches of {batch_size}")

        return batch_size

    def get_per_item_scores(self) -> list[float]:
        """Get per-item F1 scores from last compute() call."""
        return getattr(self, "_last_scores", [])
