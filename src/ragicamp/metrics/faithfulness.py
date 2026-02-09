"""Faithfulness/groundedness metrics for RAG evaluation.

Measures whether answers are actually supported by the retrieved context.
This is critical for RAG systems to prevent hallucination and ensure answers
are grounded in the provided documents.
"""

from typing import Any, Optional

import numpy as np

from ragicamp.core.logging import get_logger
from ragicamp.metrics.base import Metric

logger = get_logger(__name__)


class FaithfulnessMetric(Metric):
    """Evaluates if answers are grounded in retrieved context.

    Uses Natural Language Inference (NLI) to determine if the answer
    is entailed by the retrieved documents. This helps identify when
    the model hallucinates or uses parametric knowledge instead of
    the provided context.

    Methods:
    - "nli": Fast, high-quality NLI model (default)
    - "token_overlap": Simple baseline using token overlap
    - "llm": Use LLM for judgment (requires judge_model)

    Example:
        >>> from ragicamp.metrics.faithfulness import FaithfulnessMetric
        >>> metric = FaithfulnessMetric(method="nli")
        >>> score = metric.compute(
        ...     prediction="Paris is the capital of France",
        ...     reference="France",
        ...     context=["Paris is France's capital city."]
        ... )
        >>> print(f"Faithfulness: {score:.2f}")
    """

    def __init__(
        self,
        method: str = "nli",
        nli_model: str = "microsoft/deberta-base-mnli",
        threshold: float = 0.5,
        batch_size: int = 8,
        judge_model: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize faithfulness metric.

        Args:
            method: Evaluation method ("nli", "token_overlap", or "llm")
            nli_model: NLI model for entailment checking
            threshold: Confidence threshold for entailment (0-1)
            batch_size: Batch size for NLI inference
            judge_model: Optional LLM for "llm" method
            **kwargs: Additional arguments passed to base Metric
        """
        super().__init__(name="faithfulness", **kwargs)
        self.method = method
        self.nli_model_name = nli_model
        self.threshold = threshold
        self.batch_size = batch_size
        self.judge_model = judge_model
        self._nli_pipeline = None

    def _get_nli_pipeline(self):
        """Lazy load NLI pipeline."""
        if self._nli_pipeline is None:
            try:
                from transformers import pipeline

                self._nli_pipeline = pipeline(
                    "text-classification",
                    model=self.nli_model_name,
                    device=0 if self._has_cuda() else -1,
                    batch_size=self.batch_size,
                )
            except ImportError as e:
                raise ImportError(
                    "transformers library required for NLI-based faithfulness. "
                    "Install with: pip install transformers"
                ) from e
        return self._nli_pipeline

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def compute(
        self, prediction: str, reference: str, context: Optional[list[str]] = None, **kwargs: Any
    ) -> float:
        """Compute faithfulness score.

        Args:
            prediction: Generated answer
            reference: Ground truth answer (not used for faithfulness, but kept for API compatibility)
            context: Retrieved documents/passages
            **kwargs: Additional arguments

        Returns:
            Faithfulness score between 0 and 1:
            - 1.0: Answer fully supported by context
            - 0.0: Answer not supported by context

        Note:
            If context is None or empty, returns 0.0 (cannot verify faithfulness).
        """
        # Need context to evaluate faithfulness
        if not context or len(context) == 0:
            return 0.0

        if not prediction or not prediction.strip():
            return 0.0

        # Choose method
        if self.method == "nli":
            return self._compute_nli_faithfulness(prediction, context)
        elif self.method == "token_overlap":
            return self._compute_token_overlap(prediction, context)
        elif self.method == "llm":
            return self._compute_llm_faithfulness(prediction, context)
        else:
            raise ValueError(f"Unknown faithfulness method: {self.method}")

    def _compute_nli_faithfulness(self, prediction: str, context: list[str]) -> float:
        """Compute faithfulness using NLI entailment.

        Strategy:
        1. For each context passage, check if it entails the prediction
        2. If any passage entails the prediction with confidence > threshold, faithful
        3. Return max entailment score across all passages
        """
        nli = self._get_nli_pipeline()

        # Create premise-hypothesis pairs
        # Premise = context passage, Hypothesis = prediction
        max_entailment = 0.0

        for passage in context:
            if not passage or not passage.strip():
                continue

            # NLI expects premise-hypothesis format
            # Format depends on model, but generally: premise [SEP] hypothesis
            result = nli(f"{passage} [SEP] {prediction}", top_k=None)

            # Find entailment score
            entailment_score = 0.0
            for label_result in result:
                if label_result["label"].lower() in ["entailment", "entailment"]:
                    entailment_score = label_result["score"]
                    break

            max_entailment = max(max_entailment, entailment_score)

            # Early exit if we found strong entailment
            if max_entailment > 0.9:
                break

        return float(max_entailment)

    def _compute_token_overlap(self, prediction: str, context: list[str]) -> float:
        """Simple token overlap baseline.

        Measures what fraction of prediction tokens appear in context.
        Not as accurate as NLI, but fast and interpretable.
        """
        pred_tokens = set(prediction.lower().split())

        # Remove very common words that don't indicate faithfulness
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "but",
        }
        pred_tokens = pred_tokens - stopwords

        if not pred_tokens:
            return 0.0

        # Combine all context
        context_text = " ".join(context).lower()
        context_tokens = set(context_text.split())

        # Calculate overlap
        overlap = len(pred_tokens & context_tokens)
        precision = overlap / len(pred_tokens)

        return float(precision)

    def _compute_llm_faithfulness(self, prediction: str, context: list[str]) -> float:
        """Use LLM to judge faithfulness.

        Requires judge_model to be set.
        """
        if self.judge_model is None:
            raise ValueError("judge_model required for LLM-based faithfulness")

        # Combine context
        combined_context = "\n\n".join(context[:3])  # Use top 3 passages

        prompt = f"""Given the following context and answer, determine if the answer is fully supported by the context.

Context:
{combined_context}

Answer:
{prediction}

Is the answer fully supported by the context? Respond with only:
- "YES" if every claim in the answer is supported by the context
- "NO" if the answer contains unsupported claims or contradicts the context

Response:"""

        try:
            response = self.judge_model.generate(prompt, max_tokens=10, temperature=0.0)
            response = response.strip().upper()

            if "YES" in response:
                return 1.0
            elif "NO" in response:
                return 0.0
            else:
                # Unclear response, default to neutral
                return 0.5
        except Exception as e:
            logger.warning("LLM faithfulness check failed: %s", e)
            return 0.5

    def aggregate(self, scores: list[float]) -> dict[str, Any]:
        """Aggregate faithfulness scores across examples.

        Args:
            scores: List of per-example faithfulness scores

        Returns:
            Dictionary with aggregate statistics
        """
        if not scores:
            return {"faithfulness": 0.0, "faithful_ratio": 0.0, "method": self.method}

        scores_array = np.array(scores)

        # Count how many are "faithful" (> threshold)
        faithful_count = np.sum(scores_array >= self.threshold)

        return {
            "faithfulness": float(np.mean(scores_array)),
            "faithful_ratio": float(faithful_count / len(scores)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "method": self.method,
            "threshold": self.threshold,
        }
