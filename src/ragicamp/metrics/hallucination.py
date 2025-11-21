"""Hallucination detection metrics for RAG evaluation.

Detects when the model generates content not supported by retrieved documents.
Complementary to faithfulness - focuses on identifying problematic outputs.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ragicamp.metrics.base import Metric


class HallucinationMetric(Metric):
    """Detects hallucinations in generated answers.

    A hallucination occurs when the model makes claims that are:
    1. Not supported by the retrieved context
    2. Contradicted by the context
    3. Invented from parametric knowledge

    This is essentially the inverse of faithfulness, but provides
    additional features like contradiction detection and claim-level analysis.

    Methods:
    - "nli": NLI-based detection (checks for contradiction)
    - "claim_analysis": Break answer into claims, check each
    - "simple": Token-based heuristic (fast baseline)

    Example:
        >>> from ragicamp.metrics.hallucination import HallucinationMetric
        >>> metric = HallucinationMetric(method="nli")
        >>> score = metric.compute(
        ...     prediction="Paris has 10 million people",
        ...     reference="Paris",
        ...     context=["Paris is the capital of France."]
        ... )
        >>> print(f"Hallucination rate: {score:.2f}")
    """

    def __init__(
        self,
        method: str = "nli",
        nli_model: str = "microsoft/deberta-base-mnli",
        threshold: float = 0.5,
        **kwargs: Any,
    ):
        """Initialize hallucination detection metric.

        Args:
            method: Detection method ("nli", "claim_analysis", or "simple")
            nli_model: NLI model for contradiction detection
            threshold: Confidence threshold for contradiction (0-1)
            **kwargs: Additional arguments passed to base Metric
        """
        super().__init__(name="hallucination", **kwargs)
        self.method = method
        self.nli_model_name = nli_model
        self.threshold = threshold
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
                )
            except ImportError:
                raise ImportError(
                    "transformers library required for NLI-based hallucination detection. "
                    "Install with: pip install transformers"
                )
        return self._nli_pipeline

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def compute(
        self, prediction: str, reference: str, context: Optional[List[str]] = None, **kwargs: Any
    ) -> float:
        """Compute hallucination score.

        Args:
            prediction: Generated answer
            reference: Ground truth answer (not used directly)
            context: Retrieved documents/passages
            **kwargs: Additional arguments

        Returns:
            Hallucination score between 0 and 1:
            - 0.0: No hallucination (fully grounded)
            - 1.0: Complete hallucination (nothing supported)

        Note:
            If context is None/empty, assumes hallucination (returns 1.0).
            Empty predictions return 0.0 (can't hallucinate with no output).
        """
        # No context means we can't verify - assume hallucination
        if not context or len(context) == 0:
            return 1.0

        # Empty prediction can't hallucinate
        if not prediction or not prediction.strip():
            return 0.0

        # Choose method
        if self.method == "nli":
            return self._detect_nli_hallucination(prediction, context)
        elif self.method == "simple":
            return self._detect_simple_hallucination(prediction, context)
        elif self.method == "claim_analysis":
            # For now, fall back to NLI
            # Could be extended to break into claims
            return self._detect_nli_hallucination(prediction, context)
        else:
            raise ValueError(f"Unknown hallucination method: {self.method}")

    def _detect_nli_hallucination(self, prediction: str, context: List[str]) -> float:
        """Detect hallucination using NLI.

        Strategy:
        1. Check if prediction is contradicted by context
        2. Check if prediction is not entailed by any passage
        3. Combine signals to estimate hallucination

        Returns:
            Hallucination score (0=grounded, 1=hallucinated)
        """
        nli = self._get_nli_pipeline()

        max_entailment = 0.0
        max_contradiction = 0.0

        for passage in context:
            if not passage or not passage.strip():
                continue

            # Check entailment and contradiction
            result = nli(f"{passage} [SEP] {prediction}", top_k=None)

            for label_result in result:
                label = label_result["label"].lower()
                score = label_result["score"]

                if "entailment" in label:
                    max_entailment = max(max_entailment, score)
                elif "contradiction" in label:
                    max_contradiction = max(max_contradiction, score)

        # Hallucination indicators:
        # - High contradiction score = likely hallucination
        # - Low entailment score = possibly hallucination

        if max_contradiction > 0.7:
            # Strong contradiction = definite hallucination
            return float(max_contradiction)

        # Otherwise, inverse of entailment (low entailment = high hallucination)
        hallucination_score = 1.0 - max_entailment

        return float(hallucination_score)

    def _detect_simple_hallucination(self, prediction: str, context: List[str]) -> float:
        """Simple token-based hallucination detection.

        Measures what fraction of content words in prediction
        do NOT appear in context. High fraction = likely hallucination.
        """
        pred_tokens = set(prediction.lower().split())

        # Focus on content words (not stopwords)
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
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "their",
        }
        content_tokens = pred_tokens - stopwords

        if not content_tokens:
            return 0.0

        # Combine all context
        context_text = " ".join(context).lower()
        context_tokens = set(context_text.split())

        # Calculate how many content tokens are NOT in context
        unsupported_tokens = content_tokens - context_tokens
        hallucination_ratio = len(unsupported_tokens) / len(content_tokens)

        return float(hallucination_ratio)

    def aggregate(self, scores: List[float]) -> Dict[str, Any]:
        """Aggregate hallucination scores across examples.

        Args:
            scores: List of per-example hallucination scores

        Returns:
            Dictionary with aggregate statistics
        """
        if not scores:
            return {"hallucination_rate": 0.0, "hallucinated_count": 0, "method": self.method}

        scores_array = np.array(scores)

        # Count how many have significant hallucination (> threshold)
        hallucinated_count = np.sum(scores_array >= self.threshold)

        return {
            "hallucination_rate": float(np.mean(scores_array)),
            "hallucinated_ratio": float(hallucinated_count / len(scores)),
            "hallucinated_count": int(hallucinated_count),
            "total": len(scores),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "method": self.method,
            "threshold": self.threshold,
        }
