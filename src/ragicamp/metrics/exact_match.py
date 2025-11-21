"""Exact match and F1 metrics with normalization.

Follows best practices from SQuAD and other QA benchmarks:
- Lowercase normalization
- Article removal (a, an, the)
- Punctuation removal
- Whitespace normalization
- Optional: stemming/lemmatization for even more robustness
"""

import re
import string
from collections import Counter
from typing import Any, Dict, List, Union, Optional

from ragicamp.metrics.base import Metric


def normalize_answer(
    text: str,
    lowercase: bool = True,
    remove_articles: bool = True,
    remove_punctuation: bool = True,
    remove_extra_whitespace: bool = True,
    stemmer: Optional[Any] = None,
) -> str:
    """Normalize answer text following SQuAD evaluation practices.

    Args:
        text: Text to normalize
        lowercase: Convert to lowercase
        remove_articles: Remove articles (a, an, the)
        remove_punctuation: Remove punctuation
        remove_extra_whitespace: Collapse multiple spaces
        stemmer: Optional stemmer (e.g., PorterStemmer or SnowballStemmer)

    Returns:
        Normalized text

    Note:
        This follows the official SQuAD evaluation script normalization.
        Stemming is NOT standard in SQuAD but can help with word variations.
    """
    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove articles (standard in SQuAD)
    if remove_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Normalize whitespace
    if remove_extra_whitespace:
        text = " ".join(text.split())

    # Optional stemming (not standard, but can be useful)
    if stemmer is not None:
        tokens = text.split()
        tokens = [stemmer.stem(token) for token in tokens]
        text = " ".join(tokens)

    return text.strip()


class ExactMatchMetric(Metric):
    """Exact match metric with normalization.

    Follows SQuAD evaluation best practices:
    - Case-insensitive matching
    - Removes articles (a, an, the)
    - Removes punctuation
    - Normalizes whitespace

    Optional:
    - Stemming for more aggressive normalization
    """

    def __init__(self, normalize: bool = True, use_stemming: bool = False, **kwargs: Any):
        """Initialize exact match metric.

        Args:
            normalize: Whether to normalize text before comparison (default: True)
            use_stemming: Use Porter stemmer for aggressive normalization (default: False)
            **kwargs: Additional configuration
        """
        super().__init__(name="exact_match", **kwargs)
        self.normalize = normalize
        self.use_stemming = use_stemming

        # Lazy import stemmer only if needed
        self.stemmer = None
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer

                self.stemmer = PorterStemmer()
            except ImportError:
                print("⚠️  NLTK not installed. Stemming disabled.")
                print("   Install with: pip install nltk")
                self.use_stemming = False

    def compute(
        self, predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute exact match score.

        Returns:
            Dict with exact_match score
        """
        scores = []

        for pred, ref in zip(predictions, references):
            # Handle multiple references
            refs = [ref] if isinstance(ref, str) else ref

            if self.normalize:
                pred_norm = normalize_answer(
                    pred, stemmer=self.stemmer if self.use_stemming else None
                )
                refs_norm = [
                    normalize_answer(r, stemmer=self.stemmer if self.use_stemming else None)
                    for r in refs
                ]
            else:
                pred_norm = pred
                refs_norm = refs

            # Check if prediction matches any reference
            score = 1.0 if any(pred_norm == r for r in refs_norm) else 0.0
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"exact_match": avg_score}


class F1Metric(Metric):
    """Token-level F1 metric with normalization.

    Computes F1 score at the token level following SQuAD practices:
    - Normalizes text before tokenization
    - Uses bag-of-words token matching
    - Takes maximum F1 when multiple references

    This is more lenient than exact match and handles:
    - Word order differences
    - Partial matches
    - Extra/missing words
    """

    def __init__(self, normalize: bool = True, use_stemming: bool = False, **kwargs: Any):
        """Initialize F1 metric.

        Args:
            normalize: Whether to normalize text before tokenization (default: True)
            use_stemming: Use Porter stemmer for aggressive normalization (default: False)
            **kwargs: Additional configuration
        """
        super().__init__(name="f1", **kwargs)
        self.normalize = normalize
        self.use_stemming = use_stemming

        # Lazy import stemmer only if needed
        self.stemmer = None
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer

                self.stemmer = PorterStemmer()
            except ImportError:
                print("⚠️  NLTK not installed. Stemming disabled.")
                print("   Install with: pip install nltk")
                self.use_stemming = False

    def compute(
        self, predictions: List[str], references: Union[List[str], List[List[str]]], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute F1 score.

        Returns:
            Dict with f1 score
        """
        scores = []

        for pred, ref in zip(predictions, references):
            # Handle multiple references - take max F1
            refs = [ref] if isinstance(ref, str) else ref

            f1_scores = [self._compute_f1(pred, r) for r in refs]
            scores.append(max(f1_scores))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"f1": avg_score}

    def _compute_f1(self, prediction: str, reference: str) -> float:
        """Compute F1 between prediction and single reference.

        Args:
            prediction: Predicted answer
            reference: Reference answer

        Returns:
            F1 score between 0 and 1
        """
        # Normalize text
        if self.normalize:
            pred_norm = normalize_answer(
                prediction, stemmer=self.stemmer if self.use_stemming else None
            )
            ref_norm = normalize_answer(
                reference, stemmer=self.stemmer if self.use_stemming else None
            )
        else:
            pred_norm = prediction
            ref_norm = reference

        # Tokenize
        pred_tokens = pred_norm.split()
        ref_tokens = ref_norm.split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 1.0 if len(pred_tokens) == len(ref_tokens) else 0.0

        # Compute token overlap using bag-of-words
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1
