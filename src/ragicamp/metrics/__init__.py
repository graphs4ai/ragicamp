"""Evaluation metrics for RAG systems."""

from ragicamp.metrics.base import Metric

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
    from ragicamp.metrics.llm_judge import LLMJudgeMetric

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

__all__ = ["Metric"]

# Add available metrics to __all__
if _has_exact_match:
    __all__.extend(["ExactMatchMetric", "F1Metric"])
if _has_bertscore:
    __all__.append("BERTScoreMetric")
if _has_bleurt:
    __all__.append("BLEURTMetric")
if _has_llm_judge:
    __all__.append("LLMJudgeMetric")
if _has_faithfulness:
    __all__.append("FaithfulnessMetric")
if _has_hallucination:
    __all__.append("HallucinationMetric")
