"""LLM-as-a-judge metric implementation."""

from typing import Any, Dict, List, Union

from ragicamp.metrics.base import Metric
from ragicamp.models.base import LanguageModel


class LLMJudgeMetric(Metric):
    """Use an LLM to judge answer quality."""

    def __init__(
        self,
        judge_model: LanguageModel,
        criteria: str = "accuracy",
        scale: int = 10,
        **kwargs: Any,
    ):
        """Initialize LLM judge metric.

        Args:
            judge_model: The LLM to use as judge
            criteria: Evaluation criteria (accuracy, completeness, etc.)
            scale: Rating scale (e.g., 1-10)
            **kwargs: Additional configuration
        """
        super().__init__(name="llm_judge", **kwargs)
        self.judge_model = judge_model
        self.criteria = criteria
        self.scale = scale

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        questions: List[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute LLM judge scores.

        Args:
            predictions: Predicted answers
            references: Reference answers
            questions: Optional questions for context
            **kwargs: Additional parameters

        Returns:
            Dict with average score and individual scores
        """
        scores = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Handle multiple references
            refs = [ref] if isinstance(ref, str) else ref
            question = questions[i] if questions and i < len(questions) else None

            # Create judgment prompt
            prompt = self._create_judgment_prompt(
                prediction=pred, references=refs, question=question
            )

            # Get judgment
            judgment = self.judge_model.generate(prompt, temperature=0.0)

            # Extract score
            score = self._extract_score(judgment)
            scores.append(score)

        return {
            "llm_judge_score": sum(scores) / len(scores) if scores else 0.0,
            "individual_scores": scores,
        }

    def _create_judgment_prompt(
        self, prediction: str, references: List[str], question: str = None
    ) -> str:
        """Create a prompt for the judge LLM."""
        prompt_parts = [
            f"Evaluate the following answer based on {self.criteria}.",
            f"Rate on a scale of 1-{self.scale}, where {self.scale} is perfect.",
        ]

        if question:
            prompt_parts.append(f"\nQuestion: {question}")

        prompt_parts.append(f"\nReference Answer(s):")
        for i, ref in enumerate(references, 1):
            prompt_parts.append(f"{i}. {ref}")

        prompt_parts.append(f"\nCandidate Answer: {prediction}")
        prompt_parts.append(f"\nProvide your rating (1-{self.scale}) and brief justification:")

        return "\n".join(prompt_parts)

    def _extract_score(self, judgment: str) -> float:
        """Extract numerical score from judgment text."""
        import re

        # Look for patterns like "Rating: 8" or "Score: 7/10" or just a number
        patterns = [
            r"rating:\s*(\d+)",
            r"score:\s*(\d+)",
            r"(\d+)\s*/\s*" + str(self.scale),
            r"^(\d+)",  # Number at start
        ]

        for pattern in patterns:
            match = re.search(pattern, judgment.lower())
            if match:
                score = float(match.group(1))
                return min(score / self.scale, 1.0)  # Normalize to 0-1

        # Default to 0 if can't parse
        return 0.0
