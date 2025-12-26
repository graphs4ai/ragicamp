"""Enhanced LLM-as-a-judge for QA evaluation with categorical judgments.

This module provides async-capable LLM-as-a-judge metrics for evaluating
QA systems. Uses parallel async API calls for efficient batch evaluation.

Example:
    >>> from ragicamp.models import OpenAIModel
    >>> from ragicamp.metrics import LLMJudgeQAMetric
    >>>
    >>> judge = OpenAIModel("gpt-4o-mini")
    >>> metric = LLMJudgeQAMetric(judge_model=judge, judgment_type="binary")
    >>>
    >>> # Sync usage (internally uses async)
    >>> scores = metric.compute(predictions, references, questions)
    >>>
    >>> # Async usage (for integration with async pipelines)
    >>> scores = await metric.acompute(predictions, references, questions)
"""

import re
from typing import Any, Dict, List, Optional, Union

from ragicamp.metrics.async_base import AsyncAPIMetric
from ragicamp.models.base import LanguageModel


class LLMJudgeQAMetric(AsyncAPIMetric):
    """LLM-as-a-judge for QA with categorical evaluation (correct/partial/incorrect).

    Uses GPT-4 or similar to evaluate answer quality with clear categorical judgments.
    Particularly useful when standard metrics don't capture semantic correctness.

    Inherits from AsyncAPIMetric for efficient parallel async API calls.
    """

    def __init__(
        self,
        judge_model: LanguageModel,
        judgment_type: str = "binary",  # "binary" or "ternary"
        max_concurrent: int = 20,  # Number of concurrent API calls
        show_progress: bool = True,
        **kwargs: Any,
    ):
        """Initialize LLM judge for QA evaluation.

        Args:
            judge_model: The LLM to use as judge (e.g., GPT-4, must have agenerate_single)
            judgment_type: Type of judgment
                - "binary": correct (1.0) or incorrect (0.0)
                - "ternary": correct (1.0), partial (0.5), or incorrect (0.0)
            max_concurrent: Maximum concurrent API calls (default: 20)
            show_progress: Show progress bar during computation
            **kwargs: Additional configuration
        """
        super().__init__(
            name="llm_judge_qa",
            max_concurrent=max_concurrent,
            show_progress=show_progress,
            **kwargs,
        )
        self.judge_model = judge_model
        self.judgment_type = judgment_type

        # Define categories
        if judgment_type == "binary":
            self.categories = ["correct", "incorrect"]
        else:  # ternary
            self.categories = ["correct", "partially_correct", "incorrect"]

    async def acompute_single(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        question: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute judgment for a single prediction-reference pair (async).

        Args:
            prediction: Predicted answer
            reference: Reference answer(s)
            question: Question for context (highly recommended)
            **kwargs: Additional parameters

        Returns:
            Dict with score and category info
        """
        # Handle multiple references
        refs = [reference] if isinstance(reference, str) else reference

        # Create judgment prompt
        prompt = self._create_judgment_prompt(
            question=question,
            prediction=prediction,
            references=refs,
        )

        # Call LLM asynchronously
        if hasattr(self.judge_model, "agenerate_single"):
            judgment = await self.judge_model.agenerate_single(
                prompt,
                temperature=0.0,
                max_tokens=200,
            )
        else:
            # Fallback to sync if async not available
            import asyncio

            loop = asyncio.get_event_loop()
            judgment = await loop.run_in_executor(
                None, lambda: self.judge_model.generate(prompt, temperature=0.0, max_tokens=200)
            )

        # Extract categorical judgment
        category, score = self._extract_judgment(judgment)

        return {
            "llm_judge_qa": score,
            f"llm_judge_qa_{category}": 1.0,  # For category counting
        }

    def _aggregate_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate individual results with category statistics.

        Args:
            results: List of per-item result dictionaries

        Returns:
            Aggregated metrics including category proportions
        """
        if not results:
            return {"llm_judge_qa": 0.0}

        total = len(results)

        # Sum scores
        avg_score = sum(r.get("llm_judge_qa", 0.0) for r in results) / total

        # Count categories
        correct_count = sum(1 for r in results if r.get("llm_judge_qa_correct", 0) > 0)
        incorrect_count = sum(1 for r in results if r.get("llm_judge_qa_incorrect", 0) > 0)

        aggregated = {
            "llm_judge_qa": avg_score,
            "llm_judge_qa_correct": correct_count / total,
            "llm_judge_qa_incorrect": incorrect_count / total,
        }

        # Add partial category for ternary
        if self.judgment_type == "ternary":
            partial_count = sum(
                1 for r in results if r.get("llm_judge_qa_partially_correct", 0) > 0
            )
            aggregated["llm_judge_qa_partial"] = partial_count / total

        return aggregated

    def _create_judgment_prompt(self, question: str, prediction: str, references: List[str]) -> str:
        """Create a clear prompt for categorical QA judgment."""

        if self.judgment_type == "binary":
            categories_desc = """
- CORRECT: The prediction accurately answers the question with the same information as the reference
- INCORRECT: The prediction is wrong, incomplete, or doesn't match the reference
"""
            output_format = "First line: JUDGMENT: [CORRECT/INCORRECT]"
        else:  # ternary
            categories_desc = """
- CORRECT: The prediction accurately answers the question with the same core information as the reference
- PARTIALLY_CORRECT: The prediction contains the right information but with extra/missing details, or is close but not exact
- INCORRECT: The prediction is fundamentally wrong or completely misses the reference answer
"""
            output_format = "First line: JUDGMENT: [CORRECT/PARTIALLY_CORRECT/INCORRECT]"

        # Build reference answers section
        ref_section = "\n".join([f"  - {ref}" for ref in references])

        prompt = f"""You are an expert evaluator for question-answering systems. Evaluate if the predicted answer is correct compared to the reference answer(s).

Question: {question}

Reference Answer(s):
{ref_section}

Predicted Answer: {prediction}

Task: Determine if the predicted answer is semantically correct compared to the reference answer(s).

Categories:
{categories_desc}

Instructions:
1. Focus on semantic correctness, not exact wording
2. Consider that answers may be phrased differently but still correct
3. For dates, numbers, names: small variations matter (e.g., "1776" vs "1777" is incorrect)
4. For descriptive answers: core meaning should match

Output Format:
{output_format}
Second line: Brief 1-sentence explanation

Your evaluation:"""

        return prompt

    def _extract_judgment(self, judgment_text: str) -> tuple:
        """Extract categorical judgment and convert to score.

        Returns:
            Tuple of (category, score) where score is 0.0, 0.5, or 1.0
        """
        judgment_lower = judgment_text.lower()

        # Look for judgment in text
        if (
            "judgment:" in judgment_lower
            or "correct" in judgment_lower
            or "incorrect" in judgment_lower
        ):
            # Check for categories in order of specificity
            if (
                "partially_correct" in judgment_lower
                or "partially correct" in judgment_lower
                or "partial" in judgment_lower
            ):
                # For binary mode, map partially_correct to incorrect (conservative)
                if self.judgment_type == "binary":
                    return ("incorrect", 0.0)
                else:
                    return ("partially_correct", 0.5)
            elif re.search(r"\bcorrect\b", judgment_lower) and not re.search(
                r"\bincorrect\b", judgment_lower
            ):
                return ("correct", 1.0)
            elif "incorrect" in judgment_lower:
                return ("incorrect", 0.0)

        # Fallback: look for explicit CORRECT/INCORRECT markers
        if re.search(r"(?:^|\s)correct(?:\s|$|[:\.])", judgment_lower):
            return ("correct", 1.0)
        elif re.search(r"(?:^|\s)incorrect(?:\s|$|[:\.])", judgment_lower):
            return ("incorrect", 0.0)

        # Default to incorrect if can't parse (conservative)
        return ("incorrect", 0.0)


# Convenience functions for creating pre-configured judges


def create_binary_judge(
    judge_model: LanguageModel,
    max_concurrent: int = 20,
) -> LLMJudgeQAMetric:
    """Create a binary LLM judge (correct/incorrect).

    Args:
        judge_model: GPT-4 or similar model
        max_concurrent: Maximum concurrent API calls

    Returns:
        Configured LLMJudgeQAMetric
    """
    return LLMJudgeQAMetric(
        judge_model=judge_model,
        judgment_type="binary",
        max_concurrent=max_concurrent,
    )


def create_ternary_judge(
    judge_model: LanguageModel,
    max_concurrent: int = 20,
) -> LLMJudgeQAMetric:
    """Create a ternary LLM judge (correct/partial/incorrect).

    Args:
        judge_model: GPT-4 or similar model
        max_concurrent: Maximum concurrent API calls

    Returns:
        Configured LLMJudgeQAMetric
    """
    return LLMJudgeQAMetric(
        judge_model=judge_model,
        judgment_type="ternary",
        max_concurrent=max_concurrent,
    )
