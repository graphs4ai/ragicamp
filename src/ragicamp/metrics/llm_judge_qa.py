"""Enhanced LLM-as-a-judge for QA evaluation with categorical judgments."""

import re
from typing import Any, Dict, List, Union, Optional

from ragicamp.metrics.base import Metric
from ragicamp.models.base import LanguageModel


class LLMJudgeQAMetric(Metric):
    """LLM-as-a-judge for QA with categorical evaluation (correct/partial/incorrect).
    
    Uses GPT-4 or similar to evaluate answer quality with clear categorical judgments.
    Particularly useful when standard metrics don't capture semantic correctness.
    """
    
    def __init__(
        self,
        judge_model: LanguageModel,
        judgment_type: str = "binary",  # "binary" or "ternary"
        **kwargs: Any
    ):
        """Initialize LLM judge for QA evaluation.
        
        Args:
            judge_model: The LLM to use as judge (e.g., GPT-4)
            judgment_type: Type of judgment
                - "binary": correct (1.0) or incorrect (0.0)
                - "ternary": correct (1.0), partial (0.5), or incorrect (0.0)
            **kwargs: Additional configuration
        """
        super().__init__(name="llm_judge_qa", **kwargs)
        self.judge_model = judge_model
        self.judgment_type = judgment_type
        
        # Define categories
        if judgment_type == "binary":
            self.categories = ["correct", "incorrect"]
        else:  # ternary
            self.categories = ["correct", "partially_correct", "incorrect"]
    
    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        questions: List[str] = None,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Compute LLM judge scores.
        
        Args:
            predictions: Predicted answers
            references: Reference answers
            questions: Questions for context (highly recommended)
            **kwargs: Additional parameters
            
        Returns:
            Dict with:
            - llm_judge_qa: Average score (0.0-1.0)
            - llm_judge_qa_correct: Proportion marked as correct
            - llm_judge_qa_partial: Proportion marked as partially correct (ternary only)
            - llm_judge_qa_incorrect: Proportion marked as incorrect
        """
        scores = []
        categories_count = {cat: 0 for cat in self.categories}
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Handle multiple references
            refs = [ref] if isinstance(ref, str) else ref
            question = questions[i] if questions and i < len(questions) else None
            
            # Create judgment prompt
            prompt = self._create_judgment_prompt(
                question=question,
                prediction=pred,
                references=refs
            )
            
            # Get judgment with temperature=0 for consistency
            judgment = self.judge_model.generate(prompt, temperature=0.0, max_tokens=200)
            
            # Extract categorical judgment
            category, score = self._extract_judgment(judgment)
            scores.append(score)
            
            # Safely increment category count (handle unexpected categories from LLM)
            if category in categories_count:
                categories_count[category] += 1
            else:
                # LLM returned unexpected category - treat as incorrect
                print(f"Warning: LLM returned unexpected category '{category}' (expected {self.categories}). Treating as 'incorrect'.")
                categories_count["incorrect"] += 1
                scores[-1] = 0.0  # Override score to 0.0 for safety
        
        # Compute metrics
        total = len(scores)
        avg_score = sum(scores) / total if total > 0 else 0.0
        
        results = {
            "llm_judge_qa": avg_score,
            "llm_judge_qa_correct": categories_count.get("correct", 0) / total if total > 0 else 0.0,
            "llm_judge_qa_incorrect": categories_count.get("incorrect", 0) / total if total > 0 else 0.0,
        }
        
        # Add partial category for ternary
        if self.judgment_type == "ternary":
            results["llm_judge_qa_partial"] = categories_count.get("partially_correct", 0) / total if total > 0 else 0.0
        
        return results
    
    def _create_judgment_prompt(
        self,
        question: str,
        prediction: str,
        references: List[str]
    ) -> str:
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
    
    def _extract_judgment(self, judgment_text: str) -> tuple[str, float]:
        """Extract categorical judgment and convert to score.
        
        Returns:
            Tuple of (category, score) where score is 0.0, 0.5, or 1.0
        """
        judgment_lower = judgment_text.lower()
        
        # Look for judgment in text
        if "judgment:" in judgment_lower or "correct" in judgment_lower or "incorrect" in judgment_lower:
            # Check for categories in order of specificity
            if "partially_correct" in judgment_lower or "partially correct" in judgment_lower or "partial" in judgment_lower:
                # For binary mode, map partially_correct to incorrect (conservative)
                if self.judgment_type == "binary":
                    return ("incorrect", 0.0)
                else:
                    return ("partially_correct", 0.5)
            elif re.search(r'\bcorrect\b', judgment_lower) and not re.search(r'\bincorrect\b', judgment_lower):
                return ("correct", 1.0)
            elif "incorrect" in judgment_lower:
                return ("incorrect", 0.0)
        
        # Fallback: look for explicit CORRECT/INCORRECT markers
        if re.search(r'(?:^|\s)correct(?:\s|$|[:\.])', judgment_lower):
            return ("correct", 1.0)
        elif re.search(r'(?:^|\s)incorrect(?:\s|$|[:\.])', judgment_lower):
            return ("incorrect", 0.0)
        
        # Default to incorrect if can't parse (conservative)
        print(f"⚠️ Warning: Could not parse LLM judgment: '{judgment_text[:100]}...'")
        return ("incorrect", 0.0)
    
    def compute_single(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        question: str = None,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Compute metric for a single prediction-reference pair.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer(s)
            question: Question for context
            **kwargs: Additional parameters
            
        Returns:
            Dict with score
        """
        predictions = [prediction]
        references = [reference] if isinstance(reference, str) else [reference]
        questions = [question] if question else None
        
        return self.compute(predictions, references, questions, **kwargs)


# Convenience functions for creating pre-configured judges

def create_binary_judge(judge_model: LanguageModel) -> LLMJudgeQAMetric:
    """Create a binary LLM judge (correct/incorrect).
    
    Args:
        judge_model: GPT-4 or similar model
        
    Returns:
        Configured LLMJudgeQAMetric
    """
    return LLMJudgeQAMetric(judge_model=judge_model, judgment_type="binary")


def create_ternary_judge(judge_model: LanguageModel) -> LLMJudgeQAMetric:
    """Create a ternary LLM judge (correct/partial/incorrect).
    
    Args:
        judge_model: GPT-4 or similar model
        
    Returns:
        Configured LLMJudgeQAMetric
    """
    return LLMJudgeQAMetric(judge_model=judge_model, judgment_type="ternary")

