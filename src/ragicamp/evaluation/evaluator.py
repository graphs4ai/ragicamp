"""Evaluator for RAG systems.

Provides a clean API for evaluating RAG agents on datasets with multiple metrics.

Example:
    evaluator = Evaluator(agent=agent, dataset=dataset, metrics=[F1Metric(), ExactMatchMetric()])
    results = evaluator.evaluate(batch_size=8)
"""

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from ragicamp.agents.base import RAGAgent
from ragicamp.core.logging import get_logger

logger = get_logger(__name__)
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric
from ragicamp.utils.paths import ensure_dir


class Evaluator:
    """Evaluator for RAG agents.

    Runs evaluation on a dataset and computes multiple metrics.
    Supports batch processing and checkpointing.
    """

    def __init__(
        self,
        agent: RAGAgent,
        dataset: QADataset,
        metrics: Optional[List[Metric]] = None,
    ):
        """Initialize evaluator.

        Args:
            agent: The RAG agent to evaluate
            dataset: Evaluation dataset
            metrics: List of metrics to compute
        """
        self.agent = agent
        self.dataset = dataset
        self.metrics = metrics or []

    def evaluate(
        self,
        num_examples: Optional[int] = None,
        batch_size: int = 1,
        save_predictions: bool = False,
        output_path: Optional[str] = None,
        checkpoint_every: int = 50,
        resume_from_checkpoint: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate the agent on the dataset.

        Args:
            num_examples: Evaluate on first N examples (None = all)
            batch_size: Number of examples to process in parallel
            save_predictions: Whether to save predictions
            output_path: Path to save results
            checkpoint_every: Save checkpoint every N examples
            resume_from_checkpoint: Resume from existing checkpoint
            **kwargs: Additional parameters passed to agent

        Returns:
            Dictionary with metric scores
        """
        examples = list(self.dataset)
        if num_examples:
            examples = examples[:num_examples]

        logger.info("Evaluating %d examples...", len(examples))

        # Generate predictions
        predictions, references, questions = self._generate_predictions(
            examples=examples,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
            checkpoint_path=(
                output_path.replace(".json", "_checkpoint.json") if output_path else None
            ),
            resume=resume_from_checkpoint,
            **kwargs,
        )

        # Unload model before metrics
        self._unload_model()

        # Compute metrics
        results = self._compute_metrics(predictions, references, questions)

        # Add metadata
        results["num_examples"] = len(examples)
        results["agent_name"] = self.agent.name
        results["dataset_name"] = self.dataset.name

        # Save if requested
        if save_predictions and output_path:
            self._save_results(examples, predictions, results, output_path)

        return results

    def _generate_predictions(
        self,
        examples: List[Any],
        batch_size: int,
        checkpoint_every: int,
        checkpoint_path: Optional[str],
        resume: bool,
        **kwargs: Any,
    ) -> tuple:
        """Generate predictions for all examples."""
        predictions = []
        references = []
        questions = []
        start_idx = 0

        # Resume from checkpoint
        if resume and checkpoint_path and Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)
                predictions = data["predictions"]
                references = data["references"]
                questions = data["questions"]
                start_idx = len(predictions)
                logger.info("Resumed from %d/%d", start_idx, len(examples))
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s", e)

        remaining = examples[start_idx:]

        if batch_size > 1 and hasattr(self.agent, "batch_answer"):
            # Batch processing
            for i in tqdm(range(0, len(remaining), batch_size), desc="Batches"):
                batch = remaining[i : i + batch_size]
                queries = [ex.question for ex in batch]

                try:
                    responses = self.agent.batch_answer(queries, **kwargs)
                    for ex, resp in zip(batch, responses):
                        predictions.append(resp.answer)
                        references.append(ex.answers)
                        questions.append(ex.question)
                except Exception as e:
                    for ex in batch:
                        predictions.append(f"[ERROR: {str(e)[:50]}]")
                        references.append(ex.answers)
                        questions.append(ex.question)

                self._checkpoint_if_needed(
                    checkpoint_path, checkpoint_every, predictions, references, questions
                )
                self._clear_cache()
        else:
            # Sequential processing
            for ex in tqdm(remaining, desc="Questions", initial=start_idx, total=len(examples)):
                try:
                    response = self.agent.answer(ex.question, **kwargs)
                    predictions.append(response.answer)
                except Exception as e:
                    predictions.append(f"[ERROR: {str(e)[:50]}]")

                references.append(ex.answers)
                questions.append(ex.question)

                self._checkpoint_if_needed(
                    checkpoint_path, checkpoint_every, predictions, references, questions
                )
                self._clear_cache()

        return predictions, references, questions

    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[Any],
        questions: List[str],
    ) -> Dict[str, Any]:
        """Compute all metrics."""
        logger.info("Computing metrics...")
        results = {}

        for metric in self.metrics:
            logger.debug("Computing %s...", metric.name)
            try:
                if metric.name == "llm_judge":
                    scores = metric.compute(
                        predictions=predictions, references=references, questions=questions
                    )
                else:
                    scores = metric.compute(predictions=predictions, references=references)
                results.update(scores)
            except Exception as e:
                logger.warning("%s failed: %s", metric.name, e)
                results[metric.name] = None

        return results

    def _checkpoint_if_needed(
        self,
        path: Optional[str],
        every: int,
        predictions: List[str],
        references: List[Any],
        questions: List[str],
    ) -> None:
        """Save checkpoint if conditions are met."""
        if not path or not every:
            return
        if len(predictions) % every != 0:
            return

        ensure_dir(path)
        temp = str(path) + ".tmp"
        with open(temp, "w") as f:
            json.dump(
                {"predictions": predictions, "references": references, "questions": questions},
                f,
            )
        Path(temp).replace(path)

    def generate_predictions(
        self,
        output_path: str,
        num_examples: Optional[int] = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> str:
        """Generate predictions only (Phase 1 of two-phase evaluation).

        Saves predictions to disk without computing metrics. Useful for
        separating expensive generation from metric computation.

        Args:
            output_path: Path to save predictions
            num_examples: Limit number of examples
            batch_size: Batch size for generation
            **kwargs: Additional generation arguments

        Returns:
            Path to the saved predictions file
        """
        from datetime import datetime

        examples = list(self.dataset)
        if num_examples is not None:
            examples = examples[:num_examples]

        # Generate predictions
        predictions, references, questions = self._generate_predictions(
            examples=examples,
            batch_size=batch_size,
            checkpoint_every=0,
            checkpoint_path=None,
            resume=False,
            **kwargs,
        )

        # Build output data
        output_dir = Path(output_path).parent
        ensure_dir(output_dir)

        pred_file = str(output_path).replace(".json", "_predictions_raw.json")
        pred_data = {
            "agent_name": self.agent.name,
            "dataset_name": self.dataset.name,
            "num_examples": len(examples),
            "status": "predictions_only",
            "timestamp": datetime.now().isoformat(),
            "predictions": [
                {
                    "question_id": ex.id,
                    "question": ex.question,
                    "prediction": pred,
                    "expected_answers": ref,
                }
                for ex, pred, ref in zip(examples, predictions, references)
            ],
        }

        with open(pred_file, "w") as f:
            json.dump(pred_data, f, indent=2)

        # Also save questions file
        questions_file = output_dir / f"{self.dataset.name}_questions.json"
        questions_data = {
            "dataset_name": self.dataset.name,
            "num_questions": len(examples),
            "questions": [
                {
                    "id": ex.id,
                    "question": ex.question,
                    "expected_answer": ex.answers[0] if ex.answers else "",
                    "all_acceptable_answers": ex.answers,
                }
                for ex in examples
            ],
        }
        with open(questions_file, "w") as f:
            json.dump(questions_data, f, indent=2)

        return pred_file

    def _unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if hasattr(self.agent, "model") and hasattr(self.agent.model, "unload"):
            self.agent.model.unload()
        self._clear_cache()
        logger.debug("Model unloaded")

    def _clear_cache(self) -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _save_results(
        self,
        examples: List[Any],
        predictions: List[str],
        results: Dict[str, Any],
        output_path: str,
    ) -> None:
        """Save evaluation results."""
        from datetime import datetime

        output_dir = Path(output_path).parent
        ensure_dir(output_dir)

        # Save predictions
        pred_data = {
            "agent": self.agent.name,
            "dataset": self.dataset.name,
            "timestamp": datetime.now().isoformat(),
            "predictions": [
                {
                    "id": ex.id,
                    "question": ex.question,
                    "prediction": pred,
                    "expected": ex.answers,
                }
                for ex, pred in zip(examples, predictions)
            ],
        }
        with open(output_path, "w") as f:
            json.dump(pred_data, f, indent=2)
        logger.debug("Predictions saved: %s", output_path)

        # Save summary
        summary_path = str(output_path).replace(".json", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.debug("Summary saved: %s", summary_path)

    @staticmethod
    def compute_metrics_from_file(
        predictions_path: str,
        metrics: List[Metric],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute metrics from saved predictions file.

        Args:
            predictions_path: Path to predictions JSON
            metrics: List of metrics to compute
            output_path: Optional path to save results

        Returns:
            Dictionary with metric scores
        """
        logger.debug("Loading predictions from: %s", predictions_path)

        with open(predictions_path) as f:
            data = json.load(f)

        preds = data.get("predictions", [])
        predictions = [p["prediction"] for p in preds]
        questions = [p["question"] for p in preds]

        # Try to get references from predictions file
        if preds and "expected" in preds[0]:
            references = [p["expected"] for p in preds]
        else:
            # Look for questions file in same directory
            pred_path = Path(predictions_path)
            questions_files = list(pred_path.parent.glob("*_questions.json"))

            if questions_files:
                logger.info("Loading expected answers from: %s", questions_files[0])
                with open(questions_files[0]) as f:
                    q_data = json.load(f)

                # Build lookup by question_id
                q_lookup = {}
                for q in q_data.get("questions", []):
                    q_lookup[q["id"]] = q.get("expected_answer", q.get("answer", ""))

                # Match predictions to expected answers
                references = []
                for p in preds:
                    qid = p.get("question_id", "")
                    references.append(q_lookup.get(qid, ""))
            else:
                logger.warning("No expected answers found - metrics may be inaccurate")
                references = [""] * len(predictions)

        logger.info("Computing %d metrics on %d predictions...", len(metrics), len(predictions))

        results = {}
        for metric in metrics:
            logger.debug("Computing %s...", metric.name)
            try:
                if metric.name == "llm_judge":
                    scores = metric.compute(
                        predictions=predictions, references=references, questions=questions
                    )
                else:
                    scores = metric.compute(predictions=predictions, references=references)
                results.update(scores)
            except Exception as e:
                logger.warning("%s failed: %s", metric.name, e)

        results["num_examples"] = len(predictions)
        results["predictions_file"] = predictions_path

        # Update per-item metrics in predictions file if metric supports it
        updated_predictions = False
        for metric in metrics:
            if hasattr(metric, "get_per_item_scores"):
                per_item_scores = metric.get_per_item_scores()
                if len(per_item_scores) == len(preds):
                    for i, score in enumerate(per_item_scores):
                        if "metrics" not in preds[i]:
                            preds[i]["metrics"] = {}
                        preds[i]["metrics"][metric.name] = score
                    updated_predictions = True

        # Save updated predictions back to original file
        if updated_predictions:
            with open(predictions_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Updated predictions file with new metrics: %s", predictions_path)

            # Also update summary file if it exists
            pred_path = Path(predictions_path)
            summary_path = pred_path.parent / pred_path.name.replace("_predictions.json", "_summary.json")
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary_data = json.load(f)
                    if "overall_metrics" in summary_data:
                        summary_data["overall_metrics"].update(results)
                    else:
                        summary_data.update(results)
                    with open(summary_path, "w") as f:
                        json.dump(summary_data, f, indent=2)
                    logger.info("Updated summary file: %s", summary_path)
                except Exception as e:
                    logger.warning("Could not update summary: %s", e)

        if output_path:
            ensure_dir(output_path)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.debug("Summary saved: %s", output_path)

        return results
