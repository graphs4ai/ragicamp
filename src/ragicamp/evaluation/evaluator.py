"""Evaluator for RAG systems."""

import json
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ragicamp.agents.base import RAGAgent
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric
from ragicamp.utils.paths import ensure_dir


class Evaluator:
    """Evaluator for RAG agents.

    Runs evaluation on a dataset and computes multiple metrics.
    """

    def __init__(self, agent: RAGAgent, dataset: QADataset, metrics: List[Metric], **kwargs: Any):
        """Initialize evaluator.

        Args:
            agent: The RAG agent to evaluate
            dataset: Evaluation dataset
            metrics: List of metrics to compute
            **kwargs: Additional configuration
        """
        self.agent = agent
        self.dataset = dataset
        self.metrics = metrics
        self.config = kwargs

    def evaluate(
        self,
        num_examples: Optional[int] = None,
        save_predictions: bool = False,
        output_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate the agent on the dataset.

        Args:
            num_examples: Evaluate on first N examples (None = all)
            save_predictions: Whether to save predictions
            output_path: Path to save results
            batch_size: Number of examples to process in parallel (None = sequential)
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary with metric scores and statistics
        """
        # Prepare examples
        examples = list(self.dataset)
        if num_examples is not None:
            examples = examples[:num_examples]

        # Generate predictions
        predictions = []
        references = []
        questions = []
        responses = []

        print(f"Evaluating on {len(examples)} examples...")

        # Use batch processing if batch_size is specified
        if batch_size and batch_size > 1:
            print(f"Using batch processing with batch_size={batch_size}")
            print(f"Total batches: {(len(examples) + batch_size - 1) // batch_size}")

            import time

            batch_times = []

            # Process in batches
            for i in tqdm(range(0, len(examples), batch_size), desc="Processing batches"):
                batch_start = time.time()

                batch_examples = examples[i : i + batch_size]
                batch_queries = [ex.question for ex in batch_examples]

                # Batch generate answers
                batch_responses = self.agent.batch_answer(batch_queries, **kwargs)

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                # Store results
                for example, response in zip(batch_examples, batch_responses):
                    predictions.append(response.answer)
                    references.append(example.answers)
                    questions.append(example.question)
                    responses.append(response)

            # Print timing stats
            if batch_times:
                avg_time = sum(batch_times) / len(batch_times)
                print(f"\nBatch processing stats:")
                print(f"  Average time per batch: {avg_time:.2f}s")
                print(f"  Average time per question: {avg_time / batch_size:.2f}s")
                print(f"  Throughput: {batch_size / avg_time:.2f} questions/second")
        else:
            # Sequential processing (original behavior)
            for example in tqdm(examples, desc="Generating answers"):
                # Generate answer
                response = self.agent.answer(example.question)

                # Store
                predictions.append(response.answer)
                references.append(example.answers)
                questions.append(example.question)
                responses.append(response)

        # Compute metrics
        print("\nComputing metrics...")
        results = {}

        for metric in self.metrics:
            print(f"  - {metric.name}")

            # Handle metrics that need questions
            if metric.name == "llm_judge":
                scores_dict = metric.compute(
                    predictions=predictions, references=references, questions=questions
                )
            else:
                scores_dict = metric.compute(predictions=predictions, references=references)

            # All metrics now return Dict[str, float]
            results.update(scores_dict)

        # Add statistics
        results["num_examples"] = len(examples)
        results["agent_name"] = self.agent.name
        results["dataset_name"] = self.dataset.name

        # Compute per-question metrics
        per_question_metrics = self._compute_per_question_metrics(
            predictions, references, questions
        )

        # Save if requested
        if save_predictions and output_path:
            self._save_results(
                examples=examples,
                predictions=predictions,
                responses=responses,
                results=results,
                per_question_metrics=per_question_metrics,
                output_path=output_path,
            )

        return results

    def _compute_per_question_metrics(
        self, predictions: List[str], references: List[Any], questions: List[str]
    ) -> List[Dict[str, Any]]:
        """Compute metrics for each individual question.

        Args:
            predictions: List of predictions
            references: List of references
            questions: List of questions

        Returns:
            List of per-question metric scores
        """
        per_question = []

        for i, (pred, ref, q) in enumerate(zip(predictions, references, questions)):
            question_metrics = {"question_index": i, "question": q}

            # Compute each metric individually for this question
            for metric in self.metrics:
                try:
                    score = metric.compute_single(pred, ref)
                    if isinstance(score, dict):
                        # Handle metrics that return multiple scores (like BERTScore)
                        for key, value in score.items():
                            question_metrics[key] = value
                    else:
                        question_metrics[metric.name] = score
                except Exception as e:
                    # If metric fails for this question, record None
                    question_metrics[metric.name] = None

            per_question.append(question_metrics)

        return per_question

    def _compute_metric_statistics(
        self, per_question_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each metric across all questions.

        Args:
            per_question_metrics: List of per-question metrics

        Returns:
            Dict mapping metric names to their statistics
        """
        if not per_question_metrics:
            return {}

        # Get all metric names (excluding metadata fields)
        metric_names = set()
        for item in per_question_metrics:
            for key in item.keys():
                if key not in ["question_index", "question"] and isinstance(
                    item.get(key), (int, float)
                ):
                    metric_names.add(key)

        # Compute statistics for each metric
        stats = {}
        for metric_name in metric_names:
            values = [
                item[metric_name]
                for item in per_question_metrics
                if metric_name in item and item[metric_name] is not None
            ]

            if values:
                stats[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (
                        (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values))
                        ** 0.5
                        if len(values) > 1
                        else 0.0
                    ),
                }

        return stats

    def _save_results(
        self,
        examples: List[Any],
        predictions: List[str],
        responses: List[Any],
        results: Dict[str, Any],
        per_question_metrics: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Save evaluation results in a clean, modular structure.

        Saves three files:
        1. {dataset}_questions.json - Dataset questions and expected answers (reusable)
        2. {agent}_predictions.json - Predictions with per-question metrics
        3. {agent}_summary.json - Overall metrics summary with statistics

        Args:
            examples: Dataset examples
            predictions: Generated predictions
            responses: Full agent responses
            results: Metric scores
            per_question_metrics: Per-question metric scores
            output_path: Base path for output files
        """
        import os
        from datetime import datetime
        from pathlib import Path

        # Determine output directory and base name
        output_dir = Path(output_path).parent
        base_name = Path(output_path).stem

        # Ensure output directory exists
        ensure_dir(output_dir)

        agent_name = results.get("agent_name", "unknown")
        dataset_name = results.get("dataset_name", "unknown")
        timestamp = datetime.now().isoformat()

        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")

        # 1. Save dataset questions (reusable across runs)
        questions_path = output_dir / f"{dataset_name}_questions.json"
        questions_data = {
            "dataset_name": dataset_name,
            "num_questions": len(examples),
            "questions": [
                {
                    "id": ex.id,
                    "question": ex.question,
                    "expected_answer": ex.answers[0] if ex.answers else None,
                    "all_acceptable_answers": ex.answers,
                }
                for ex in examples
            ],
        }

        with open(questions_path, "w") as f:
            json.dump(questions_data, f, indent=2)

        print(f"✓ Dataset questions: {questions_path}")

        # 2. Save predictions with per-question metrics
        predictions_path = output_dir / f"{agent_name}_predictions.json"
        predictions_data = {
            "agent_name": agent_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "num_examples": len(examples),
            "predictions": [
                {
                    "question_id": ex.id,
                    "question": ex.question,
                    "prediction": pred,
                    "metrics": {
                        k: v for k, v in metrics.items() if k not in ["question_index", "question"]
                    },
                    "metadata": resp.metadata if hasattr(resp, "metadata") else {},
                }
                for ex, pred, resp, metrics in zip(
                    examples, predictions, responses, per_question_metrics
                )
            ],
        }

        with open(predictions_path, "w") as f:
            json.dump(predictions_data, f, indent=2)

        print(f"✓ Predictions + metrics: {predictions_path}")

        # 3. Save overall summary with statistics
        summary_path = output_dir / f"{agent_name}_summary.json"
        summary_data = {
            "agent_name": agent_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "num_examples": results.get("num_examples", 0),
            "overall_metrics": {
                k: v
                for k, v in results.items()
                if k not in ["num_examples", "agent_name", "dataset_name"]
            },
            "metric_statistics": self._compute_metric_statistics(per_question_metrics),
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"✓ Summary: {summary_path}")

        print(f"{'='*70}\n")

    def compare_agents(
        self, agents: List[RAGAgent], num_examples: Optional[int] = None, **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple agents on the same dataset.

        Args:
            agents: List of agents to compare
            num_examples: Number of examples to evaluate
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping agent names to their results
        """
        all_results = {}

        for agent in agents:
            print(f"\n{'='*60}")
            print(f"Evaluating agent: {agent.name}")
            print(f"{'='*60}")

            # Temporarily set agent
            original_agent = self.agent
            self.agent = agent

            # Evaluate
            results = self.evaluate(num_examples=num_examples, **kwargs)
            all_results[agent.name] = results

            # Restore original agent
            self.agent = original_agent

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")

        for agent_name, results in all_results.items():
            print(f"\n{agent_name}:")
            for metric_name, score in results.items():
                if metric_name not in ["num_examples", "agent_name", "dataset_name"]:
                    print(
                        f"  {metric_name}: {score:.4f}"
                        if isinstance(score, float)
                        else f"  {metric_name}: {score}"
                    )

        return all_results
