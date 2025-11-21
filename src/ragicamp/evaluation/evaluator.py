"""Evaluator for RAG systems.

The evaluator works in two phases for robustness:
1. Generate predictions and save them (never lose progress)
2. Compute metrics on saved predictions (can retry/recompute)
"""

import json
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ragicamp.agents.base import RAGAgent
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric
from ragicamp.utils.paths import ensure_dir


class Evaluator:
    """Evaluator for RAG agents.
    
    Works in two phases:
    1. generate_predictions() - Generate and save predictions
    2. compute_metrics() - Compute metrics on saved predictions
    
    This two-phase approach ensures you never lose predictions due to
    metrics computation failures (API errors, crashes, etc).
    """
    
    def __init__(
        self,
        agent: RAGAgent,
        dataset: QADataset,
        metrics: Optional[List[Metric]] = None,
        **kwargs: Any
    ):
        """Initialize evaluator.
        
        Args:
            agent: The RAG agent to evaluate
            dataset: Evaluation dataset
            metrics: List of metrics to compute (optional, can add later)
            **kwargs: Additional configuration
        """
        self.agent = agent
        self.dataset = dataset
        self.metrics = metrics or []
        self.config = kwargs
    
    def generate_predictions(
        self,
        output_path: str,
        num_examples: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate predictions and save them (Phase 1).
        
        This generates predictions for the dataset and saves them to a JSON file.
        Metrics are NOT computed in this phase - use compute_metrics() afterward.
        
        Args:
            output_path: Path to save predictions (e.g., "outputs/predictions.json")
            num_examples: Generate for first N examples (None = all)
            batch_size: Number of examples to process in parallel (None = sequential)
            **kwargs: Additional generation parameters
            
        Returns:
            Path to saved predictions file
        """
        # Prepare examples
        examples = list(self.dataset)
        if num_examples is not None:
            examples = examples[:num_examples]
        
        # Generate predictions
        predictions = []
        responses = []
        
        print(f"\n{'='*70}")
        print(f"PHASE 1: GENERATING PREDICTIONS")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset.name}")
        print(f"Agent: {self.agent.name}")
        print(f"Examples: {len(examples)}")
        print(f"{'='*70}\n")
        
        # Use batch processing if batch_size is specified
        if batch_size and batch_size > 1:
            print(f"Using batch processing with batch_size={batch_size}")
            print(f"Total batches: {(len(examples) + batch_size - 1) // batch_size}\n")
            
            import time
            batch_times = []
            
            # Process in batches
            for i in tqdm(range(0, len(examples), batch_size), desc="Generating predictions"):
                batch_start = time.time()
                
                batch_examples = examples[i:i+batch_size]
                batch_queries = [ex.question for ex in batch_examples]
                
                # Batch generate answers
                batch_responses = self.agent.batch_answer(batch_queries, **kwargs)
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Store results
                for example, response in zip(batch_examples, batch_responses):
                    predictions.append(response.answer)
                    responses.append(response)
            
            # Print timing stats
            if batch_times:
                avg_time = sum(batch_times) / len(batch_times)
                print(f"\n📊 Generation stats:")
                print(f"  Average time per batch: {avg_time:.2f}s")
                print(f"  Average time per question: {avg_time / batch_size:.2f}s")
                print(f"  Throughput: {batch_size / avg_time:.2f} questions/second")
        else:
            # Sequential processing
            for example in tqdm(examples, desc="Generating predictions"):
                response = self.agent.answer(example.question, **kwargs)
                predictions.append(response.answer)
                responses.append(response)
        
        # Save predictions
        print(f"\n{'='*70}")
        print("💾 SAVING PREDICTIONS")
        print(f"{'='*70}")
        predictions_file = self._save_predictions_only(
            examples=examples,
            predictions=predictions,
            responses=responses,
            output_path=output_path
        )
        print(f"✓ Predictions saved!")
        print(f"\n💡 Next step: Compute metrics")
        print(f"   python scripts/compute_metrics.py --predictions {predictions_file}")
        print(f"{'='*70}\n")
        
        return predictions_file
    
    def _compute_per_question_metrics(
        self,
        predictions: List[str],
        references: List[Any],
        questions: List[str]
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
        self,
        per_question_metrics: List[Dict[str, Any]]
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
                if key not in ['question_index', 'question'] and isinstance(item.get(key), (int, float)):
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
                    "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
                    if len(values) > 1 else 0.0
                }
        
        return stats
    
    def _save_predictions_only(
        self,
        examples: List[Any],
        predictions: List[str],
        responses: List[Any],
        output_path: str
    ) -> str:
        """Save predictions (before metrics computation).
        
        Saves two files:
        1. {dataset}_questions.json - Dataset questions and expected answers
        2. {agent}_predictions_raw.json - Raw predictions without metrics
        
        Args:
            examples: Dataset examples
            predictions: Generated predictions
            responses: Full agent responses
            output_path: Base path for output files
            
        Returns:
            Path to the saved predictions file
        """
        from datetime import datetime
        from pathlib import Path
        
        # Determine output directory
        output_dir = Path(output_path).parent
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Get names
        dataset_name = self.dataset.name if hasattr(self.dataset, "name") else "unknown"
        timestamp = datetime.now().isoformat()
        
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
            ]
        }
        
        with open(questions_path, 'w') as f:
            json.dump(questions_data, f, indent=2)
        
        print(f"  Dataset questions: {questions_path}")
        
        # 2. Save raw predictions (without metrics)
        predictions_path = output_dir / f"{self.agent.name}_predictions_raw.json"
        predictions_data = {
            "agent_name": self.agent.name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "num_examples": len(examples),
            "status": "predictions_only",
            "predictions": [
                {
                    "question_id": ex.id,
                    "question": ex.question,
                    "expected_answers": ex.answers,
                    "prediction": pred,
                    "metadata": resp.metadata if hasattr(resp, "metadata") else {}
                }
                for ex, pred, resp in zip(examples, predictions, responses)
            ]
        }
        
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        print(f"  Predictions: {predictions_path}")
        
        return str(predictions_path)
    
    def _save_results(
        self,
        examples: List[Any],
        predictions: List[str],
        responses: List[Any],
        results: Dict[str, Any],
        per_question_metrics: List[Dict[str, Any]],
        output_path: str
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
            ]
        }
        
        with open(questions_path, 'w') as f:
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
                        k: v for k, v in metrics.items()
                        if k not in ['question_index', 'question']
                    },
                    "metadata": resp.metadata if hasattr(resp, "metadata") else {}
                }
                for ex, pred, resp, metrics in zip(
                    examples, predictions, responses, per_question_metrics
                )
            ]
        }
        
        with open(predictions_path, 'w') as f:
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
                k: v for k, v in results.items()
                if k not in ["num_examples", "agent_name", "dataset_name"]
            },
            "metric_statistics": self._compute_metric_statistics(per_question_metrics)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"✓ Summary: {summary_path}")
        
        print(f"{'='*70}\n")
    
    def compare_agents(
        self,
        agents: List[RAGAgent],
        num_examples: Optional[int] = None,
        **kwargs: Any
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
                    print(f"  {metric_name}: {score:.4f}" if isinstance(score, float) else f"  {metric_name}: {score}")
        
        return all_results

