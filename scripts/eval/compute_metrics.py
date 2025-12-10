#!/usr/bin/env python3
"""Compute metrics on saved predictions.

This script allows you to compute metrics on predictions that were saved
without metrics, making the pipeline more robust to failures during metrics
computation (e.g., API errors, network issues).

Usage:
    python scripts/compute_metrics.py --predictions outputs/agent_predictions_raw.json --config configs/my_config.yaml
    
    # Or with specific metrics only
    python scripts/compute_metrics.py --predictions outputs/agent_predictions_raw.json --metrics exact_match f1 bertscore
    
    # Resume LLM judge from checkpoint
    python scripts/compute_metrics.py --predictions outputs/agent_predictions_raw.json --config configs/my_config.yaml --resume
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragicamp import ComponentFactory
from ragicamp.config import ConfigLoader
from ragicamp.utils.paths import ensure_dir


def load_predictions(predictions_path: str) -> Dict[str, Any]:
    """Load predictions from JSON file.
    
    Args:
        predictions_path: Path to predictions JSON file
        
    Returns:
        Dict with predictions data
    """
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {data['num_examples']} predictions")
    print(f"  Agent: {data['agent_name']}")
    print(f"  Dataset: {data['dataset_name']}")
    
    return data


def compute_metrics_on_predictions(
    predictions_data: Dict[str, Any],
    metrics: List[Any],
    judge_model: Any = None,
    checkpoint_dir: str = "outputs/checkpoints"
) -> Dict[str, Any]:
    """Compute metrics on loaded predictions.
    
    Args:
        predictions_data: Loaded predictions data
        metrics: List of metric objects
        judge_model: Optional judge model for LLM metrics
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Dict with metric results
    """
    from tqdm import tqdm
    
    # Extract predictions and references
    predictions = []
    references = []
    questions = []
    
    for item in predictions_data["predictions"]:
        predictions.append(item["prediction"])
        references.append(item["expected_answers"])
        questions.append(item["question"])
    
    print(f"\n{'='*70}")
    print("COMPUTING METRICS")
    print(f"{'='*70}\n")
    
    results = {}
    agent_name = predictions_data["agent_name"]
    
    for metric in metrics:
        print(f"  - {metric.name}")
        
        try:
            # Set up checkpoint path for LLM judge
            checkpoint_path = None
            if metric.name in ["llm_judge", "llm_judge_qa"]:
                ensure_dir(checkpoint_dir)
                checkpoint_path = Path(checkpoint_dir) / f"{agent_name}_llm_judge_checkpoint.json"
                
                # Compute with checkpoint support
                scores_dict = metric.compute(
                    predictions=predictions,
                    references=references,
                    questions=questions,
                    checkpoint_path=str(checkpoint_path)
                )
            else:
                scores_dict = metric.compute(
                    predictions=predictions,
                    references=references
                )
            
            results.update(scores_dict)
            print(f"    ✓ {metric.name} computed")
            
        except Exception as e:
            print(f"    ⚠️  Error computing {metric.name}: {e}")
            print(f"    ⚠️  Skipping this metric...")
            results[f"{metric.name}_error"] = str(e)
    
    return results


def compute_per_question_metrics(
    predictions_data: Dict[str, Any],
    metrics: List[Any]
) -> List[Dict[str, Any]]:
    """Compute metrics for each question individually.
    
    Args:
        predictions_data: Loaded predictions data
        metrics: List of metric objects
        
    Returns:
        List of per-question metric scores
    """
    per_question = []
    
    print("\nComputing per-question metrics...")
    
    for i, item in enumerate(predictions_data["predictions"]):
        pred = item["prediction"]
        ref = item["expected_answers"]
        question = item["question"]
        
        question_metrics = {
            "question_index": i,
            "question_id": item["question_id"],
            "question": question
        }
        
        for metric in metrics:
            try:
                score = metric.compute_single(pred, ref, question=question)
                if isinstance(score, dict):
                    question_metrics.update(score)
                else:
                    question_metrics[metric.name] = score
            except Exception as e:
                question_metrics[metric.name] = None
        
        per_question.append(question_metrics)
    
    return per_question


def save_results_with_metrics(
    predictions_data: Dict[str, Any],
    overall_metrics: Dict[str, Any],
    per_question_metrics: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Save updated predictions with metrics.
    
    Args:
        predictions_data: Original predictions data
        overall_metrics: Computed overall metrics
        per_question_metrics: Per-question metrics
        output_dir: Output directory
    """
    from datetime import datetime
    
    agent_name = predictions_data["agent_name"]
    dataset_name = predictions_data["dataset_name"]
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS WITH METRICS")
    print(f"{'='*70}")
    
    # 1. Save predictions with per-question metrics
    predictions_path = Path(output_dir) / f"{agent_name}_predictions.json"
    predictions_with_metrics = {
        "agent_name": agent_name,
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "num_examples": predictions_data["num_examples"],
        "predictions": []
    }
    
    for orig_item, metrics in zip(predictions_data["predictions"], per_question_metrics):
        item_with_metrics = {
            "question_id": orig_item["question_id"],
            "question": orig_item["question"],
            "prediction": orig_item["prediction"],
            "metrics": {
                k: v for k, v in metrics.items()
                if k not in ['question_index', 'question_id', 'question']
            },
            "metadata": orig_item.get("metadata", {})
        }
        predictions_with_metrics["predictions"].append(item_with_metrics)
    
    with open(predictions_path, 'w') as f:
        json.dump(predictions_with_metrics, f, indent=2)
    
    print(f"✓ Predictions + metrics: {predictions_path}")
    
    # 2. Save summary
    summary_path = Path(output_dir) / f"{agent_name}_summary.json"
    
    # Compute statistics
    metric_stats = compute_metric_statistics(per_question_metrics)
    
    summary_data = {
        "agent_name": agent_name,
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "num_examples": predictions_data["num_examples"],
        "overall_metrics": overall_metrics,
        "metric_statistics": metric_stats
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✓ Summary: {summary_path}")
    print(f"{'='*70}\n")


def compute_metric_statistics(
    per_question_metrics: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each metric.
    
    Args:
        per_question_metrics: List of per-question metrics
        
    Returns:
        Dict of statistics per metric
    """
    if not per_question_metrics:
        return {}
    
    # Get metric names
    metric_names = set()
    for item in per_question_metrics:
        for key in item.keys():
            if key not in ['question_index', 'question_id', 'question'] and isinstance(item.get(key), (int, float)):
                metric_names.add(key)
    
    # Compute stats
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics on saved predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute all metrics from config
  python scripts/compute_metrics.py --predictions outputs/agent_predictions_raw.json --config configs/my_config.yaml
  
  # Compute specific metrics
  python scripts/compute_metrics.py --predictions outputs/agent_predictions_raw.json --metrics exact_match f1
  
  # Resume from checkpoint (for LLM judge)
  python scripts/compute_metrics.py --predictions outputs/agent_predictions_raw.json --config configs/my_config.yaml --resume
        """
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file (*_predictions_raw.json)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config (for metric and judge model settings)"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Specific metrics to compute (default: all from config)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: same as predictions file)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory for checkpoints (default: outputs/checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}")
    predictions_data = load_predictions(args.predictions)
    
    # Determine output directory
    output_dir = args.output if args.output else Path(args.predictions).parent
    ensure_dir(output_dir)
    
    # Load config if provided
    config = None
    judge_model = None
    if args.config:
        print(f"\nLoading configuration from {args.config}")
        try:
            config = ConfigLoader.load_and_validate(args.config)
            print("✓ Configuration loaded")
            
            # Create judge model if needed
            if config.judge_model:
                print("Creating judge model...")
                judge_model = ComponentFactory.create_model(config.judge_model.model_dump())
                print(f"✓ Judge model created: {config.judge_model.model_name}")
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            if not args.metrics:
                print("❌ Either --config or --metrics must be provided")
                sys.exit(1)
    
    # Determine which metrics to compute
    if args.metrics:
        # Use specified metrics
        print(f"\nUsing specified metrics: {', '.join(args.metrics)}")
        metrics_config = args.metrics
    elif config:
        # Use metrics from config
        print(f"\nUsing metrics from config")
        metrics_config = [
            m if isinstance(m, str) else m
            for m in config.metrics
        ]
    else:
        print("❌ Either --config or --metrics must be provided")
        sys.exit(1)
    
    # Create metrics
    print("\nCreating metrics...")
    metrics = ComponentFactory.create_metrics(metrics_config, judge_model=judge_model)
    for metric in metrics:
        print(f"  - {metric.name}")
    
    # Compute overall metrics
    overall_metrics = compute_metrics_on_predictions(
        predictions_data,
        metrics,
        judge_model,
        args.checkpoint_dir
    )
    
    # Compute per-question metrics
    per_question_metrics = compute_per_question_metrics(predictions_data, metrics)
    
    # Save results
    save_results_with_metrics(
        predictions_data,
        overall_metrics,
        per_question_metrics,
        output_dir
    )
    
    # Print summary
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    for metric_name, score in overall_metrics.items():
        if isinstance(score, float):
            print(f"  {metric_name}: {score:.4f}")
        else:
            print(f"  {metric_name}: {score}")
    print("="*70 + "\n")
    
    print("✓ Done! Metrics computed and saved.")


if __name__ == "__main__":
    main()

