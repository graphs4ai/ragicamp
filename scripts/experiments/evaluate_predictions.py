#!/usr/bin/env python3
"""Re-evaluate existing predictions with new/additional metrics.

Usage:
    # Evaluate a single experiment with specific metrics
    python scripts/experiments/evaluate_predictions.py outputs/simple_hf/direct_hf_google_gemma2bit_default_nq --metrics bertscore,bleurt

    # Evaluate all experiments in a directory
    python scripts/experiments/evaluate_predictions.py outputs/simple_hf --metrics llm_judge --llm-judge-model openai:gpt-4o-mini

    # Evaluate with all available metrics
    python scripts/experiments/evaluate_predictions.py outputs/simple_hf --metrics all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def get_model(model_spec: str):
    """Create model from specification string."""
    if model_spec.startswith("openai:"):
        from ragicamp.models import OpenAIModel

        return OpenAIModel(model_spec.replace("openai:", ""))
    elif model_spec.startswith("hf:"):
        from ragicamp.models import HuggingFaceModel

        return HuggingFaceModel(model_spec.replace("hf:", ""))
    else:
        from ragicamp.models import OpenAIModel

        return OpenAIModel(model_spec)


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    import gc

    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def evaluate_predictions(
    predictions_path: Path,
    metrics: List[str],
    llm_judge_model: str = "openai:gpt-4o-mini",
    llm_judge_type: str = "binary",
) -> Dict[str, Any]:
    """Evaluate predictions with specified metrics.

    Args:
        predictions_path: Path to predictions.json
        metrics: List of metric names to compute
        llm_judge_model: Model spec for LLM judge
        llm_judge_type: "binary" or "ternary"

    Returns:
        Dict with aggregated results and per-question updates
    """
    # Load predictions
    with open(predictions_path) as f:
        data = json.load(f)

    preds_list = data.get("predictions", [])
    if not preds_list:
        print(f"  ⚠️ No predictions found in {predictions_path}")
        return {}

    # Extract data for metrics
    predictions = [p["prediction"] for p in preds_list]
    references = [p["expected_answers"] for p in preds_list]
    questions = [p["question"] for p in preds_list]

    # Initialize per-question metrics (preserve existing)
    for p in preds_list:
        if "metrics" not in p:
            p["metrics"] = {}

    aggregated = {}

    for m in metrics:
        try:
            print(f"  📊 Computing {m}...")

            if m == "f1":
                from ragicamp.metrics import F1Metric

                metric = F1Metric()
                result = metric.compute(predictions, references)
                aggregated.update(result)
                # Per-question
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    score = metric.compute([pred], [ref])
                    preds_list[i]["metrics"]["f1"] = score.get("f1", 0.0)

            elif m == "exact_match":
                from ragicamp.metrics import ExactMatchMetric

                metric = ExactMatchMetric()
                result = metric.compute(predictions, references)
                aggregated.update(result)
                # Per-question
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    score = metric.compute([pred], [ref])
                    preds_list[i]["metrics"]["exact_match"] = score.get("exact_match", 0.0)

            elif m == "bertscore":
                clear_gpu_memory()
                from ragicamp.metrics import BERTScoreMetric

                metric = BERTScoreMetric(model_type="microsoft/deberta-base-mnli")
                result = metric.compute(predictions, references)
                aggregated.update(result)
                # Per-question
                item_scores = metric.get_per_item_scores()
                for i, score in enumerate(item_scores):
                    preds_list[i]["metrics"]["bertscore_f1"] = score

            elif m == "bleurt":
                clear_gpu_memory()
                from ragicamp.metrics import BLEURTMetric

                metric = BLEURTMetric()
                result = metric.compute(predictions, references)
                aggregated.update(result)
                # Per-question
                item_scores = metric.get_per_item_scores()
                for i, score in enumerate(item_scores):
                    preds_list[i]["metrics"]["bleurt"] = score

            elif m == "llm_judge":
                from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric

                judge_model = get_model(llm_judge_model)
                metric = LLMJudgeQAMetric(
                    judge_model=judge_model,
                    judgment_type=llm_judge_type,
                )
                import asyncio

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    metric.acompute(predictions, references, questions)
                )
                loop.close()
                aggregated.update(result)
                # Per-question
                item_scores = metric.get_per_item_scores()
                for i, score in enumerate(item_scores):
                    preds_list[i]["metrics"]["llm_judge"] = score

            else:
                print(f"  ⚠️ Unknown metric: {m}")

        except Exception as e:
            print(f"  ❌ Error computing {m}: {e}")

    # Save updated predictions
    with open(predictions_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✅ Updated {predictions_path}")

    # Update results.json if it exists
    results_path = predictions_path.parent / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        results.update(aggregated)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✅ Updated {results_path}")

    return aggregated


def find_experiments(base_path: Path) -> List[Path]:
    """Find all experiment directories with predictions.json."""
    experiments = []

    # Check if base_path itself is an experiment
    if (base_path / "predictions.json").exists():
        experiments.append(base_path)
    else:
        # Search subdirectories
        for pred_file in base_path.rglob("predictions.json"):
            experiments.append(pred_file.parent)

    return sorted(experiments)


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing predictions with new metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single experiment with BERTScore
  python scripts/experiments/evaluate_predictions.py outputs/simple_hf/direct_hf_google_gemma2bit_default_nq --metrics bertscore

  # Evaluate all experiments with LLM judge
  python scripts/experiments/evaluate_predictions.py outputs/simple_hf --metrics llm_judge

  # Evaluate with multiple metrics
  python scripts/experiments/evaluate_predictions.py outputs/simple --metrics f1,exact_match,bertscore,bleurt,llm_judge
        """,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to experiment directory or parent directory containing experiments",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated list of metrics: f1,exact_match,bertscore,bleurt,llm_judge (or 'all')",
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        default="openai:gpt-4o-mini",
        help="Model for LLM-as-judge (default: openai:gpt-4o-mini)",
    )
    parser.add_argument(
        "--llm-judge-type",
        type=str,
        choices=["binary", "ternary"],
        default="binary",
        help="LLM judge type (default: binary)",
    )

    args = parser.parse_args()

    # Parse metrics
    if args.metrics.lower() == "all":
        metrics = ["f1", "exact_match", "bertscore", "bleurt", "llm_judge"]
    else:
        metrics = [m.strip() for m in args.metrics.split(",")]

    print(f"📊 Metrics to compute: {', '.join(metrics)}")

    # Find experiments
    experiments = find_experiments(args.path)
    if not experiments:
        print(f"❌ No experiments found at {args.path}")
        sys.exit(1)

    print(f"📁 Found {len(experiments)} experiment(s)")

    # Evaluate each experiment
    all_results = {}
    for exp_path in experiments:
        exp_name = exp_path.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        predictions_path = exp_path / "predictions.json"
        results = evaluate_predictions(
            predictions_path,
            metrics,
            llm_judge_model=args.llm_judge_model,
            llm_judge_type=args.llm_judge_type,
        )
        all_results[exp_name] = results

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Evaluated {len(experiments)} experiment(s) with metrics: {', '.join(metrics)}")


if __name__ == "__main__":
    main()
