#!/usr/bin/env python3
"""Main experiment runner script using ComponentFactory and ConfigLoader.

This script provides a clean, config-driven way to run RAG experiments.
- Configuration validation with Pydantic schemas
- Component instantiation via ComponentFactory
- Type-safe config access throughout
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp import ComponentFactory
from ragicamp.config import ConfigLoader
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.retrievers import DenseRetriever
from ragicamp.training.trainer import Trainer


def create_judge_model(config):
    """Create judge model if configured.

    Args:
        config: Judge model configuration

    Returns:
        LanguageModel instance or None
    """
    if not config:
        return None

    try:
        return ComponentFactory.create_model(config)
    except Exception as e:
        print(f"⚠️  Failed to create judge model: {e}")
        return None


def compute_and_save_metrics(predictions_file, config, judge_model=None):
    """Compute metrics on saved predictions and save results.

    Extracted common logic for computing and saving metrics.

    Args:
        predictions_file: Path to predictions JSON file
        config: Experiment configuration
        judge_model: Optional judge model for LLM-based metrics
    """
    from datetime import datetime

    # Load predictions
    with open(predictions_file, "r") as f:
        predictions_data = json.load(f)

    # Create metrics
    print("Creating metrics...")
    metrics_config = [m if isinstance(m, str) else m for m in config.metrics]

    # Add checkpoint support for LLM judge metrics
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for metric_config in metrics_config:
        if isinstance(metric_config, dict):
            metric_name = metric_config.get("name", "")
            if metric_name in ["llm_judge", "llm_judge_qa"]:
                if "params" not in metric_config:
                    metric_config["params"] = {}
                metric_config["params"]["checkpoint_path"] = str(
                    checkpoint_dir / f"{predictions_data['agent_name']}_llm_judge_checkpoint.json"
                )

    metrics = ComponentFactory.create_metrics(metrics_config, judge_model=judge_model)

    for metric in metrics:
        print(f"  - {metric.name}")
    print()

    # Extract predictions and references
    predictions = []
    references = []
    questions = []

    for item in predictions_data["predictions"]:
        predictions.append(item["prediction"])
        references.append(item["expected_answers"])
        questions.append(item["question"])

    # Compute overall metrics
    print("=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70 + "\n")

    results = {}
    for metric in metrics:
        print(f"  - {metric.name}")
        try:
            if metric.name in ["llm_judge", "llm_judge_qa"]:
                scores_dict = metric.compute(
                    predictions=predictions, references=references, questions=questions
                )
            else:
                scores_dict = metric.compute(predictions=predictions, references=references)
            results.update(scores_dict)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            results[f"{metric.name}_error"] = str(e)

    # Add metadata
    results["num_examples"] = len(predictions)
    results["agent_name"] = predictions_data["agent_name"]
    results["dataset_name"] = predictions_data["dataset_name"]

    # Compute per-question metrics
    print("\nComputing per-question metrics...")
    per_question_metrics = []
    for i, (pred, ref, q) in enumerate(zip(predictions, references, questions)):
        question_metrics = {"question_index": i, "question": q}

        for metric in metrics:
            try:
                score = metric.compute_single(pred, ref, question=q)
                if isinstance(score, dict):
                    question_metrics.update(score)
                else:
                    question_metrics[metric.name] = score
            except Exception as e:
                question_metrics[metric.name] = None

        per_question_metrics.append(question_metrics)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for metric_name, score in results.items():
        if metric_name not in ["num_examples", "agent_name", "dataset_name"]:
            if isinstance(score, float):
                print(f"  {metric_name}: {score:.4f}")
            else:
                print(f"  {metric_name}: {score}")
    print("=" * 70 + "\n")

    # Save results
    print("=" * 70)
    print("SAVING RESULTS WITH METRICS")
    print("=" * 70)

    output_dir = Path(predictions_file).parent
    agent_name = predictions_data["agent_name"]
    dataset_name = predictions_data["dataset_name"]
    timestamp = datetime.now().isoformat()

    # Reconstruct examples and responses
    examples = []
    responses = []
    for item in predictions_data["predictions"]:
        from unittest.mock import Mock

        from ragicamp.agents.base import RAGContext, RAGResponse

        ex = Mock()
        ex.id = item["question_id"]
        ex.question = item["question"]
        ex.answers = item["expected_answers"]
        examples.append(ex)

        resp = RAGResponse(
            answer=item["prediction"],
            context=RAGContext(query=item["question"]),
            metadata=item.get("metadata", {}),
        )
        responses.append(resp)

    # Save predictions with metrics
    predictions_with_metrics_path = output_dir / f"{agent_name}_predictions.json"
    predictions_with_metrics = {
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
                    k: v
                    for k, v in metrics_dict.items()
                    if k not in ["question_index", "question"]
                },
                "metadata": resp.metadata if hasattr(resp, "metadata") else {},
            }
            for ex, pred, resp, metrics_dict in zip(
                examples, predictions, responses, per_question_metrics
            )
        ],
    }

    with open(predictions_with_metrics_path, "w") as f:
        json.dump(predictions_with_metrics, f, indent=2)

    print(f"✓ Predictions + metrics: {predictions_with_metrics_path}")

    # Save summary
    summary_path = output_dir / f"{agent_name}_summary.json"

    # Compute statistics
    metric_stats = {}
    metric_names = set()
    for item in per_question_metrics:
        for key in item.keys():
            if key not in ["question_index", "question"] and isinstance(
                item.get(key), (int, float)
            ):
                metric_names.add(key)

    for metric_name in metric_names:
        values = [
            item[metric_name]
            for item in per_question_metrics
            if metric_name in item and item[metric_name] is not None
        ]

        if values:
            metric_stats[metric_name] = {
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
        "metric_statistics": metric_stats,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"✓ Summary: {summary_path}")
    print("=" * 70 + "\n")


def load_retriever(config):
    """Load or create retriever for RAG agents.

    Args:
        config: Retriever configuration

    Returns:
        Retriever instance
    """
    # Check if loading from saved artifact
    artifact_path = config.get("artifact_path")
    if artifact_path:
        print(f"Loading retriever from artifact: {artifact_path}")
        return DenseRetriever.load_index(artifact_path)

    # Otherwise create new retriever
    return ComponentFactory.create_retriever(config)


def run_generate(config):
    """Run generate mode - only generate predictions.

    Args:
        config: Full experiment configuration
    """
    print("\n" + "=" * 70)
    print("=== GENERATE MODE: Predictions Only ===")
    print("=" * 70 + "\n")

    # Create model
    print("Creating model...")
    model = ComponentFactory.create_model(config.model.model_dump())
    print(f"✓ Model loaded: {config.model.model_name}\n")

    # Create retriever if needed for RAG agents
    retriever = None
    agent_type = config.agent.type
    if agent_type in ["fixed_rag", "bandit_rag", "mdp_rag"]:
        if not config.retriever:
            raise ValueError(f"{agent_type} agent requires a retriever configuration")
        print("Loading retriever...")
        retriever = load_retriever(config.retriever.model_dump())
        print(f"✓ Retriever loaded\n")

    # Create agent
    print("Creating agent...")
    agent = ComponentFactory.create_agent(
        config.agent.model_dump(), model=model, retriever=retriever
    )
    print(f"✓ Agent created: {agent.name}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = ComponentFactory.create_dataset(config.dataset.model_dump())
    print(f"Dataset size: {len(dataset)}\n")

    # Create evaluator (without metrics)
    evaluator = Evaluator(agent=agent, dataset=dataset)

    # Get settings
    batch_size = config.evaluation.batch_size
    num_examples = config.evaluation.num_examples
    output_path = config.output.output_path

    # Generate predictions
    predictions_file = evaluator.generate_predictions(
        output_path=output_path, num_examples=num_examples, batch_size=batch_size
    )

    print(f"✓ Predictions saved to: {predictions_file}")
    print(f"\n💡 Next: Compute metrics with:")
    print(
        f"   python scripts/compute_metrics.py --predictions {predictions_file} --config {sys.argv[sys.argv.index('--config')+1]}\n"
    )


def run_evaluate(config):
    """Run evaluate mode - only compute metrics on existing predictions.

    Args:
        config: Full experiment configuration
    """
    print("\n" + "=" * 70)
    print("=== EVALUATE MODE: Metrics Only ===")
    print("=" * 70 + "\n")

    # Get predictions file
    predictions_file = config.evaluation.predictions_file
    if not predictions_file:
        raise ValueError("predictions_file must be set in config for evaluate mode")

    print(f"Loading predictions from: {predictions_file}\n")

    # Load to show info
    with open(predictions_file, "r") as f:
        predictions_data = json.load(f)

    print(f"✓ Loaded {predictions_data['num_examples']} predictions")
    print(f"  Agent: {predictions_data['agent_name']}")
    print(f"  Dataset: {predictions_data['dataset_name']}\n")

    # Create judge model if needed
    judge_model = None
    if config.judge_model:
        print("Creating LLM judge model...")
        judge_model = create_judge_model(config.judge_model.model_dump())
        if judge_model:
            print(f"✓ Created judge model: {config.judge_model.model_name}\n")

    # Compute and save metrics (all logic in helper function)
    compute_and_save_metrics(predictions_file, config, judge_model)


def run_both(config):
    """Run both modes - generate predictions then compute metrics.

    Simply calls run_generate() followed by run_evaluate() for maximum code reuse.

    Args:
        config: Full experiment configuration
    """
    print("\n" + "=" * 70)
    print("=== BOTH MODE: Generate + Evaluate ===")
    print("=" * 70 + "\n")

    # Phase 1: Generate predictions
    run_generate(config)

    # Phase 2: Compute metrics on the predictions we just generated
    # Update config to evaluate mode with the predictions file
    predictions_file = (
        Path(config.output.output_path).parent / f"{config.agent.name}_predictions_raw.json"
    )
    config.evaluation.mode = "evaluate"
    config.evaluation.predictions_file = str(predictions_file)

    run_evaluate(config)


def run_training(config):
    """Run training mode for RL agents.

    Args:
        config: Full experiment configuration
    """
    print("\n" + "=" * 70)
    print("=== Training Mode ===")
    print("=" * 70 + "\n")

    # Create model
    print("Creating model...")
    model = ComponentFactory.create_model(config.model.model_dump())

    # Create retriever
    print("Loading retriever...")
    retriever = load_retriever(config.retriever.dict())

    # Create dataset
    print("Loading dataset...")
    dataset = ComponentFactory.create_dataset(config.dataset.model_dump())

    # Create agent (with policy for RL agents)
    print("Creating agent...")
    agent_type = config.agent.type

    # For RL agents, we need to create the policy
    if agent_type in ["bandit_rag", "mdp_rag"]:
        print("⚠️  Training for RL agents not fully implemented yet")
        print("    Please use evaluation mode for now")
        return

    agent = ComponentFactory.create_agent(
        config.agent.model_dump(), model=model, retriever=retriever
    )

    # Create metrics
    metrics_config = [m if isinstance(m, str) else m for m in config.metrics]
    metrics = ComponentFactory.create_metrics(metrics_config)

    # Create trainer
    training_config = config.training.model_dump() if config.training else {}
    trainer = Trainer(agent=agent, dataset=dataset, metrics=metrics, **training_config)

    # Run training
    print("\nStarting training...")
    trainer.train()

    print("\n✓ Training complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run RAGiCamp experiments from config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation
  python run_experiment.py --config configs/nq_baseline_gemma2b_quick.yaml --mode eval
  
  # Run training (for RL agents)
  python run_experiment.py --config configs/bandit_rag.yaml --mode train
  
  # Custom output path
  python run_experiment.py --config configs/my_config.yaml --output results/my_run.json
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment configuration file (YAML)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "train"],
        default="eval",
        help="Experiment mode: eval (evaluation) or train (training)",
    )

    parser.add_argument("--output", type=str, help="Output path (overrides config)")

    args = parser.parse_args()

    # Load and validate configuration
    print(f"Loading configuration from {args.config}")
    try:
        config = ConfigLoader.load_and_validate(args.config)
        print("✓ Configuration validated successfully\n")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

    # Override output path if specified
    if args.output:
        # Config is a Pydantic model, we need to update it properly
        config.output.output_path = args.output
        config.output.save_predictions = True

    # Determine mode from config or args
    if args.mode == "train":
        run_training(config)
    else:
        # Use evaluation mode from config
        eval_mode = config.evaluation.mode

        if eval_mode == "generate":
            run_generate(config)
        elif eval_mode == "evaluate":
            run_evaluate(config)
        elif eval_mode == "both":
            run_both(config)
        else:
            raise ValueError(f"Unknown evaluation mode: {eval_mode}")


if __name__ == "__main__":
    main()
