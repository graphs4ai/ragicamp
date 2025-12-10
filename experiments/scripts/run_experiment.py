#!/usr/bin/env python3
"""Main experiment runner script using ComponentFactory and ConfigLoader.

This script provides a clean, config-driven way to run RAG experiments.
- Configuration validation with Pydantic schemas
- Component instantiation via ComponentFactory
- Type-safe config access throughout
"""

import argparse
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

# Import new utilities
try:
    from ragicamp.utils import MLflowTracker, ExperimentState
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLflowTracker = None
    ExperimentState = None


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


def run_evaluation(config):
    """Run evaluation mode with MLflow tracking and state management.

    Args:
        config: Full experiment configuration
    """
    print("\n" + "=" * 70)
    print("=== Evaluation Mode ===")
    print("=" * 70 + "\n")

    # Initialize MLflow tracker
    mlflow_config = config.mlflow.model_dump() if hasattr(config, 'mlflow') else {}
    mlflow_tracker = None
    if MLFLOW_AVAILABLE and mlflow_config.get("enabled", True):
        mlflow_tracker = MLflowTracker(
            enabled=True,
            experiment_name=mlflow_config.get("experiment_name"),
            tracking_uri=mlflow_config.get("tracking_uri"),
            run_name=mlflow_config.get("run_name"),
            tags=mlflow_config.get("tags", {}),
        )
        print("✓ MLflow tracking enabled\n")
    
    # Initialize experiment state
    state = None
    state_path = None
    if hasattr(config, 'evaluation') and config.evaluation.save_state:
        output_path = config.output.output_path
        if output_path:
            state_path = Path(output_path).with_name(
                Path(output_path).stem + "_state.json"
            )
            
            # Create or load state
            agent_name = config.agent.name if hasattr(config, 'agent') else "experiment"
            phase_names = ["generation", "metrics"]  # Standard phases
            
            if MLFLOW_AVAILABLE and ExperimentState:
                state = ExperimentState.load_or_create(
                    path=state_path,
                    name=agent_name,
                    phase_names=phase_names,
                    config=config.model_dump() if hasattr(config, 'model_dump') else {},
                    mlflow_run_id=None,  # Will be set after MLflow run starts
                )
                print(f"📊 Experiment state: {state_path}")
                print(state.summary())
                print()

    # Start MLflow run (context manager)
    mlflow_context = mlflow_tracker.start_run() if mlflow_tracker else _DummyContext()
    
    with mlflow_context:
        # Log MLflow run ID to state
        if mlflow_tracker and state:
            state.mlflow_run_id = mlflow_tracker.get_run_id()
            state.save(state_path)
        
        # Log config parameters to MLflow
        if mlflow_tracker:
            mlflow_tracker.log_params(config.model_dump() if hasattr(config, 'model_dump') else {})
        
        # Create model
        print("Creating model...")
        model = ComponentFactory.create_model(config.model.model_dump())
        print(f"✓ Model loaded: {config.model.model_name}\n")

        # Create judge model if needed
        judge_model = None
        if config.judge_model:
            print("Creating LLM judge model...")
            judge_model = create_judge_model(config.judge_model.model_dump())
            if judge_model:
                print(f"✓ Created judge model: {config.judge_model.model_name}\n")

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

        # Create metrics
        print("Creating metrics...")
        metrics_config = [m if isinstance(m, str) else m for m in config.metrics]
        metrics = ComponentFactory.create_metrics(metrics_config, judge_model=judge_model)

        # Show which metrics are loaded
        metric_names = [m.name for m in metrics]
        for name in metric_names:
            print(f"  - {name}")
        print()

        # Create evaluator
        evaluator = Evaluator(agent=agent, dataset=dataset, metrics=metrics)

        # Get evaluation settings
        batch_size = config.evaluation.batch_size
        save_predictions = config.output.save_predictions
        output_path = config.output.output_path
        
        # Get checkpoint settings
        checkpoint_every = getattr(config.evaluation, 'checkpoint_every', None)
        resume_from_checkpoint = getattr(config.evaluation, 'resume_from_checkpoint', False)
        retry_failures = getattr(config.evaluation, 'retry_failures', False)

        # Run evaluation
        print(f"Evaluating on {len(dataset)} examples...")
        
        try:
            if state:
                state.start_phase("generation")
                state.save(state_path)
            
            results = evaluator.evaluate(
                save_predictions=save_predictions,
                output_path=output_path,
                batch_size=batch_size,
                checkpoint_every=checkpoint_every,
                resume_from_checkpoint=resume_from_checkpoint,
                retry_failures=retry_failures,
            )
            
            if state:
                state.complete_phase("generation", output_path=output_path)
                state.complete_phase("metrics", metadata={"results": results})
                state.save(state_path)
        
        except Exception as e:
            if state:
                state.fail_phase("generation" if "generation" in str(e) else "metrics", error=str(e))
                state.save(state_path)
            raise

        # Log metrics to MLflow
        if mlflow_tracker:
            # Filter numeric metrics for MLflow
            numeric_metrics = {k: v for k, v in results.items() 
                             if isinstance(v, (int, float)) and v is not None}
            mlflow_tracker.log_metrics(numeric_metrics)
            
            # Log artifacts
            if mlflow_config.get("log_artifacts", True) and output_path:
                mlflow_tracker.log_artifact(output_path)
                if state_path and state_path.exists():
                    mlflow_tracker.log_artifact(str(state_path))

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        for metric_name, score in results.items():
            if isinstance(score, float):
                print(f"  {metric_name}: {score:.4f}")
            else:
                print(f"  {metric_name}: {score}")
        print("=" * 70 + "\n")

        if save_predictions and output_path:
            print(f"✓ Results saved to: {output_path}\n")
        
        if mlflow_tracker:
            run_id = mlflow_tracker.get_run_id()
            print(f"✓ MLflow run ID: {run_id}")
            print(f"   View at: http://localhost:5000\n")


class _DummyContext:
    """Dummy context manager for when MLflow is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


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

    # Run appropriate mode
    if args.mode == "eval":
        run_evaluation(config)
    elif args.mode == "train":
        run_training(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
