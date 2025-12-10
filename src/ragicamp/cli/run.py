"""Hydra-powered experiment runner.

Usage:
    # Run with defaults
    python -m ragicamp.cli.run
    
    # Override parameters
    python -m ragicamp.cli.run model=phi3 dataset.num_examples=50
    
    # Use experiment preset
    python -m ragicamp.cli.run experiment=baseline
    
    # Multi-run sweep
    python -m ragicamp.cli.run --multirun model=gemma_2b,phi3 dataset=nq,triviaqa
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ragicamp.core import get_logger, EvaluationError

logger = get_logger(__name__)


def build_legacy_config(cfg: DictConfig) -> Dict[str, Any]:
    """Convert Hydra config to legacy ExperimentConfig format.
    
    This allows gradual migration from the old config system.
    """
    # Convert OmegaConf to dict
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Apply evaluation overrides to dataset if present
    if "evaluation" in config and "dataset_override" in config["evaluation"]:
        override = config["evaluation"].pop("dataset_override", {})
        if override:
            for key, value in override.items():
                if value is not None:
                    config["dataset"][key] = value
    
    # Build legacy format
    legacy = {
        "agent": config.get("agent", {}),
        "model": config.get("model", {}),
        "dataset": config.get("dataset", {}),
        "evaluation": {
            k: v for k, v in config.get("evaluation", {}).items()
            if k != "dataset_override"
        },
        "metrics": config.get("metrics", {}).get("metrics", []),
        "output": config.get("output", {}),
    }
    
    # Add retriever if present
    if "retriever" in config and config["retriever"]:
        legacy["retriever"] = config["retriever"]
    
    # Add judge if present
    if "judge" in config and config["judge"]:
        legacy["judge_model"] = config["judge"]
    
    # Add MLflow if present
    if "mlflow" in config and config["mlflow"]:
        legacy["mlflow"] = config["mlflow"]
    
    # Add experiment metadata if present
    if "experiment" in config:
        legacy["experiment_metadata"] = config["experiment"]
    
    return legacy


def run_evaluation(cfg: DictConfig) -> Dict[str, Any]:
    """Run evaluation with the given configuration.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Evaluation results
    """
    from ragicamp.factory import ComponentFactory
    from ragicamp.evaluation.evaluator import Evaluator
    from ragicamp.core.logging import LogContext
    
    # Convert to legacy format for now
    config = build_legacy_config(cfg)
    
    logger.info("Starting experiment")
    logger.debug("Config: %s", OmegaConf.to_yaml(cfg))
    
    # Extract config sections
    model_config = config.get("model", {})
    agent_config = config.get("agent", {})
    dataset_config = config.get("dataset", {})
    eval_config = config.get("evaluation", {})
    metrics_config = config.get("metrics", [])
    output_config = config.get("output", {})
    
    # Determine mode early to avoid loading unnecessary components
    mode = eval_config.get("mode", "both")
    
    results = {}
    
    try:
        # Phase 2 (evaluate only): Skip model loading, just compute metrics
        if mode == "evaluate":
            predictions_path = eval_config.get("predictions_path")
            if not predictions_path:
                raise ValueError(
                    "evaluation.predictions_path required for mode=evaluate. "
                    "Run with mode=generate first to create predictions."
                )
            
            # Load metrics only
            with LogContext(logger, "Creating metrics"):
                judge_model = None
                judge_config = config.get("judge_model") or config.get("judge")
                if judge_config:
                    judge_model = ComponentFactory.create_model(dict(judge_config))
                
                if isinstance(metrics_config, dict):
                    metrics_list = metrics_config.get("metrics", [])
                else:
                    metrics_list = metrics_config if metrics_config else ["exact_match", "f1"]
                
                metrics = ComponentFactory.create_metrics(
                    metrics_list,
                    judge_model=judge_model,
                )
            
            # Compute metrics from saved predictions
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            output_path = str(output_dir / "metrics.json")
            
            with LogContext(logger, "Metrics", predictions=predictions_path):
                results = Evaluator.compute_metrics_from_file(
                    predictions_path=predictions_path,
                    metrics=metrics,
                    output_path=output_path,
                )
            
            # Skip to results logging
            logger.info("Evaluation complete!")
            for key, value in results.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logger.info(f"  {key}: {value:.4f}")
            return results
        
        # For generate or both modes: Load model and agent
        with LogContext(logger, "Loading model", model=model_config.get("model_name")):
            model = ComponentFactory.create_model(dict(model_config))
        
        # Create retriever if needed
        retriever = None
        if agent_config.get("type") in ["fixed_rag", "bandit_rag", "mdp_rag"]:
            retriever_config = config.get("retriever", {})
            if retriever_config:
                with LogContext(logger, "Loading retriever"):
                    retriever = ComponentFactory.create_retriever(dict(retriever_config))
        
        # Create agent
        with LogContext(logger, "Creating agent", agent_type=agent_config.get("type")):
            agent = ComponentFactory.create_agent(
                config=dict(agent_config),
                model=model,
                retriever=retriever,
            )
        
        # Load dataset
        with LogContext(logger, "Loading dataset", dataset=dataset_config.get("name")):
            dataset = ComponentFactory.create_dataset(dict(dataset_config))
        
        # Skip metrics for generate-only mode (Phase 1)
        metrics = []
        if mode != "generate":
            with LogContext(logger, "Creating metrics"):
                judge_model = None
                judge_config = config.get("judge_model") or config.get("judge")
                if judge_config:
                    judge_model = ComponentFactory.create_model(dict(judge_config))
                
                if isinstance(metrics_config, dict):
                    metrics_list = metrics_config.get("metrics", [])
                else:
                    metrics_list = metrics_config if metrics_config else ["exact_match", "f1"]
                
                metrics = ComponentFactory.create_metrics(
                    metrics_list,
                    judge_model=judge_model,
                )
        
        # Create evaluator
        evaluator = Evaluator(
            agent=agent,
            dataset=dataset,
            metrics=metrics,
        )
        
        # Determine output path
        output_path = output_config.get("output_path")
        if not output_path:
            # Use Hydra's output directory
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            output_path = str(output_dir / "results.json")
        
        # Run based on mode (evaluate mode already handled above with early return)
        if mode == "generate":
            # Phase 1: Generate predictions only (no metrics)
            with LogContext(logger, "Generation", examples=len(dataset)):
                predictions_path = evaluator.generate_predictions(
                    output_path=output_path,
                    num_examples=dataset_config.get("num_examples"),
                    batch_size=eval_config.get("batch_size"),
                    checkpoint_every=eval_config.get("checkpoint_every"),
                    resume_from_checkpoint=eval_config.get("resume_from_checkpoint", True),
                    retry_failures=eval_config.get("retry_failures", True),
                )
            logger.info("Generation complete! Predictions saved to: %s", predictions_path)
            results = {"predictions_path": predictions_path, "mode": "generate"}
            
        else:
            # Default (mode=both): Generate and evaluate in one pass
            with LogContext(logger, "Evaluation", mode=mode, examples=len(dataset)):
                results = evaluator.evaluate(
                    num_examples=dataset_config.get("num_examples"),
                    save_predictions=output_config.get("save_predictions", True),
                    output_path=output_path,
                    batch_size=eval_config.get("batch_size"),
                    checkpoint_every=eval_config.get("checkpoint_every"),
                    resume_from_checkpoint=eval_config.get("resume_from_checkpoint", True),
                    retry_failures=eval_config.get("retry_failures", True),
                )
        
        # Log results
        logger.info("Evaluation complete!")
        for metric_name, score in results.items():
            if isinstance(score, (int, float)):
                logger.info("  %s: %.4f", metric_name, score)
        
        return results
        
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        raise EvaluationError(
            message="Experiment failed",
            details={"error": str(e)},
            cause=e,
        )


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Main entry point for Hydra-powered experiments.
    
    Args:
        cfg: Hydra configuration (auto-populated)
        
    Returns:
        Evaluation results (for Optuna integration)
    """
    # Print config summary
    print("\n" + "="*60)
    print("RAGiCamp Experiment")
    print("="*60)
    
    if "experiment" in cfg and cfg.experiment:
        print(f"Experiment: {cfg.experiment.get('name', 'unnamed')}")
        if cfg.experiment.get("description"):
            print(f"Description: {cfg.experiment.description}")
    
    print(f"\nModel: {cfg.model.get('model_name', 'unknown')}")
    print(f"Agent: {cfg.agent.get('type', 'unknown')}")
    print(f"Dataset: {cfg.dataset.get('name', 'unknown')} ({cfg.dataset.get('num_examples', 'all')} examples)")
    print(f"Mode: {cfg.evaluation.get('mode', 'both')}")
    print("="*60 + "\n")
    
    # Run evaluation
    results = run_evaluation(cfg)
    
    return results


if __name__ == "__main__":
    main()
