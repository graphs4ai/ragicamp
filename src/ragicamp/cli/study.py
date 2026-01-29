"""Study runner for CLI.

This module provides the main orchestration for running studies from YAML config files.
The actual implementation is delegated to specialized modules:
- config.validation: Config validation
- indexes.builder: Index building
- execution.runner: Experiment execution
- factory: Component creation
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ragicamp.config.validation import ConfigError, validate_config, validate_model_spec
from ragicamp.execution.runner import ExpSpec, build_specs, run_spec
from ragicamp.indexes.builder import ensure_indexes_exist
from ragicamp.models import OpenAIModel
from ragicamp.utils.artifacts import get_artifact_manager

# Re-export for backward compatibility
from ragicamp.config.validation import (
    ConfigError,
    validate_config,
    validate_dataset,
    validate_model_spec,
    VALID_DATASETS,
    VALID_PROVIDERS,
    VALID_QUANTIZATIONS,
)

from ragicamp.indexes.builder import (
    get_embedding_index_name,
    build_embedding_index,
    build_hierarchical_index,
    build_retriever_from_index,
    ensure_indexes_exist,
)

from ragicamp.execution.runner import (
    ExpSpec,
    build_specs,
    run_spec,
    run_spec_subprocess,
)

from ragicamp.factory import (
    load_retriever,
    create_query_transformer,
    create_reranker,
)


def get_prompt_builder(prompt_type: str, dataset: str):
    """Get a PromptBuilder configured for the given prompt type and dataset.

    Args:
        prompt_type: One of "default", "concise", "fewshot", etc.
        dataset: Dataset name (for loading appropriate fewshot examples)

    Returns:
        Configured PromptBuilder instance
    """
    from ragicamp.utils.prompts import PromptBuilder

    return PromptBuilder.from_config(prompt_type, dataset=dataset)


def create_model(spec: str, quant: str = "4bit"):
    """Create model from spec using ComponentFactory."""
    from ragicamp.factory import ComponentFactory

    validate_model_spec(spec)
    config = ComponentFactory.parse_model_spec(spec, quantization=quant)
    return ComponentFactory.create_model(config)


def create_dataset(name: str, limit: Optional[int] = None):
    """Create dataset using ComponentFactory."""
    from ragicamp.factory import ComponentFactory

    config = ComponentFactory.parse_dataset_spec(name, limit=limit)
    return ComponentFactory.create_dataset(config)


def create_judge_model(llm_judge_config: Optional[Dict[str, Any]]):
    """Create LLM judge model from config."""
    if not llm_judge_config:
        return None

    model_spec = llm_judge_config.get("model", "openai:gpt-4o-mini")

    if model_spec.startswith("openai:"):
        model_name = model_spec.split(":", 1)[1]
        return OpenAIModel(model_name=model_name)
    return None


def run_study(
    config: Dict[str, Any],
    dry_run: bool = False,
    skip_existing: bool = True,
    validate_only: bool = False,
    limit: Optional[int] = None,
    force: bool = False,
    experiment_filter: Optional[str] = None,
) -> None:
    """Run a study from a configuration dictionary.

    This is the main orchestration function that:
    1. Validates the config
    2. Ensures all required indexes exist
    3. Builds the experiment matrix
    4. Runs each experiment

    Args:
        config: Study configuration dict (already loaded from YAML)
        dry_run: If True, just print what would be done
        skip_existing: If True, skip completed experiments
        validate_only: If True, just validate config and exit
        limit: Optional limit on examples per experiment
        force: Force re-run of completed/failed experiments
        experiment_filter: Optional filter pattern for experiment names
    """
    # Validate
    warnings = validate_config(config)
    for w in warnings:
        print(f"⚠️  {w}")

    if validate_only:
        print("✓ Configuration is valid")
        return

    study_name = config["name"]
    description = config.get("description", "")
    out = Path(config.get("output_dir", f"outputs/{study_name}"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Study: {study_name}")
    if description:
        print(f"  {description}")
    print(f"{'='*70}")

    # Override limit from config if not specified
    if limit is None:
        limit = config.get("num_questions")

    # Get metrics
    metrics = config.get("metrics", ["f1", "exact_match"])

    # Get LLM judge config
    llm_judge_config = config.get("llm_judge")
    judge_model = create_judge_model(llm_judge_config)

    # Ensure indexes exist for RAG experiments
    rag_config = config.get("rag", {})
    if rag_config.get("enabled"):
        corpus_config = rag_config.get("corpus", {})
        retriever_configs = rag_config.get("retrievers", [])
        ensure_indexes_exist(retriever_configs, corpus_config)

    # Build experiment specs
    specs = build_specs(config)

    # Filter if requested
    if experiment_filter:
        specs = [s for s in specs if experiment_filter in s.name]

    print(f"\n📋 Experiments: {len(specs)}")
    for s in specs:
        print(f"   - {s.name}")

    # Dry run - just show what would be done
    if dry_run:
        print("\n[DRY RUN] Would run the above experiments")
        return

    # Run experiments
    results = {"completed": 0, "failed": 0, "skipped": 0}

    # Determine if we should skip existing (inverse of force)
    should_force = force or not skip_existing

    for i, spec in enumerate(specs):
        print(f"\n[{i+1}/{len(specs)}] {spec.name}")

        status = run_spec(
            spec=spec,
            limit=limit,
            metrics=metrics,
            out=out,
            judge_model=judge_model,
            llm_judge_config=llm_judge_config,
            force=should_force,
            use_subprocess=True,
        )

        if status in ("complete", "ran", "resumed"):
            results["completed"] += 1
        elif status == "failed":
            results["failed"] += 1
        else:
            results["skipped"] += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"Study Complete: {study_name}")
    print(f"  Completed: {results['completed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"{'='*70}")

    # Save study metadata
    meta = {
        "name": study_name,
        "num_experiments": len(specs),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out / "study_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def compare(out: Path) -> None:
    """Compare experiment results in a study output directory.

    Args:
        out: Path to study output directory
    """
    from ragicamp.analysis.comparison import compare_experiments

    compare_experiments(out)
