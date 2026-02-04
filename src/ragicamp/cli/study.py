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
from typing import Any, Optional

# Re-export for backward compatibility
from ragicamp.config.validation import (
    validate_config,
    validate_model_spec,
)
from ragicamp.execution.runner import run_spec
from ragicamp.indexes.builder import (
    ensure_indexes_exist,
)
from ragicamp.models import OpenAIModel
from ragicamp.spec.builder import build_specs


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


def create_generator_provider(spec: str, quant: str | None = None):
    """Create a GeneratorProvider from spec."""
    from ragicamp.factory import ProviderFactory

    validate_model_spec(spec)
    return ProviderFactory.create_generator(spec, quantization=quant)


def create_dataset(name: str, limit: Optional[int] = None):
    """Create dataset using DatasetFactory."""
    from ragicamp.factory import DatasetFactory

    config = DatasetFactory.parse_spec(name, limit=limit)
    return DatasetFactory.create(config)


def create_judge_model(llm_judge_config: Optional[dict[str, Any]]):
    """Create LLM judge model from config."""
    if not llm_judge_config:
        return None

    model_spec = llm_judge_config.get("model", "openai:gpt-4o-mini")

    if model_spec.startswith("openai:"):
        model_name = model_spec.split(":", 1)[1]
        return OpenAIModel(model_name=model_name)
    return None


def _get_completed_experiment_names(output_dir: Path) -> set[str]:
    """Scan output directory for completed experiments.

    Used to exclude already-completed experiments from sampling, ensuring
    that when resampling with a new seed, we only sample NEW experiments.

    Args:
        output_dir: Path to study output directory

    Returns:
        Set of experiment names that are already complete
    """
    from ragicamp.state.health import check_health

    completed = set()

    if not output_dir.exists():
        return completed

    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        # Skip non-experiment directories
        if exp_dir.name.startswith(".") or exp_dir.name == "study_meta.json":
            continue

        try:
            health = check_health(exp_dir)
            if health.is_complete:
                completed.add(exp_dir.name)
        except Exception:
            # If health check fails, assume incomplete
            pass

    return completed


def run_study(
    config: dict[str, Any],
    dry_run: bool = False,
    skip_existing: bool = True,
    validate_only: bool = False,
    limit: Optional[int] = None,
    force: bool = False,
    experiment_filter: Optional[str] = None,
    sampling_override: Optional[dict[str, Any]] = None,
) -> None:
    """Run a study from a configuration dictionary.

    This is the main orchestration function that:
    1. Validates the config
    2. Ensures all required indexes exist
    3. Builds the experiment matrix (with optional sampling)
    4. Runs each experiment

    Args:
        config: Study configuration dict (already loaded from YAML)
        dry_run: If True, just print what would be done
        skip_existing: If True, skip completed experiments
        validate_only: If True, just validate config and exit
        limit: Optional limit on examples per experiment
        force: Force re-run of completed/failed experiments
        experiment_filter: Optional filter pattern for experiment names
        sampling_override: Optional sampling config to override YAML settings
            Example: {"mode": "random", "n_experiments": 50, "seed": 42}
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

    print(f"\n{'=' * 70}")
    print(f"Study: {study_name}")
    if description:
        print(f"  {description}")
    print(f"{'=' * 70}")

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
        all_retrievers = rag_config.get("retrievers", [])
        # Only build indexes for retrievers that are actually used in the study
        active_retriever_names = set(rag_config.get("retriever_names", []))
        retriever_configs = [
            r for r in all_retrievers if r.get("name") in active_retriever_names
        ]
        ensure_indexes_exist(retriever_configs, corpus_config)

    # When sampling, exclude already-completed experiments from the pool
    # This prevents resampling experiments that were already run
    exclude_names: Optional[set[str]] = None
    if sampling_override or rag_config.get("sampling"):
        exclude_names = _get_completed_experiment_names(out)
        if exclude_names:
            print(f"📊 Found {len(exclude_names)} completed experiments in {out}")

    # Build experiment specs (with optional sampling, excluding completed)
    specs = build_specs(config, sampling_override=sampling_override, exclude_names=exclude_names)

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
        print(f"\n[{i + 1}/{len(specs)}] {spec.name}")

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
    print(f"\n{'=' * 70}")
    print(f"Study Complete: {study_name}")
    print(f"  Completed: {results['completed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"{'=' * 70}")

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
