"""Optuna-powered experiment search for RAG configurations.

Provides a unified trial-by-trial loop that works with any Optuna sampler:

- **random**: ``RandomSampler`` -- equivalent to random search, but with
  SQLite persistence and resume capability.
- **tpe**: ``TPESampler`` -- Bayesian optimization that learns from previous
  trials to focus on high-performing regions of the search space.

Each trial:
1. The sampler suggests a parameter combination (model, retriever, top_k, ...)
2. An ExperimentSpec is built from those parameters
3. The experiment is run via the standard runner (subprocess isolation)
4. The metric is extracted from results and reported back to Optuna

Study state is persisted to ``{output_dir}/optuna_study.db`` (SQLite),
enabling transparent resume across runs.

Usage (YAML config):
    rag:
      sampling:
        mode: tpe            # or "random"
        n_experiments: 100
        optimize_metric: f1  # only matters for tpe
        seed: 42

Usage (CLI override):
    ragicamp run config.yaml --sample 100 --sample-mode tpe
"""

import json
from pathlib import Path
from typing import Any, Optional

import optuna

from ragicamp.core.logging import get_logger
from ragicamp.execution.runner import run_spec
from ragicamp.spec.experiment import ExperimentSpec
from ragicamp.spec.naming import name_rag

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Search space extraction
# ---------------------------------------------------------------------------


def _extract_search_space(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Extract categorical search space dimensions from the study config.

    Reads the same YAML fields that grid search uses, so the search space
    is always consistent with what the user configured.

    Returns:
        Dict mapping dimension names to their possible values.
    """
    rag = config.get("rag", {})

    space: dict[str, list[Any]] = {
        "model": rag.get("models", []),
        "retriever": rag.get("retriever_names", []),
        "top_k": rag.get("top_k_values", [5]),
        "prompt": rag.get("prompts", ["concise"]),
        "query_transform": rag.get("query_transform", ["none"]),
        "dataset": config.get("datasets", ["nq"]),
    }

    # Build unique reranker names from reranker configs
    reranker_config = rag.get("reranker", {})
    if isinstance(reranker_config, dict):
        reranker_cfgs = reranker_config.get("configs", [{"enabled": False, "name": "none"}])
    else:
        reranker_cfgs = [{"enabled": False, "name": "none"}]

    reranker_names: list[str] = []
    for cfg in reranker_cfgs:
        name = cfg.get("name", "none") if cfg.get("enabled") else "none"
        if name not in reranker_names:
            reranker_names.append(name)

    if not reranker_names:
        reranker_names = ["none"]

    space["reranker"] = reranker_names

    return space


# ---------------------------------------------------------------------------
# Config lookups (reuse retriever/reranker YAML definitions)
# ---------------------------------------------------------------------------


def _build_retriever_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build retriever name -> config dict lookup from the YAML."""
    rag = config.get("rag", {})
    lookup: dict[str, dict[str, Any]] = {}
    for ret_config in rag.get("retrievers", []):
        if isinstance(ret_config, dict):
            lookup[ret_config["name"]] = ret_config
    return lookup


def _build_reranker_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build reranker name -> config dict lookup from the YAML."""
    rag = config.get("rag", {})
    reranker_config = rag.get("reranker", {})
    if isinstance(reranker_config, dict):
        reranker_cfgs = reranker_config.get("configs", [])
    else:
        reranker_cfgs = []

    lookup: dict[str, dict[str, Any]] = {}
    for cfg in reranker_cfgs:
        name = cfg.get("name", "none") if cfg.get("enabled") else "none"
        lookup[name] = cfg
    return lookup


# ---------------------------------------------------------------------------
# Trial -> ExperimentSpec conversion
# ---------------------------------------------------------------------------


def _trial_to_spec(
    trial: optuna.Trial,
    search_space: dict[str, list[Any]],
    retriever_lookup: dict[str, dict[str, Any]],
    reranker_lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> ExperimentSpec:
    """Convert an Optuna trial's suggested parameters into an ExperimentSpec.

    Uses the same logic as ``_build_rag_specs`` for fetch_k computation,
    naming, and retriever/reranker resolution so that Optuna-generated
    experiments are fully compatible with the rest of the framework.
    """
    rag = config.get("rag", {})

    # --- Suggest one value per dimension ---
    model = trial.suggest_categorical("model", search_space["model"])
    retriever = trial.suggest_categorical("retriever", search_space["retriever"])
    top_k = trial.suggest_categorical("top_k", search_space["top_k"])
    prompt = trial.suggest_categorical("prompt", search_space["prompt"])
    qt = trial.suggest_categorical("query_transform", search_space["query_transform"])
    dataset = trial.suggest_categorical("dataset", search_space["dataset"])
    reranker = trial.suggest_categorical("reranker", search_space["reranker"])

    # --- Resolve retriever config ---
    ret_cfg = retriever_lookup.get(retriever, {})
    embedding_index = ret_cfg.get("embedding_index")
    sparse_index = ret_cfg.get("sparse_index")

    # --- Resolve reranker config ---
    rr_cfg = reranker_lookup.get(reranker, {})
    has_reranker = reranker != "none" and rr_cfg.get("enabled", False)
    rr_model = rr_cfg.get("model") if has_reranker else None
    rr_name = reranker if has_reranker else None

    # --- Compute fetch_k (same logic as builder.py) ---
    fetch_k_config = rag.get("fetch_k")
    fetch_k_multiplier = rag.get("fetch_k_multiplier", 4)

    if fetch_k_config is not None:
        fetch_k = fetch_k_config
    elif has_reranker:
        fetch_k = top_k * fetch_k_multiplier
    else:
        fetch_k = None

    # --- Build canonical name ---
    name = name_rag(
        model,
        prompt,
        dataset,
        retriever,
        top_k,
        qt,
        reranker if has_reranker else "none",
    )

    batch_size = config.get("batch_size", 8)
    metrics = config.get("metrics", ["f1", "exact_match"])

    return ExperimentSpec(
        name=name,
        exp_type="rag",
        model=model,
        dataset=dataset,
        prompt=prompt,
        retriever=retriever,
        embedding_index=embedding_index,
        sparse_index=sparse_index,
        top_k=top_k,
        fetch_k=fetch_k,
        query_transform=qt if qt != "none" else None,
        reranker=rr_name,
        reranker_model=rr_model,
        batch_size=batch_size,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Metric extraction from completed experiments
# ---------------------------------------------------------------------------


def _get_experiment_metric(
    exp_name: str,
    output_dir: Path,
    metric_name: str = "f1",
) -> Optional[float]:
    """Extract a metric value from a completed experiment's output files.

    Tries ``results.json`` first, then ``predictions.json``.

    Returns:
        The metric value, or None if not found.
    """
    exp_dir = output_dir / exp_name

    # Try results.json first (written on COMPLETE phase)
    results_path = exp_dir / "results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            if metric_name in metrics:
                return float(metrics[metric_name])
        except Exception as e:
            logger.warning("Failed to read results.json for %s: %s", exp_name, e)

    # Fallback: predictions.json (aggregate_metrics section)
    predictions_path = exp_dir / "predictions.json"
    if predictions_path.exists():
        try:
            with open(predictions_path) as f:
                data = json.load(f)
            metrics = data.get("aggregate_metrics", {})
            if metric_name in metrics:
                return float(metrics[metric_name])
        except Exception as e:
            logger.warning("Failed to read predictions.json for %s: %s", exp_name, e)

    return None


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------


def _create_sampler(
    mode: str, seed: Optional[int] = None
) -> optuna.samplers.BaseSampler:
    """Map a sampling mode name to an Optuna sampler instance.

    Args:
        mode: ``"random"``, ``"tpe"``, or any legacy alias.
        seed: Optional random seed for reproducibility.
    """
    if mode == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    # "tpe", "optuna", "stratified", or anything else → TPE
    return optuna.samplers.TPESampler(seed=seed)


# ---------------------------------------------------------------------------
# Warm-start from existing experiments on disk
# ---------------------------------------------------------------------------


def _seed_from_existing(
    study: optuna.Study,
    search_space: dict[str, list[Any]],
    output_dir: Path,
    optimize_metric: str,
) -> int:
    """Register completed experiments on disk as Optuna trials (warm-start).

    Scans ``output_dir`` for completed RAG experiments that are NOT already
    in the Optuna study, reads their parameters from ``metadata.json`` and
    their metric from ``results.json`` / ``predictions.json``, and adds
    them as completed trials.

    This lets TPE learn from prior runs immediately instead of starting
    blind, and lets RandomSampler skip already-explored combinations.

    Only experiments whose parameters fall within the current search space
    are registered (experiments from a different study config are skipped).

    Returns:
        Number of trials seeded.
    """
    from optuna.distributions import CategoricalDistribution

    # Build Optuna distributions from the search space
    distributions = {
        dim: CategoricalDistribution(choices=values)
        for dim, values in search_space.items()
    }

    # Build a set of param fingerprints already in the study to avoid dupes
    existing_params: set[tuple] = set()
    for trial in study.trials:
        key = tuple(sorted(trial.params.items()))
        existing_params.add(key)

    seeded = 0

    for exp_dir in sorted(output_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        metadata_path = exp_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path) as f:
                meta = json.load(f)
        except Exception:
            continue

        # Only seed RAG experiments
        if meta.get("type") != "rag":
            continue

        # Map metadata fields to Optuna param names
        params = {
            "model": meta.get("model"),
            "retriever": meta.get("retriever"),
            "top_k": meta.get("top_k"),
            "prompt": meta.get("prompt"),
            "query_transform": meta.get("query_transform") or "none",
            "dataset": meta.get("dataset"),
            "reranker": meta.get("reranker") or "none",
        }

        # Validate every param is within the current search space
        valid = True
        for dim, value in params.items():
            if value not in search_space.get(dim, []):
                valid = False
                break

        if not valid:
            continue

        # Skip if already registered
        key = tuple(sorted(params.items()))
        if key in existing_params:
            continue

        # Extract metric value
        metric_value = _get_experiment_metric(exp_dir.name, output_dir, optimize_metric)
        if metric_value is None:
            continue

        # Create and register the trial
        trial = optuna.trial.FrozenTrial(
            number=0,  # reassigned by storage
            state=optuna.trial.TrialState.COMPLETE,
            value=metric_value,
            datetime_start=None,
            datetime_complete=None,
            params=params,
            distributions=distributions,
            user_attrs={"seeded_from": exp_dir.name},
            system_attrs={},
            intermediate_values={},
        )
        study.add_trial(trial)
        existing_params.add(key)
        seeded += 1

    return seeded


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_optuna_study(
    config: dict[str, Any],
    n_trials: int,
    output_dir: Path,
    sampler_mode: str = "tpe",
    optimize_metric: str = "f1",
    limit: Optional[int] = None,
    judge_model: Any = None,
    llm_judge_config: Optional[dict[str, Any]] = None,
    study_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> optuna.Study:
    """Run an Optuna-driven experiment search over the RAG search space.

    Works with any supported sampler mode:

    - ``"random"``: Uniform random exploration (``RandomSampler``).
    - ``"tpe"``: Bayesian optimization via TPE (``TPESampler``).

    The study is persisted to ``{output_dir}/optuna_study.db`` (SQLite),
    so it can be resumed transparently by re-running the same command.

    Args:
        config: Full study YAML config dict.
        n_trials: Total number of trials to run.
        output_dir: Directory for experiment outputs.
        sampler_mode: Sampler strategy -- ``"random"`` or ``"tpe"``.
        optimize_metric: Metric to maximize (default: ``"f1"``).
        limit: Max questions per experiment (overrides config).
        judge_model: LLM judge model instance.
        llm_judge_config: LLM judge config dict.
        study_name: Optuna study name (defaults to ``{config.name}_{mode}``).
        seed: Random seed for the sampler.

    Returns:
        The :class:`optuna.Study` object containing all trial results.
    """
    config_name = config.get("name", "ragicamp")
    if study_name is None:
        study_name = f"{config_name}_{sampler_mode}"

    # Persist to SQLite for resume capability
    storage_path = output_dir / "optuna_study.db"
    storage = f"sqlite:///{storage_path}"

    sampler = _create_sampler(sampler_mode, seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    # Extract search space and config lookups
    search_space = _extract_search_space(config)
    retriever_lookup = _build_retriever_lookup(config)
    reranker_lookup = _build_reranker_lookup(config)
    metrics = config.get("metrics", ["f1", "exact_match"])

    # --- Warm-start: seed from existing experiments on disk ---
    seeded = _seed_from_existing(study, search_space, output_dir, optimize_metric)
    if seeded > 0:
        print(f"📊 Seeded {seeded} existing experiments into Optuna study")

    # --- Log search space summary ---
    total_combos = 1
    for values in search_space.values():
        total_combos *= len(values)

    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining = max(0, n_trials - completed_trials)

    mode_label = "TPE Optimization" if sampler_mode != "random" else "Random Search"
    print(f"\n{'=' * 70}")
    print(f"{mode_label}: {study_name}")
    print(f"  Sampler: {type(sampler).__name__}")
    print(f"  Metric: {optimize_metric} (maximize)")
    print(f"  Trials: {n_trials} (target)")
    print(f"  Storage: {storage_path}")
    print(f"  Search space:")
    for dim, values in search_space.items():
        print(f"    {dim}: {len(values)} values")
    print(f"  Total possible combinations: {total_combos:,}")

    if completed_trials > 0:
        print(f"  Completed trials: {completed_trials} ({seeded} seeded from disk)")
        print(f"  New trials to run: {remaining}")
    print(f"{'=' * 70}\n")

    if remaining == 0:
        print("All trials already completed. Use a higher --sample value to run more.")
        _print_study_summary(study, optimize_metric, n_trials)
        return study

    # --- Define objective function ---
    def objective(trial: optuna.Trial) -> float:
        """Run one experiment and return the target metric."""
        spec = _trial_to_spec(
            trial, search_space, retriever_lookup, reranker_lookup, config,
        )

        trial_num = trial.number + 1
        model_short = spec.model.split("/")[-1] if "/" in spec.model else spec.model
        print(
            f"\n[Trial {trial_num}/{n_trials}] {spec.name}\n"
            f"  model={model_short}, retriever={spec.retriever}, "
            f"top_k={spec.top_k}, prompt={spec.prompt}, "
            f"qt={spec.query_transform or 'none'}, rr={spec.reranker or 'none'}, "
            f"dataset={spec.dataset}"
        )

        # Run the experiment (handles already-complete via health check)
        status = run_spec(
            spec=spec,
            limit=limit,
            metrics=metrics,
            out=output_dir,
            judge_model=judge_model,
            llm_judge_config=llm_judge_config,
            force=False,
            use_subprocess=True,
        )

        # Extract metric from results
        if status in ("complete", "ran", "resumed"):
            value = _get_experiment_metric(spec.name, output_dir, optimize_metric)
            if value is not None:
                print(f"  -> {optimize_metric} = {value:.4f}")
                return value

        # Failed/timeout experiments get 0 so Optuna avoids this region
        print(f"  -> experiment {status}, reporting {optimize_metric}=0.0")
        return 0.0

    # --- Run optimization ---
    study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    # --- Print summary ---
    _print_study_summary(study, optimize_metric, n_trials)

    return study


def _print_study_summary(
    study: optuna.Study,
    optimize_metric: str,
    n_trials: int,
) -> None:
    """Print a summary of the Optuna study results."""
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print(f"\n{'=' * 70}")
    print(f"Optuna Study Complete: {study.study_name}")
    print(f"  Total trials: {len(study.trials)}")

    if not completed:
        print("  No completed trials.")
        print(f"{'=' * 70}")
        return

    print(f"  Best {optimize_metric}: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        display = value
        if key == "model" and isinstance(value, str) and "/" in value:
            display = value.split("/")[-1]
        print(f"    {key}: {display}")

    # Top 5 trials
    trials_sorted = sorted(completed, key=lambda t: t.value or 0.0, reverse=True)
    print(f"\n  Top 5 trials:")
    for t in trials_sorted[:5]:
        model_short = t.params.get("model", "?").split("/")[-1]
        print(
            f"    #{t.number}: {optimize_metric}={t.value:.4f} "
            f"({model_short}, {t.params.get('retriever', '?')}, "
            f"top_k={t.params.get('top_k', '?')}, "
            f"prompt={t.params.get('prompt', '?')}, "
            f"{t.params.get('dataset', '?')})"
        )

    print(f"{'=' * 70}")
