"""Experiment execution runner.

This module handles experiment execution with phase-aware dispatching:
- run_spec: Dispatch to appropriate runner based on phase
- run_generation: Run generation phase (needs model, GPU)
- run_metrics_only: Run metrics phase (no model needed)

The phase-aware design ensures resources are only loaded when needed:
- GENERATING phase: Loads model, agent, retriever
- COMPUTING_METRICS phase: Only loads predictions file + metrics

For experiment specification and building, see the spec/ package.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragicamp.core.logging import get_logger
from ragicamp.spec import ExperimentSpec, build_specs, name_direct, name_rag

logger = get_logger(__name__)


# Backward compatibility alias
ExpSpec = ExperimentSpec


# =============================================================================
# Phase-Aware Runners
# =============================================================================


def _log_gpu_mem(label: str, always: bool = False) -> None:
    """Log GPU memory usage for debugging."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            if always or alloc > 0.1:  # Only log if significant
                print(f"  [GPU] {label}: {alloc:.2f} GiB allocated")
    except Exception:
        pass


def run_metrics_only(
    exp_name: str,
    output_path: Path,
    metrics: List[str],
    judge_model: Any = None,
) -> str:
    """Run metrics computation only - no model needed.

    This is the lightweight path when predictions already exist.
    Uses minimal GPU memory (only for GPU-based metrics like BERTScore).

    Args:
        exp_name: Experiment name
        output_path: Path to experiment output directory
        metrics: List of metric names to compute
        judge_model: Optional LLM judge model for llm_judge metric

    Returns:
        Status string
    """
    from ragicamp.experiment_state import ExperimentPhase, detect_state
    from ragicamp.factory import ComponentFactory
    from ragicamp.metrics import compute_metrics_batched

    print(f"  📊 Metrics-only mode (no model loaded)")

    predictions_path = output_path / "predictions.json"
    if not predictions_path.exists():
        print(f"  ❌ No predictions file found at {predictions_path}")
        return "failed"

    # Load predictions
    with open(predictions_path) as f:
        data = json.load(f)

    preds = data["predictions"]
    predictions = [p["prediction"] for p in preds]
    references = [p["expected"] for p in preds]
    questions = [p["question"] for p in preds]

    # Load state to track computed metrics
    state_path = output_path / "state.json"
    state = detect_state(output_path, metrics)

    # Create metric objects
    metric_objs = ComponentFactory.create_metrics(metrics, judge_model=judge_model)

    # Callback to save state after each metric
    def on_metric_complete(metric_name: str) -> None:
        if metric_name not in state.metrics_computed:
            state.metrics_computed.append(metric_name)
            state.save(state_path)

    # Use shared metrics computation
    aggregate_results, per_item_metrics, computed, failed = compute_metrics_batched(
        metrics=metric_objs,
        predictions=predictions,
        references=references,
        questions=questions,
        already_computed=state.metrics_computed,
        on_metric_complete=on_metric_complete,
    )

    # Update predictions with per-item metrics
    for i, pred in enumerate(preds):
        if "metrics" not in pred:
            pred["metrics"] = {}
        for metric_name, scores in per_item_metrics.items():
            if i < len(scores):
                pred["metrics"][metric_name] = scores[i]

    # Merge with existing aggregate metrics
    existing_metrics = data.get("aggregate_metrics", {})
    existing_metrics.update(aggregate_results)
    data["aggregate_metrics"] = existing_metrics

    # Save updated predictions
    with open(predictions_path, "w") as f:
        json.dump(data, f, indent=2)

    # Check if all metrics are done
    missing = set(metrics) - set(state.metrics_computed)
    if missing:
        print(f"  ⚠ Missing metrics: {list(missing)}")
        return "incomplete"

    # Mark complete
    state.phase = ExperimentPhase.COMPLETE
    state.save(state_path)

    # Save results.json
    results = {
        "name": exp_name,
        "metrics": existing_metrics,
        "num_predictions": len(preds),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ All metrics computed")
    return "ran"


def run_generation(
    spec: ExpSpec,
    limit: Optional[int],
    metrics: List[str],
    out: Path,
    judge_model: Any = None,
) -> str:
    """Run full experiment including generation.

    This path loads the model and agent, runs generation, then metrics.

    Args:
        spec: Experiment specification
        limit: Max examples to process
        metrics: List of metric names
        out: Output directory
        judge_model: Optional LLM judge model

    Returns:
        Status string
    """
    from ragicamp import Experiment
    from ragicamp.agents import DirectLLMAgent, FixedRAGAgent
    from ragicamp.factory import ComponentFactory
    from ragicamp.utils.prompts import PromptBuilder
    from ragicamp.utils.resource_manager import ResourceManager

    exp_out = out / spec.name

    start = time.time()

    # Note: GPU memory is cleared by parent before subprocess spawn (line ~302)
    # No need to clear again here - subprocess starts with clean GPU state

    # Create dataset
    config = ComponentFactory.parse_dataset_spec(spec.dataset, limit=limit)
    dataset = ComponentFactory.create_dataset(config)
    print(f"  Dataset: {len(dataset)} examples")

    # Create model
    print(f"  Loading model: {spec.model}")
    model_config = ComponentFactory.parse_model_spec(spec.model, quantization=spec.quant)
    model = ComponentFactory.create_model(model_config)

    # Get prompt builder
    prompt_builder = PromptBuilder.from_config(spec.prompt, dataset=spec.dataset)

    if spec.exp_type == "direct":
        agent = DirectLLMAgent(name=spec.name, model=model, prompt_builder=prompt_builder)
    else:
        # Load retriever
        from ragicamp.factory import load_retriever

        retriever = load_retriever(spec.retriever)

        # Create query transformer if specified
        query_transformer = None
        if spec.query_transform:
            from ragicamp.factory import create_query_transformer

            query_transformer = create_query_transformer(spec.query_transform, model)

        # Create reranker if specified
        reranker = None
        if spec.reranker and spec.reranker_model:
            from ragicamp.factory import create_reranker

            reranker = create_reranker(spec.reranker_model)

        agent = FixedRAGAgent(
            spec.name,
            model,
            retriever,
            spec.top_k,
            prompt_builder=prompt_builder,
            query_transformer=query_transformer,
            reranker=reranker,
        )

    metric_objs = ComponentFactory.create_metrics(metrics, judge_model=judge_model)

    exp = Experiment(spec.name, agent, dataset, metric_objs, out, _model=model)
    result = exp.run(
        batch_size=spec.batch_size,
        min_batch_size=spec.min_batch_size,
        checkpoint_every=50,
        resume=True,
    )

    # Save metadata
    meta = {
        "name": spec.name,
        "type": spec.exp_type,
        "model": spec.model,
        "prompt": spec.prompt,
        "dataset": spec.dataset,
        "quantization": spec.quant,
        "retriever": spec.retriever,
        "top_k": spec.top_k,
        "query_transform": spec.query_transform,
        "reranker": spec.reranker,
        "reranker_model": spec.reranker_model,
        "metrics": result.metrics,
        "duration": time.time() - start,
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return "ran"


# =============================================================================
# Main Entry Points
# =============================================================================


def run_spec_subprocess(
    spec: ExpSpec,
    limit: Optional[int],
    metrics: List[str],
    out: Path,
    llm_judge_config: Optional[Dict[str, Any]] = None,
    timeout: int = 7200,
) -> str:
    """Run experiment in subprocess for CUDA crash isolation.

    Args:
        spec: Experiment specification
        limit: Max examples to process
        metrics: List of metric names
        out: Output directory
        llm_judge_config: LLM judge configuration
        timeout: Timeout in seconds

    Returns:
        Status string
    """
    import os
    
    # Ensure TensorFlow doesn't grab all GPU memory in subprocess
    # These are inherited by the subprocess
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # CRITICAL: Clear GPU memory in parent process before spawning subprocess
    # The parent may have loaded embedding models for index checking.
    # If we don't clear, the subprocess will see a full GPU and OOM.
    from ragicamp.utils.resource_manager import ResourceManager
    ResourceManager.clear_gpu_memory()
    
    from ragicamp.experiment_state import ExperimentPhase, check_health

    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)

    # Check health before running
    health = check_health(exp_out, metrics)

    if health.is_complete:
        print(f"✓ {spec.name} (complete)")
        return "complete"

    if health.phase == ExperimentPhase.FAILED:
        print(f"⚠ {spec.name} (previously failed, retrying)")

    # Note: Don't print header here - the subprocess will print it
    # This avoids duplicate output

    script_path = (
        Path(__file__).parent.parent.parent.parent
        / "scripts"
        / "experiments"
        / "run_single_experiment.py"
    )

    current_batch_size = spec.batch_size
    min_batch_size = spec.min_batch_size
    attempt = 0
    max_retries = 5

    while current_batch_size >= min_batch_size:
        attempt += 1

        spec_dict = {
            "name": spec.name,
            "exp_type": spec.exp_type,
            "model": spec.model,
            "dataset": spec.dataset,
            "prompt": spec.prompt,
            "quant": spec.quant,
            "retriever": spec.retriever,
            "top_k": spec.top_k,
            "query_transform": spec.query_transform,
            "reranker": spec.reranker,
            "reranker_model": spec.reranker_model,
            "batch_size": current_batch_size,
            "min_batch_size": min_batch_size,
        }

        cmd = [
            sys.executable,
            str(script_path),
            "--spec-json",
            json.dumps(spec_dict),
            "--output-dir",
            str(out),
            "--metrics",
            ",".join(metrics),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
        if llm_judge_config:
            cmd.extend(["--llm-judge-config", json.dumps(llm_judge_config)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                timeout=timeout,
                check=False,
            )

            if result.returncode == 0:
                return "ran"
            else:
                current_batch_size = max(current_batch_size // 2, min_batch_size)
                if current_batch_size < min_batch_size:
                    break
                print(f"  Retrying with batch_size={current_batch_size}")

        except subprocess.TimeoutExpired:
            print(f"  ⏱ Timeout after {timeout}s")
            return "timeout"

        if attempt >= max_retries:
            break

    return "failed"


def run_spec(
    spec: ExpSpec,
    limit: Optional[int],
    metrics: List[str],
    out: Path,
    judge_model: Any = None,
    llm_judge_config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    use_subprocess: bool = True,
) -> str:
    """Run a single experiment with phase-aware dispatching.

    Dispatches to the appropriate runner based on experiment phase:
    - COMPUTING_METRICS: Uses run_metrics_only() (no model loaded)
    - Other phases: Uses run_generation() (full model load)

    Args:
        spec: Experiment specification
        limit: Max examples to process
        metrics: List of metric names
        out: Output directory
        judge_model: LLM judge model instance
        llm_judge_config: LLM judge configuration
        force: Force re-run even if complete/failed
        use_subprocess: Run in subprocess for isolation

    Returns:
        Status string
    """
    if use_subprocess:
        return run_spec_subprocess(
            spec, limit, metrics, out, llm_judge_config=llm_judge_config
        )

    # In-process execution with phase-aware dispatching
    from ragicamp.experiment_state import ExperimentPhase, check_health
    from ragicamp.utils.resource_manager import ResourceManager

    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)

    health = check_health(exp_out, metrics)

    if health.is_complete and not force:
        print(f"✓ {spec.name} (complete)")
        return "complete"

    if health.phase == ExperimentPhase.FAILED and not force:
        print(f"✗ {spec.name} (failed: {health.error})")
        print(f"  Use --force to retry")
        return "skipped"

    # Determine which phase we're resuming from
    if health.can_resume:
        action = f"↻ Resuming from {health.resume_phase.value}"
    else:
        action = "▶ Starting"

    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{action}")
    print(f"{'='*60}")

    try:
        # Note: GPU memory cleared by parent before subprocess spawn
        # Phase-aware dispatch: metrics-only vs full generation
        if (
            health.can_resume
            and health.resume_phase == ExperimentPhase.COMPUTING_METRICS
        ):
            # Metrics-only path - no model needed
            return run_metrics_only(
                exp_name=spec.name,
                output_path=exp_out,
                metrics=metrics,
                judge_model=judge_model,
            )
        else:
            # Full path - load model, run generation + metrics
            return run_generation(
                spec=spec,
                limit=limit,
                metrics=metrics,
                out=out,
                judge_model=judge_model,
            )

    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return "interrupted"

    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Save error info
        with open(exp_out / "error.log", "w") as f:
            import traceback

            f.write(f"Error: {e}\n\n")
            f.write(f"Experiment: {spec.name}\n")
            f.write(f"Model: {spec.model}\n")
            f.write(f"Quantization: {spec.quant}\n\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        return "failed"
