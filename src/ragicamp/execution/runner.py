"""Experiment execution runner.

This module handles experiment specification and execution:
- ExpSpec: Experiment specification dataclass
- build_specs: Generate experiment matrix from config
- run_spec: Execute a single experiment
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpSpec:
    """Experiment specification."""

    name: str
    exp_type: str
    model: str
    dataset: str
    prompt: str
    quant: str = "4bit"
    retriever: Optional[str] = None
    top_k: int = 5
    query_transform: Optional[str] = None
    reranker: Optional[str] = None
    reranker_model: Optional[str] = None
    batch_size: int = 8
    min_batch_size: int = 1


def build_specs(config: Dict[str, Any]) -> List[ExpSpec]:
    """Build experiment specs from config.

    Args:
        config: Study configuration dict

    Returns:
        List of ExpSpec objects for all experiments
    """
    specs = []
    datasets = config.get("datasets", ["nq"])
    batch = config.get("batch_size", 8)
    min_batch = config.get("min_batch_size", 1)

    # Direct experiments
    direct = config.get("direct", {})
    if direct.get("enabled"):
        for model in direct.get("models", []):
            for prompt in direct.get("prompts", ["default"]):
                for quant in direct.get("quantization", ["4bit"]):
                    if model.startswith("openai:") and quant != "4bit":
                        continue
                    for ds in datasets:
                        name = _name_direct(model, prompt, ds, quant)
                        specs.append(
                            ExpSpec(
                                name=name,
                                exp_type="direct",
                                model=model,
                                dataset=ds,
                                prompt=prompt,
                                quant=quant,
                                batch_size=batch,
                                min_batch_size=min_batch,
                            )
                        )

    # RAG experiments
    rag = config.get("rag", {})
    if rag.get("enabled"):
        query_transforms = rag.get("query_transform", ["none"])
        if not query_transforms:
            query_transforms = ["none"]

        reranker_cfgs = rag.get("reranker", {}).get(
            "configs", [{"enabled": False, "name": "none"}]
        )
        if not reranker_cfgs:
            reranker_cfgs = [{"enabled": False, "name": "none"}]

        for model in rag.get("models", []):
            for ret_config in rag.get("retrievers", []):
                ret_name = (
                    ret_config["name"] if isinstance(ret_config, dict) else ret_config
                )
                for k in rag.get("top_k_values", [5]):
                    for prompt in rag.get("prompts", ["default"]):
                        for quant in rag.get("quantization", ["4bit"]):
                            if model.startswith("openai:") and quant != "4bit":
                                continue
                            for qt in query_transforms:
                                for rr_cfg in reranker_cfgs:
                                    rr_name = (
                                        rr_cfg.get("name", "none")
                                        if rr_cfg.get("enabled")
                                        else "none"
                                    )
                                    rr_model = (
                                        rr_cfg.get("model")
                                        if rr_cfg.get("enabled")
                                        else None
                                    )
                                    for ds in datasets:
                                        name = _name_rag(
                                            model, prompt, ds, quant, ret_name, k, qt, rr_name
                                        )
                                        specs.append(
                                            ExpSpec(
                                                name=name,
                                                exp_type="rag",
                                                model=model,
                                                dataset=ds,
                                                prompt=prompt,
                                                quant=quant,
                                                retriever=ret_name,
                                                top_k=k,
                                                query_transform=qt if qt != "none" else None,
                                                reranker=rr_name if rr_name != "none" else None,
                                                reranker_model=rr_model,
                                                batch_size=batch,
                                                min_batch_size=min_batch,
                                            )
                                        )

    return specs


def _name_direct(m: str, p: str, d: str, q: str) -> str:
    """Generate experiment name for direct experiments."""
    m = m.replace(":", "_").replace("/", "_").replace("-", "")
    s = f"_{q}" if q != "4bit" else ""
    return f"direct_{m}_{p}_{d}{s}"


def _name_rag(
    m: str, p: str, d: str, q: str, r: str, k: int, qt: str = "none", rr: str = "none"
) -> str:
    """Generate experiment name for RAG experiments."""
    m = m.replace(":", "_").replace("/", "_").replace("-", "")
    s = f"_{q}" if q != "4bit" else ""

    parts = ["rag", m, r, f"k{k}"]

    if qt and qt != "none":
        parts.append(qt)

    if rr and rr != "none":
        parts.append(rr)

    parts.extend([p, d])

    name = "_".join(parts)
    if s:
        name += s

    return name


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

    # Determine action
    if health.can_resume:
        action = f"↻ Resuming from {health.resume_phase.value}"
        if health.needs_generation:
            action += f" ({health.predictions_complete}/{health.total_questions} predictions)"
        if health.needs_metrics:
            action += f" (missing: {', '.join(health.metrics_missing)})"
    else:
        action = "▶ Starting"

    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{action}")
    print(f"{'='*60}")

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
    """Run a single experiment.

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

    # In-process execution
    from ragicamp import Experiment
    from ragicamp.agents import DirectLLMAgent, FixedRAGAgent
    from ragicamp.experiment_state import ExperimentPhase, check_health
    from ragicamp.factory import ComponentFactory
    from ragicamp.utils.prompts import PromptBuilder
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

    if health.can_resume:
        action = f"↻ Resuming from {health.resume_phase.value}"
    else:
        action = "▶ Starting"

    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{action}")
    print(f"{'='*60}")

    start = time.time()

    try:
        ResourceManager.clear_gpu_memory()

        # Create dataset
        config = ComponentFactory.parse_dataset_spec(spec.dataset, limit=limit)
        dataset = ComponentFactory.create_dataset(config)
        print(f"Dataset: {len(dataset)} examples")

        # Create model
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

        return "resumed" if health.can_resume else "ran"

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
