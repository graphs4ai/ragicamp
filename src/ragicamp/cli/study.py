"""Study runner for CLI.

Handles running studies from YAML config files.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ragicamp import Experiment

# ============================================================================
# Validation
# ============================================================================

VALID_DATASETS = {"nq", "triviaqa", "hotpotqa"}
VALID_PROVIDERS = {"hf", "openai"}
VALID_QUANTIZATIONS = {"4bit", "8bit", "none"}


class ConfigError(ValueError):
    """Configuration validation error."""

    pass


def validate_model_spec(spec: str) -> None:
    """Validate model specification format.

    Args:
        spec: Model spec like 'hf:google/gemma-2b-it' or 'openai:gpt-4o-mini'

    Raises:
        ConfigError: If spec format is invalid
    """
    if ":" not in spec:
        raise ConfigError(
            f"Invalid model spec: '{spec}'. "
            f"Expected format: 'provider:model_name' (e.g., 'hf:google/gemma-2b-it', 'openai:gpt-4o-mini')"
        )
    provider = spec.split(":")[0]
    if provider not in VALID_PROVIDERS:
        raise ConfigError(
            f"Unknown model provider: '{provider}'. " f"Valid providers: {VALID_PROVIDERS}"
        )


def validate_dataset(name: str) -> None:
    """Validate dataset name.

    Args:
        name: Dataset name like 'nq', 'triviaqa', 'hotpotqa'

    Raises:
        ConfigError: If dataset name is invalid
    """
    if name not in VALID_DATASETS:
        raise ConfigError(f"Unknown dataset: '{name}'. " f"Valid datasets: {VALID_DATASETS}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate study configuration.

    Args:
        config: Study config dict

    Returns:
        List of warning messages (empty if all valid)

    Raises:
        ConfigError: If required fields are missing or invalid
    """
    warnings = []

    # Required fields
    if "name" not in config:
        raise ConfigError("Config missing required field: 'name'")

    # Validate datasets
    datasets = config.get("datasets", [])
    if not datasets:
        warnings.append("No datasets specified")
    for ds in datasets:
        validate_dataset(ds)

    # Validate direct experiments
    direct = config.get("direct", {})
    if direct.get("enabled"):
        if not direct.get("models"):
            warnings.append("Direct experiments enabled but no models specified")
        for model in direct.get("models", []):
            validate_model_spec(model)
        for q in direct.get("quantization", []):
            if q not in VALID_QUANTIZATIONS:
                raise ConfigError(f"Invalid quantization: '{q}'. Valid: {VALID_QUANTIZATIONS}")

    # Validate RAG experiments
    rag = config.get("rag", {})
    if rag.get("enabled"):
        if not rag.get("models"):
            warnings.append("RAG experiments enabled but no models specified")
        if not rag.get("retrievers"):
            warnings.append("RAG experiments enabled but no retrievers specified")
        for model in rag.get("models", []):
            validate_model_spec(model)
        for q in rag.get("quantization", []):
            if q not in VALID_QUANTIZATIONS:
                raise ConfigError(f"Invalid quantization: '{q}'. Valid: {VALID_QUANTIZATIONS}")

    return warnings


from ragicamp.agents import DirectLLMAgent, FixedRAGAgent
from ragicamp.datasets import HotpotQADataset, NaturalQuestionsDataset, TriviaQADataset
from ragicamp.factory import ComponentFactory
from ragicamp.models import HuggingFaceModel, OpenAIModel
from ragicamp.retrievers import DenseRetriever
from ragicamp.utils.resource_manager import ResourceManager

# ============================================================================
# Few-shot prompts
# ============================================================================

_FEWSHOT_CACHE: Dict[str, Any] = {}


def load_fewshot() -> Dict[str, Any]:
    """Load few-shot examples."""
    if _FEWSHOT_CACHE:
        return _FEWSHOT_CACHE
    path = Path(__file__).parent.parent.parent.parent / "conf" / "prompts" / "fewshot_examples.yaml"
    if path.exists():
        with open(path) as f:
            _FEWSHOT_CACHE.update(yaml.safe_load(f))
    return _FEWSHOT_CACHE


def get_prompt(key: str, dataset: str) -> Optional[str]:
    """Get prompt template."""
    if key == "default":
        return None
    if key == "concise":
        return "Answer with ONLY the answer, nothing else.\n\nQuestion: {question}\nAnswer:"
    if key.startswith("fewshot"):
        n = {"fewshot": 5, "fewshot_3": 3, "fewshot_1": 1}.get(key, 5)
        data = load_fewshot().get(dataset, {})
        examples = data.get("examples", [])[:n]
        style = data.get("style", "")
        stop_inst = data.get("stop_instruction", "")
        ex = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in examples)
        return f"{style}\n{stop_inst}\n\n{ex}Question: {{question}}\nAnswer:"
    return None


def get_rag_template(key: str, dataset: str) -> str:
    """Get RAG context template."""
    if key == "default":
        return "Use the context to answer. Give ONLY the answer, nothing else.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    if key == "concise":
        return "Answer with ONLY the answer from the context, nothing else.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    if key.startswith("fewshot"):
        n = {"fewshot": 5, "fewshot_3": 3, "fewshot_1": 1}.get(key, 5)
        data = load_fewshot().get(dataset, {})
        examples = data.get("examples", [])[:n]
        style = data.get("style", "")
        stop_inst = data.get("stop_instruction", "")
        knowledge_inst = data.get("knowledge_instruction", "")
        ex = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in examples)
        return f"Use the context to answer. {knowledge_inst} {style}\n{stop_inst}\n\n{ex}Context:\n{{context}}\n\nQuestion: {{query}}\nAnswer:"
    return "Use the context to answer, but you may also use your own knowledge. Give ONLY the answer, nothing else.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"


# ============================================================================
# Component creation
# ============================================================================


def create_model(spec: str, quant: str = "4bit"):
    """Create model from spec using ComponentFactory."""
    validate_model_spec(spec)
    config = ComponentFactory.parse_model_spec(spec, quantization=quant)
    return ComponentFactory.create_model(config)


def create_dataset(name: str, limit: Optional[int] = None):
    """Create dataset using ComponentFactory."""
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


# ============================================================================
# Study execution
# ============================================================================


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
    batch_size: int = 8
    min_batch_size: int = 1  # Floor for auto batch size reduction on CUDA errors


def build_specs(config: Dict[str, Any]) -> List[ExpSpec]:
    """Build experiment specs from config."""
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
                        name = _name("direct", model, prompt, ds, quant)
                        specs.append(
                            ExpSpec(
                                name,
                                "direct",
                                model,
                                ds,
                                prompt,
                                quant,
                                batch_size=batch,
                                min_batch_size=min_batch,
                            )
                        )

    # RAG experiments
    rag = config.get("rag", {})
    if rag.get("enabled"):
        for model in rag.get("models", []):
            for ret in rag.get("retrievers", []):
                for k in rag.get("top_k_values", [5]):
                    for prompt in rag.get("prompts", ["default"]):
                        for quant in rag.get("quantization", ["4bit"]):
                            if model.startswith("openai:") and quant != "4bit":
                                continue
                            for ds in datasets:
                                name = _name("rag", model, prompt, ds, quant, ret, k)
                                specs.append(
                                    ExpSpec(
                                        name,
                                        "rag",
                                        model,
                                        ds,
                                        prompt,
                                        quant,
                                        ret,
                                        k,
                                        batch_size=batch,
                                        min_batch_size=min_batch,
                                    )
                                )

    return specs


def _name(t, m, p, d, q, r=None, k=None):
    """Generate experiment name."""
    m = m.replace(":", "_").replace("/", "_").replace("-", "")
    s = f"_{q}" if q != "4bit" else ""
    return f"{t}_{m}_{p}_{d}{s}" if t == "direct" else f"{t}_{m}_{r}_k{k}_{p}_{d}{s}"


def run_spec_subprocess(
    spec: ExpSpec,
    limit: Optional[int],
    metrics: List[str],
    out: Path,
    llm_judge_config: Optional[Dict[str, Any]] = None,
    timeout: int = 7200,  # 2 hour default timeout
) -> str:
    """Run experiment in subprocess (isolated from CUDA crashes).
    
    If the subprocess crashes (e.g., CUDA error), retries with halved batch size
    until min_batch_size is reached.
    
    Returns:
        Status string: 'complete', 'resumed', 'ran', 'failed', 'crashed', 'timeout'
    """
    import subprocess
    import sys
    
    from ragicamp.experiment_state import ExperimentPhase, check_health
    
    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)
    
    # Check health before running
    health = check_health(exp_out, metrics)
    
    if health.is_complete:
        print(f"✓ {spec.name} (complete)")
        return "complete"
    
    if health.phase == ExperimentPhase.FAILED:
        print(f"✗ {spec.name} (failed previously: {health.error[:50] if health.error else 'unknown'})")
        print(f"  Retrying in subprocess...")
    
    # Determine action
    if health.can_resume:
        action = f"↻ Resuming from {health.resume_phase.value}"
        if health.needs_generation:
            action += f" ({health.predictions_complete}/{health.total_questions} predictions)"
    else:
        action = "▶ Starting"
    
    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{action}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "experiments" / "run_single_experiment.py"
    
    # Dynamic batch size reduction on crash
    current_batch_size = spec.batch_size
    min_batch_size = spec.min_batch_size
    attempt = 0
    
    while current_batch_size >= min_batch_size:
        attempt += 1
        
        # Build spec JSON for subprocess with current batch size
        spec_dict = {
            "name": spec.name,
            "exp_type": spec.exp_type,
            "model": spec.model,
            "dataset": spec.dataset,
            "prompt": spec.prompt,
            "quant": spec.quant,
            "retriever": spec.retriever,
            "top_k": spec.top_k,
            "batch_size": current_batch_size,
            "min_batch_size": min_batch_size,
        }
        
        cmd = [
            sys.executable,
            str(script_path),
            "--spec-json", json.dumps(spec_dict),
            "--output-dir", str(out),
            "--metrics", ",".join(metrics),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
        if llm_judge_config:
            cmd.extend(["--llm-judge-config", json.dumps(llm_judge_config)])
        
        if attempt > 1:
            print(f"🔄 Retrying with batch_size={current_batch_size} (attempt {attempt})")
        
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=False,  # Let output stream to console
            )
            
            if result.returncode == 0:
                return "resumed" if health.can_resume else "ran"
            else:
                # Subprocess crashed - check if we should retry with smaller batch
                if current_batch_size > min_batch_size:
                    new_batch_size = max(current_batch_size // 2, min_batch_size)
                    print(f"💥 Crashed (exit code {result.returncode}), reducing batch: {current_batch_size} → {new_batch_size}")
                    current_batch_size = new_batch_size
                    # Clear GPU memory before retry
                    try:
                        ResourceManager.clear_gpu_memory()
                    except:
                        pass
                    continue
                else:
                    # Already at min batch size, give up
                    error_log = exp_out / "error.log"
                    if not error_log.exists():
                        with open(error_log, "w") as f:
                            f.write(f"Subprocess crashed with exit code: {result.returncode}\n")
                            f.write(f"Tried batch sizes: {spec.batch_size} → {min_batch_size}\n")
                            f.write(f"This is typically a CUDA/bitsandbytes fatal error.\n")
                            f.write(f"Experiment: {spec.name}\n")
                            f.write(f"Model: {spec.model}\n")
                            f.write(f"Quantization: {spec.quant}\n")
                    
                    print(f"❌ Failed after {attempt} attempts (min batch={min_batch_size}). See: {error_log}")
                    
                    # Mark as failed in state
                    try:
                        from ragicamp.experiment_state import ExperimentState
                        state_path = exp_out / "state.json"
                        if state_path.exists():
                            state = ExperimentState.load(state_path)
                        else:
                            state = ExperimentState.new(metrics=metrics)
                        state.set_error(f"Crashed after {attempt} attempts (batch sizes {spec.batch_size}→{min_batch_size})")
                        state.save(state_path)
                    except:
                        pass
                    
                    return "crashed"
                
        except subprocess.TimeoutExpired:
            print(f"⏱️  Experiment TIMEOUT after {timeout}s")
            
            # Mark as timed out
            with open(exp_out / "error.log", "w") as f:
                f.write(f"Experiment timed out after {timeout} seconds\n")
                f.write(f"Experiment: {spec.name}\n")
            
            try:
                from ragicamp.experiment_state import ExperimentState
                state_path = exp_out / "state.json"
                if state_path.exists():
                    state = ExperimentState.load(state_path)
                else:
                    state = ExperimentState.new(metrics=metrics)
                state.set_error(f"Timeout after {timeout}s")
                state.save(state_path)
            except:
                pass
            
            return "timeout"
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            raise
    
    # Should never reach here, but just in case
    return "failed"


def run_spec(
    spec: ExpSpec,
    limit: Optional[int],
    metrics: List[str],
    out: Path,
    judge_model=None,
    llm_judge_config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    use_subprocess: bool = True,
) -> str:
    """Run single experiment with health-aware execution.

    Returns:
        Status string: 'complete', 'resumed', 'ran', 'failed', 'skipped', 'crashed', 'timeout'
    """
    # Use subprocess isolation for robustness against CUDA crashes
    if use_subprocess:
        return run_spec_subprocess(spec, limit, metrics, out, llm_judge_config=llm_judge_config)
    
    # Original in-process execution (kept for backwards compatibility)
    import time

    from ragicamp.experiment_state import ExperimentPhase, check_health

    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)

    # Check health before running
    health = check_health(exp_out, metrics)

    if health.is_complete and not force:
        print(f"✓ {spec.name} (complete)")
        return "complete"

    if health.phase == ExperimentPhase.FAILED and not force:
        print(f"✗ {spec.name} (failed: {health.error})")
        print(f"  Use --force to retry")
        return "skipped"

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

    start = time.time()

    try:
        ResourceManager.clear_gpu_memory()

        dataset = create_dataset(spec.dataset, limit)
        print(f"Dataset: {len(dataset)} examples")

        model = create_model(spec.model, spec.quant)

        if spec.exp_type == "direct":
            prompt = get_prompt(spec.prompt, spec.dataset)
            agent = DirectLLMAgent(name=spec.name, model=model, prompt_template=prompt)
        else:
            retriever = DenseRetriever.load_index(spec.retriever)
            template = get_rag_template(spec.prompt, spec.dataset)
            agent = FixedRAGAgent(
                spec.name, model, retriever, spec.top_k, context_template=template
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
            "metrics": result.metrics,
            "duration": time.time() - start,
            "timestamp": datetime.now().isoformat(),
        }
        with open(exp_out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return "resumed" if health.can_resume else "ran"

    except KeyboardInterrupt:
        # User interrupted - save state and exit gracefully
        print(f"\n⚠️  Interrupted by user")
        # Try to save state as interrupted (not failed)
        try:
            from ragicamp.experiment_state import ExperimentState
            state_path = exp_out / "state.json"
            if state_path.exists():
                state = ExperimentState.load(state_path)
                state.error = "Interrupted by user (Ctrl+C)"
                state.save(state_path)
        except:
            pass
        raise  # Re-raise to stop the study
    
    except Exception as e:
        # Check if it's the metrics incomplete error
        if type(e).__name__ == "_MetricsIncompleteError":
            print(f"⚠ Incomplete: missing metrics {getattr(e, 'missing_metrics', [])}")
            return "incomplete"
        
        # Log the error
        error_msg = str(e)
        print(f"❌ Failed: {error_msg[:100]}")
        
        # Save detailed error to state
        try:
            from ragicamp.experiment_state import ExperimentState
            state_path = exp_out / "state.json"
            
            if state_path.exists():
                state = ExperimentState.load(state_path)
            else:
                state = ExperimentState.new(metrics=metrics)
            
            state.set_error(error_msg)
            state.save(state_path)
            
            # Also save error details to a separate file for debugging
            error_log_path = exp_out / "error.log"
            with open(error_log_path, "w") as f:
                import traceback
                f.write(f"Error: {error_msg}\n\n")
                f.write(f"Experiment: {spec.name}\n")
                f.write(f"Model: {spec.model}\n")
                f.write(f"Quantization: {spec.quant}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            
            print(f"  Error details saved to: {error_log_path}")
        except Exception as save_error:
            print(f"  Warning: Could not save error state: {save_error}")
        
        # Don't print full traceback to console (it's in error.log)
        # But do log it for debugging
        import traceback
        logger = __import__('logging').getLogger(__name__)
        logger.debug("Full traceback:", exc_info=True)
        
        return "failed"
    
    finally:
        # Always clean up GPU memory after experiment (success or failure)
        try:
            ResourceManager.clear_gpu_memory()
        except:
            pass


def run_study(
    config: Dict[str, Any],
    dry_run: bool = False,
    skip_existing: bool = False,
    validate_only: bool = False,
):
    """Run complete study."""
    # Validate config first
    try:
        warnings = validate_config(config)
        for w in warnings:
            print(f"⚠️  {w}")
    except ConfigError as e:
        print(f"❌ Config error: {e}")
        return

    if validate_only:
        print("✓ Config validation passed")
        return

    print("\n" + "=" * 70)
    print(f"Study: {config['name']}")
    print(f"  {config.get('description', '')}")
    print("=" * 70)

    specs = build_specs(config)
    limit = config.get("num_questions")
    metrics = config.get("metrics", ["f1", "exact_match"])
    llm_judge_config = config.get("llm_judge")
    out = Path(config.get("output_dir", "outputs"))
    out.mkdir(parents=True, exist_ok=True)

    # Create judge model once if needed
    judge_model = None
    if llm_judge_config and "llm_judge" in metrics:
        judge_model = create_judge_model(llm_judge_config)
        print(f"LLM Judge: {llm_judge_config.get('model', 'openai:gpt-4o-mini')}")

    print(f"\nExperiments: {len(specs)}")
    print(f"Questions: {limit or 'all'}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Output: {out}")

    if dry_run:
        print("\n[DRY RUN] - Checking experiment status:")
        from ragicamp.experiment_state import check_health

        for s in specs:
            health = check_health(out / s.name, metrics)
            print(f"  {health.summary()} - {s.name}")
        return

    # Track results by status
    status_counts = {
        "complete": 0,
        "resumed": 0,
        "ran": 0,
        "failed": 0,
        "skipped": 0,
        "incomplete": 0,
        "crashed": 0,  # CUDA/subprocess crashes
        "timeout": 0,  # Experiments that timed out
    }

    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] ", end="")

        status = run_spec(spec, limit, metrics, out, judge_model, llm_judge_config=llm_judge_config, force=not skip_existing)
        status_counts[status] += 1

    # Comparison
    compare(out)

    print("\n" + "=" * 70)
    print(
        f"Done! Ran: {status_counts['ran']}, Resumed: {status_counts['resumed']}, "
        f"Complete: {status_counts['complete']}, Incomplete: {status_counts['incomplete']}, "
        f"Failed: {status_counts['failed']}, Crashed: {status_counts['crashed']}, "
        f"Timeout: {status_counts['timeout']}"
    )
    if status_counts['crashed'] > 0:
        print(f"⚠️  {status_counts['crashed']} experiments crashed (CUDA errors) - see error.log files")
    print("=" * 70)


def compare(out: Path):
    """Print comparison table."""
    results = []
    for d in out.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            with open(d / "metadata.json") as f:
                results.append(json.load(f))

    if not results:
        return

    results.sort(key=lambda x: x.get("metrics", {}).get("f1", 0), reverse=True)

    print(f"\n{'='*80}")
    print("Results (by F1)")
    print("=" * 80)
    print(f"{'Experiment':<50} {'F1':>10} {'EM':>10}")
    print("-" * 80)

    for r in results[:20]:
        n = r["name"][:48] + ".." if len(r["name"]) > 48 else r["name"]
        f1 = r.get("metrics", {}).get("f1", 0) * 100
        em = r.get("metrics", {}).get("exact_match", 0) * 100
        print(f"{n:<50} {f1:>9.1f}% {em:>9.1f}%")

    # Save study summary (for quick access to aggregated results)
    with open(out / "study_summary.json", "w") as f:
        json.dump(
            {
                "experiments": results,
                "count": len(results),
            },
            f,
            indent=2,
        )
