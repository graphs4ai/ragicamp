#!/usr/bin/env python3
"""Run a study from a YAML config file.

Usage:
    python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml
    python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --dry-run
    python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --skip-existing
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp import Experiment, run_experiments
from ragicamp.agents import DirectLLMAgent, FixedRAGAgent
from ragicamp.datasets import HotpotQADataset, NaturalQuestionsDataset, TriviaQADataset
from ragicamp.factory import ComponentFactory
from ragicamp.models import HuggingFaceModel, OpenAIModel
from ragicamp.retrievers import DenseRetriever
from ragicamp.utils.resource_manager import ResourceManager

# ============================================================================
# Few-shot prompt handling
# ============================================================================

_FEWSHOT_CACHE: Dict[str, Any] = {}


def load_fewshot_examples() -> Dict[str, Any]:
    """Load few-shot examples from config file."""
    if _FEWSHOT_CACHE:
        return _FEWSHOT_CACHE
    path = Path(__file__).parent.parent.parent / "conf" / "prompts" / "fewshot_examples.yaml"
    if path.exists():
        with open(path) as f:
            _FEWSHOT_CACHE.update(yaml.safe_load(f))
    return _FEWSHOT_CACHE


def get_prompt_template(prompt_key: str, dataset_name: str) -> Optional[str]:
    """Get prompt template for direct experiments."""
    if prompt_key == "default":
        return None
    elif prompt_key == "concise":
        return "Answer briefly.\n\nQ: {question}\nA:"
    elif prompt_key.startswith("fewshot"):
        num = {"fewshot": 5, "fewshot_3": 3, "fewshot_1": 1}.get(prompt_key, 5)
        data = load_fewshot_examples().get(dataset_name, {})
        examples = data.get("examples", [])[:num]
        style = data.get("style", "")
        ex_str = "".join(f"Q: {e['question']}\nA: {e['answer']}\n\n" for e in examples)
        return f"Answer questions. {style}\n\n{ex_str}Q: {{question}}\nA:"
    return None


def get_rag_template(prompt_key: str, dataset_name: str) -> str:
    """Get context template for RAG experiments."""
    if prompt_key == "default":
        return "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    elif prompt_key == "concise":
        return "Context:\n{context}\n\nQ: {query}\nA:"
    elif prompt_key.startswith("fewshot"):
        num = {"fewshot": 5, "fewshot_3": 3}.get(prompt_key, 5)
        data = load_fewshot_examples().get(dataset_name, {})
        examples = data.get("examples", [])[:num]
        style = data.get("style", "")
        ex_str = "".join(f"Q: {e['question']}\nA: {e['answer']}\n\n" for e in examples)
        return f"Answer using context. {style}\n\n{ex_str}Context:\n{{context}}\n\nQ: {{query}}\nA:"
    return "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


# ============================================================================
# Component creation
# ============================================================================


def create_model(model_spec: str, quantization: str = "4bit"):
    """Create model from spec string using ComponentFactory."""
    config = ComponentFactory.parse_model_spec(model_spec, quantization=quantization)
    return ComponentFactory.create_model(config)


def create_dataset(name: str, num_questions: Optional[int] = None):
    """Create dataset by name using ComponentFactory."""
    config = ComponentFactory.parse_dataset_spec(name, limit=num_questions)
    return ComponentFactory.create_dataset(config)


def create_judge_model(llm_judge_config: Optional[Dict[str, Any]]):
    """Create LLM judge model from config."""
    if not llm_judge_config:
        return None

    model_spec = llm_judge_config.get("model", "openai:gpt-4o-mini")

    if model_spec.startswith("openai:"):
        model_name = model_spec.split(":", 1)[1]
        return OpenAIModel(model_name=model_name)
    else:
        print(f"  Warning: LLM judge only supports OpenAI models, got: {model_spec}")
        return None


# ============================================================================
# Study execution
# ============================================================================


@dataclass
class ExperimentSpec:
    """Specification for a single experiment."""

    name: str
    exp_type: str  # "direct" or "rag"
    model_spec: str
    dataset_name: str
    prompt_key: str
    quantization: str = "4bit"
    retriever_name: Optional[str] = None
    top_k: int = 5
    batch_size: int = 8


def _get_batch_size(model_spec: str, config: Dict[str, Any]) -> int:
    """Get batch size for a model, checking model_batch_sizes mapping first."""
    default_batch_size = config.get("batch_size", 8)
    model_batch_sizes = config.get("model_batch_sizes", {})
    
    # Check for exact match first
    if model_spec in model_batch_sizes:
        return model_batch_sizes[model_spec]
    
    # Check for partial matches (model name without prefix)
    model_name = model_spec.split(":")[-1] if ":" in model_spec else model_spec
    for pattern, bs in model_batch_sizes.items():
        if pattern in model_name or pattern in model_spec:
            return bs
    
    return default_batch_size


def build_experiments(config: Dict[str, Any]) -> List[ExperimentSpec]:
    """Build experiment list from study config."""
    experiments = []
    datasets = config.get("datasets", ["nq"])

    # Direct experiments
    direct = config.get("direct", {})
    if direct.get("enabled", False):
        for model in direct.get("models", []):
            batch_size = _get_batch_size(model, config)
            for prompt in direct.get("prompts", ["default"]):
                for quant in direct.get("quantization", ["4bit"]):
                    if model.startswith("openai:") and quant != "4bit":
                        continue
                    for ds in datasets:
                        name = _exp_name("direct", model, prompt, ds, quant)
                        experiments.append(
                            ExperimentSpec(
                                name=name,
                                exp_type="direct",
                                model_spec=model,
                                dataset_name=ds,
                                prompt_key=prompt,
                                quantization=quant,
                                batch_size=batch_size,
                            )
                        )

    # RAG experiments
    rag = config.get("rag", {})
    if rag.get("enabled", False):
        for model in rag.get("models", []):
            batch_size = _get_batch_size(model, config)
            for retriever in rag.get("retrievers", []):
                for k in rag.get("top_k_values", [5]):
                    for prompt in rag.get("prompts", ["default"]):
                        for quant in rag.get("quantization", ["4bit"]):
                            if model.startswith("openai:") and quant != "4bit":
                                continue
                            for ds in datasets:
                                name = _exp_name("rag", model, prompt, ds, quant, retriever, k)
                                experiments.append(
                                    ExperimentSpec(
                                        name=name,
                                        exp_type="rag",
                                        model_spec=model,
                                        dataset_name=ds,
                                        prompt_key=prompt,
                                        quantization=quant,
                                        retriever_name=retriever,
                                        top_k=k,
                                        batch_size=batch_size,
                                    )
                                )

    return experiments


def _exp_name(
    exp_type: str,
    model: str,
    prompt: str,
    dataset: str,
    quant: str,
    retriever: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """Generate experiment name."""
    model_short = model.replace(":", "_").replace("/", "_").replace("-", "")
    quant_suffix = f"_{quant}" if quant != "4bit" else ""

    if exp_type == "direct":
        return f"direct_{model_short}_{prompt}_{dataset}{quant_suffix}"
    else:
        return f"rag_{model_short}_{retriever}_k{top_k}_{prompt}_{dataset}{quant_suffix}"


def run_experiment_spec(
    spec: ExperimentSpec,
    num_questions: Optional[int],
    metrics: List[str],
    output_dir: Path,
    llm_judge_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run a single experiment from spec."""
    import time

    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{'='*60}")

    start = time.time()
    exp_output = output_dir / spec.name
    exp_output.mkdir(parents=True, exist_ok=True)

    try:
        ResourceManager.clear_gpu_memory()

        # Create components
        dataset = create_dataset(spec.dataset_name, num_questions)
        print(f"Dataset: {len(dataset)} examples")

        model = create_model(spec.model_spec, spec.quantization)

        if spec.exp_type == "direct":
            prompt = get_prompt_template(spec.prompt_key, spec.dataset_name)
            agent = DirectLLMAgent(name=spec.name, model=model, prompt_template=prompt)
        else:
            retriever = DenseRetriever.load_index(spec.retriever_name)
            template = get_rag_template(spec.prompt_key, spec.dataset_name)
            agent = FixedRAGAgent(
                name=spec.name,
                model=model,
                retriever=retriever,
                top_k=spec.top_k,
                context_template=template,
            )

        # Create metrics using factory pattern
        judge_model = create_judge_model(llm_judge_config) if "llm_judge" in metrics else None
        metric_objs = ComponentFactory.create_metrics(metrics, judge_model=judge_model)

        # Create and run experiment
        exp = Experiment(
            name=spec.name,
            agent=agent,
            dataset=dataset,
            metrics=metric_objs,
            output_dir=output_dir,
            _model=model,
        )

        result = exp.run(batch_size=spec.batch_size, checkpoint_every=50, resume=True)

        duration = time.time() - start

        # Save metadata
        metadata = {
            "name": spec.name,
            "type": spec.exp_type,
            "model": spec.model_spec,
            "prompt": spec.prompt_key,
            "dataset": spec.dataset_name,
            "quantization": spec.quantization,
            "retriever": spec.retriever_name,
            "top_k": spec.top_k,
            "batch_size": spec.batch_size,
            "num_questions": result.num_examples,
            "metrics": result.metrics,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }

        with open(exp_output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except FileNotFoundError as e:
        print(f"Index not found: {e}")
        return None
    except Exception as e:
        print(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_study(config: Dict[str, Any], dry_run: bool = False, skip_existing: bool = False) -> None:
    """Run complete study from config."""
    print("\n" + "=" * 70)
    print(f"Study: {config['name']}")
    print(f"  {config.get('description', '')}")
    print("=" * 70)

    experiments = build_experiments(config)
    num_questions = config.get("num_questions")
    metrics = config.get("metrics", ["f1", "exact_match"])
    llm_judge_config = config.get("llm_judge")  # Optional LLM judge settings
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiments: {len(experiments)}")
    print(f"Questions: {num_questions or 'all'}")
    print(f"Metrics: {', '.join(metrics)}")
    if llm_judge_config and "llm_judge" in metrics:
        print(f"LLM Judge: {llm_judge_config.get('model', 'openai:gpt-4o-mini')}")
    print(f"Output: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Would run:")
        for exp in experiments:
            print(f"  - {exp.name}")
        return

    completed, skipped, failed = 0, 0, 0

    for i, spec in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")

        results_path = output_dir / spec.name / "results.json"
        if skip_existing and results_path.exists():
            print(f"Skipping {spec.name} (exists)")
            skipped += 1
            continue

        result = run_experiment_spec(spec, num_questions, metrics, output_dir, llm_judge_config)
        if result:
            completed += 1
        else:
            failed += 1

    # Generate comparison
    compare_results(output_dir)

    print("\n" + "=" * 70)
    print(f"Done! Completed: {completed}, Skipped: {skipped}, Failed: {failed}")
    print("=" * 70)


def compare_results(output_dir: Path) -> None:
    """Generate comparison summary."""
    results = []
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        meta = exp_dir / "metadata.json"
        if meta.exists():
            with open(meta) as f:
                results.append(json.load(f))

    if not results:
        return

    results.sort(key=lambda x: x.get("metrics", {}).get("f1", 0), reverse=True)

    print(f"\n{'='*80}")
    print("Results (sorted by F1)")
    print("=" * 80)
    print(f"{'Experiment':<50} {'F1':>10} {'EM':>10}")
    print("-" * 80)

    for r in results[:20]:
        name = r["name"][:48] + ".." if len(r["name"]) > 48 else r["name"]
        f1 = r.get("metrics", {}).get("f1", 0) * 100
        em = r.get("metrics", {}).get("exact_match", 0) * 100
        print(f"{name:<50} {f1:>9.1f}% {em:>9.1f}%")

    if len(results) > 20:
        print(f"... and {len(results) - 20} more")

    # Save comparison
    with open(output_dir / "comparison.json", "w") as f:
        json.dump({"experiments": results, "timestamp": datetime.now().isoformat()}, f, indent=2)


def log_to_mlflow(output_dir: Path, experiment_name: str) -> None:
    """Log results to MLflow."""
    try:
        from ragicamp.analysis import MLflowTracker, ResultsLoader
    except ImportError:
        print("⚠️  MLflow not available. Skipping.")
        return

    loader = ResultsLoader(output_dir)
    results = loader.load_all()

    if not results:
        print("No results to log")
        return

    tracker = MLflowTracker(experiment_name)
    logged = tracker.backfill_from_results(results, skip_existing=True)
    print(f"✓ Logged {logged} experiments to MLflow (experiment: {experiment_name})")


def main():
    parser = argparse.ArgumentParser(description="Run study from config")
    parser.add_argument("config", type=Path, help="Study config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-existing", action="store_true", help="Skip completed")
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: study name)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_study(config, dry_run=args.dry_run, skip_existing=args.skip_existing)

    # Log to MLflow if requested
    if args.mlflow and not args.dry_run:
        study_name = config.get("name", "ragicamp")
        mlflow_exp = args.mlflow_experiment or study_name
        output_dir = Path("outputs") / study_name
        log_to_mlflow(output_dir, mlflow_exp)


if __name__ == "__main__":
    main()
