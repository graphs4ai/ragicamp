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
from ragicamp.metrics import ExactMatchMetric, F1Metric
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
        ex = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in examples)
        return f"Use the context to answer. {style}\n{stop_inst}\n\n{ex}Context:\n{{context}}\n\nQuestion: {{query}}\nAnswer:"
    return "Use the context to answer. Give ONLY the answer, nothing else.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"


# ============================================================================
# Component creation
# ============================================================================


def create_model(spec: str, quant: str = "4bit"):
    """Create model from spec."""
    validate_model_spec(spec)
    provider, name = spec.split(":", 1)
    if provider == "openai":
        return OpenAIModel(name=name, temperature=0.0)
    return HuggingFaceModel(
        model_name=name,
        load_in_4bit=(quant == "4bit"),
        load_in_8bit=(quant == "8bit"),
    )


def create_dataset(name: str, limit: Optional[int] = None):
    """Create dataset."""
    datasets = {
        "nq": NaturalQuestionsDataset,
        "triviaqa": TriviaQADataset,
        "hotpotqa": HotpotQADataset,
    }
    ds = datasets[name](split="validation")
    if limit:
        ds.examples = ds.examples[:limit]
    return ds


def create_metrics(names: List[str]):
    """Create metrics."""
    result = []
    for name in names:
        if name == "f1":
            result.append(F1Metric())
        elif name == "exact_match":
            result.append(ExactMatchMetric())
    return result


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


def build_specs(config: Dict[str, Any]) -> List[ExpSpec]:
    """Build experiment specs from config."""
    specs = []
    datasets = config.get("datasets", ["nq"])
    batch = config.get("batch_size", 8)

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
                            ExpSpec(name, "direct", model, ds, prompt, quant, batch_size=batch)
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
                                    ExpSpec(name, "rag", model, ds, prompt, quant, ret, k, batch)
                                )

    return specs


def _name(t, m, p, d, q, r=None, k=None):
    """Generate experiment name."""
    m = m.replace(":", "_").replace("/", "_").replace("-", "")
    s = f"_{q}" if q != "4bit" else ""
    return f"{t}_{m}_{p}_{d}{s}" if t == "direct" else f"{t}_{m}_{r}_k{k}_{p}_{d}{s}"


def run_spec(spec: ExpSpec, limit: Optional[int], metrics: List[str], out: Path) -> bool:
    """Run single experiment."""
    import time

    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{'='*60}")

    start = time.time()
    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)

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

        metric_objs = create_metrics(metrics)

        exp = Experiment(spec.name, agent, dataset, metric_objs, out, _model=model)
        result = exp.run(batch_size=spec.batch_size, checkpoint_every=50, resume=True)

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

        return True

    except Exception as e:
        print(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


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
    out = Path(config.get("output_dir", "outputs"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiments: {len(specs)}")
    print(f"Questions: {limit or 'all'}")
    print(f"Output: {out}")

    if dry_run:
        print("\n[DRY RUN]")
        for s in specs:
            print(f"  - {s.name}")
        return

    ok, skip, fail = 0, 0, 0
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] ", end="")

        if skip_existing and (out / spec.name / "results.json").exists():
            print(f"Skipping {spec.name}")
            skip += 1
            continue

        if run_spec(spec, limit, metrics, out):
            ok += 1
        else:
            fail += 1

    # Comparison
    compare(out)

    print("\n" + "=" * 70)
    print(f"Done! OK: {ok}, Skipped: {skip}, Failed: {fail}")
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

    with open(out / "comparison.json", "w") as f:
        json.dump({"experiments": results}, f, indent=2)
