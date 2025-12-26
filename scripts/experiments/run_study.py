#!/usr/bin/env python3
"""Run a study from a YAML config file.

Simple config-driven experiment runner.

Usage:
    python scripts/experiments/run_study.py conf/study/simple.yaml
    python scripts/experiments/run_study.py conf/study/full.yaml --dry-run
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load study config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dataset(name: str, num_questions: Optional[int]):
    """Load dataset by name."""
    from ragicamp.datasets import NaturalQuestionsDataset, TriviaQADataset, HotpotQADataset
    
    DATASET_MAP = {
        "nq": NaturalQuestionsDataset,
        "triviaqa": TriviaQADataset,
        "hotpotqa": HotpotQADataset,
    }
    
    dataset_cls = DATASET_MAP.get(name)
    if not dataset_cls:
        raise ValueError(f"Unknown dataset: {name}")
    
    dataset = dataset_cls(split="validation")
    dataset.load()
    
    examples = dataset.examples[:num_questions] if num_questions else dataset.examples
    return examples, dataset


def get_model(model_spec: str):
    """Create model from spec like 'openai:gpt-4o-mini' or 'hf:google/gemma-2b-it'."""
    from ragicamp.models import OpenAIModel, HuggingFaceModel
    
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        # Default to openai
        provider, model_name = "openai", model_spec
    
    if provider == "openai":
        return OpenAIModel(name=model_name, temperature=0.0)
    elif provider == "hf":
        return HuggingFaceModel(model_name=model_name, load_in_4bit=True)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def compute_metrics(predictions: List[Dict], metric_names: List[str], config: Dict = None) -> Dict[str, float]:
    """Compute metrics on predictions."""
    from ragicamp.metrics import ExactMatchMetric, F1Metric
    
    results = {}
    
    preds = [p["prediction"] for p in predictions]
    refs = [p["expected_answers"] for p in predictions]
    questions = [p["question"] for p in predictions]
    
    for m in metric_names:
        try:
            if m == "f1":
                metric = F1Metric()
                results.update(metric.compute(preds, refs))
            elif m == "exact_match":
                metric = ExactMatchMetric()
                results.update(metric.compute(preds, refs))
            elif m == "bertscore":
                from ragicamp.metrics import BERTScoreMetric
                metric = BERTScoreMetric()
                results.update(metric.compute(preds, refs))
            elif m == "bleurt":
                from ragicamp.metrics import BLEURTMetric
                metric = BLEURTMetric()
                results.update(metric.compute(preds, refs))
            elif m == "llm_judge":
                from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric
                judge_cfg = config.get("llm_judge", {}) if config else {}
                judge_model = get_model(judge_cfg.get("model", "openai:gpt-4o-mini"))
                metric = LLMJudgeQAMetric(
                    judge_model=judge_model,
                    judgment_type=judge_cfg.get("type", "binary"),
                )
                # Run async
                import asyncio
                loop = asyncio.new_event_loop()
                metric_results = loop.run_until_complete(
                    metric.acompute(preds, refs, questions)
                )
                loop.close()
                results.update(metric_results)
        except ImportError as e:
            print(f"  ⚠️ Skipping {m}: {e}")
        except Exception as e:
            print(f"  ⚠️ Error computing {m}: {e}")
    
    return results


def run_direct_experiment(
    model_spec: str,
    prompt_key: str,
    dataset_name: str,
    num_questions: Optional[int],
    metric_names: List[str],
    output_dir: Path,
    config: Dict = None,
) -> Dict:
    """Run a single DirectLLM experiment."""
    from ragicamp.agents import DirectLLMAgent

    # Clean model name for path
    model_short = model_spec.replace(":", "_").replace("/", "_").replace("-", "")
    exp_name = f"direct_{model_short}_{prompt_key}_{dataset_name}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DirectLLM: {exp_name}")
    print(f"{'='*60}")

    start = time.time()

    # Load dataset
    examples, dataset = get_dataset(dataset_name, num_questions)
    print(f"Loaded {len(examples)} questions from {dataset_name}")

    # Create model and agent
    model = get_model(model_spec)
    
    prompt = None
    if prompt_key == "concise":
        prompt = "Answer briefly.\n\nQ: {question}\nA:"
    elif prompt_key == "detailed":
        prompt = "Answer the following question thoroughly.\n\nQuestion: {question}\n\nAnswer:"
    
    agent = DirectLLMAgent(name=exp_name, model=model, prompt_template=prompt)

    # Generate predictions
    predictions = []
    for i, example in enumerate(examples):
        try:
            response = agent.answer(example.question)
            predictions.append({
                "question_id": example.id,
                "question": example.question,
                "prediction": response.answer,
                "expected_answers": example.answers,
            })
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(examples)} done")
        except Exception as e:
            print(f"  Error on {i}: {e}")
            predictions.append({
                "question_id": example.id,
                "question": example.question,
                "prediction": f"[ERROR: {e}]",
                "expected_answers": example.answers,
            })

    # Cleanup model (important for HF models)
    if hasattr(model, 'unload'):
        model.unload()

    # Save predictions
    with open(exp_dir / "predictions.json", "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)

    # Compute metrics
    results = compute_metrics(predictions, metric_names, config)

    # Save results
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    duration = time.time() - start

    # Save metadata
    metadata = {
        "name": exp_name,
        "type": "direct",
        "model": model_spec,
        "prompt": prompt_key,
        "dataset": dataset_name,
        "num_questions": len(examples),
        "results": results,
        "duration": duration,
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ {exp_name}: F1={results.get('f1', 0)*100:.1f}% EM={results.get('exact_match', 0)*100:.1f}% ({duration:.1f}s)")
    return metadata


def run_rag_experiment(
    model_spec: str,
    retriever_name: str,
    top_k: int,
    dataset_name: str,
    num_questions: Optional[int],
    metric_names: List[str],
    output_dir: Path,
    config: Dict = None,
) -> Dict:
    """Run a single RAG experiment."""
    from ragicamp.agents import FixedRAGAgent
    from ragicamp.retrievers import DenseRetriever

    exp_name = f"rag_{retriever_name}_k{top_k}_{dataset_name}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RAG: {exp_name}")
    print(f"{'='*60}")

    start = time.time()

    # Load dataset
    examples, dataset = get_dataset(dataset_name, num_questions)
    print(f"Loaded {len(examples)} questions from {dataset_name}")

    # Load retriever
    retriever = DenseRetriever.load_index(retriever_name)

    # Create model and agent
    model = get_model(model_spec)
    agent = FixedRAGAgent(name=exp_name, model=model, retriever=retriever, top_k=top_k)

    # Generate predictions
    predictions = []
    for i, example in enumerate(examples):
        try:
            response = agent.answer(example.question)
            predictions.append({
                "question_id": example.id,
                "question": example.question,
                "prediction": response.answer,
                "expected_answers": example.answers,
            })
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(examples)} done")
        except Exception as e:
            print(f"  Error on {i}: {e}")
            predictions.append({
                "question_id": example.id,
                "question": example.question,
                "prediction": f"[ERROR: {e}]",
                "expected_answers": example.answers,
            })

    # Cleanup model
    if hasattr(model, 'unload'):
        model.unload()

    # Save predictions
    with open(exp_dir / "predictions.json", "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)

    # Compute metrics
    results = compute_metrics(predictions, metric_names, config)

    # Save results
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    duration = time.time() - start

    # Save metadata
    metadata = {
        "name": exp_name,
        "type": "rag",
        "model": model_spec,
        "retriever": retriever_name,
        "top_k": top_k,
        "dataset": dataset_name,
        "num_questions": len(examples),
        "results": results,
        "duration": duration,
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ {exp_name}: F1={results.get('f1', 0)*100:.1f}% EM={results.get('exact_match', 0)*100:.1f}% ({duration:.1f}s)")
    return metadata


def compare_results(output_dir: Path) -> Dict:
    """Generate comparison summary."""
    results = []
    
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        meta_path = exp_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                results.append(json.load(f))
    
    if not results:
        return {}
    
    # Sort by F1
    results.sort(key=lambda x: x.get("results", {}).get("f1", 0), reverse=True)
    
    print(f"\n{'='*60}")
    print("Results Comparison")
    print(f"{'='*60}")
    print(f"{'Experiment':<45} {'F1':>8} {'EM':>8}")
    print("-" * 65)
    
    for r in results:
        name = r["name"][:43]
        f1 = r.get("results", {}).get("f1", 0) * 100
        em = r.get("results", {}).get("exact_match", 0) * 100
        print(f"{name:<45} {f1:>7.1f}% {em:>7.1f}%")
    
    comparison = {
        "experiments": results,
        "best": results[0] if results else None,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


def run_study(config: Dict, dry_run: bool = False) -> None:
    """Run a complete study from config."""
    
    print("\n" + "=" * 70)
    print(f"📊 Study: {config['name']}")
    print(f"   {config.get('description', '')}")
    print("=" * 70)
    
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_questions = config.get("num_questions")
    datasets = config.get("datasets", ["nq"])
    metrics = config.get("metrics", ["f1", "exact_match"])
    
    # Count experiments
    direct_cfg = config.get("direct", {})
    rag_cfg = config.get("rag", {})
    
    direct_count = 0
    rag_count = 0
    
    if direct_cfg.get("enabled", False):
        direct_count = (
            len(direct_cfg.get("models", [])) *
            len(direct_cfg.get("prompts", [])) *
            len(datasets)
        )
    
    if rag_cfg.get("enabled", False):
        rag_count = (
            len(rag_cfg.get("models", [])) *
            len(rag_cfg.get("retrievers", [])) *
            len(rag_cfg.get("top_k_values", [])) *
            len(datasets)
        )
    
    print(f"\nExperiments: {direct_count} DirectLLM + {rag_count} RAG = {direct_count + rag_count} total")
    print(f"Questions per experiment: {num_questions or 'all'}")
    print(f"Output: {output_dir}")
    
    if dry_run:
        print("\n[DRY RUN] Would run these experiments:")
        if direct_cfg.get("enabled"):
            for model in direct_cfg.get("models", []):
                for prompt in direct_cfg.get("prompts", []):
                    for ds in datasets:
                        print(f"  - direct_{model}_{prompt}_{ds}")
        if rag_cfg.get("enabled"):
            for model in rag_cfg.get("models", []):
                for retr in rag_cfg.get("retrievers", []):
                    for k in rag_cfg.get("top_k_values", []):
                        for ds in datasets:
                            print(f"  - rag_{retr}_k{k}_{ds}")
        return
    
    all_results = []
    
    # Run DirectLLM
    if direct_cfg.get("enabled", False):
        print("\n" + "-" * 40)
        print("Phase 1: DirectLLM")
        print("-" * 40)
        
        for model in direct_cfg.get("models", []):
            for prompt in direct_cfg.get("prompts", []):
                for ds in datasets:
                    try:
                        result = run_direct_experiment(
                            model_spec=model,
                            prompt_key=prompt,
                            dataset_name=ds,
                            num_questions=num_questions,
                            metric_names=metrics,
                            output_dir=output_dir,
                            config=config,
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"❌ Failed: {e}")
    
    # Run RAG
    if rag_cfg.get("enabled", False):
        print("\n" + "-" * 40)
        print("Phase 2: RAG")
        print("-" * 40)
        
        for model in rag_cfg.get("models", []):
            for retr in rag_cfg.get("retrievers", []):
                for k in rag_cfg.get("top_k_values", []):
                    for ds in datasets:
                        try:
                            result = run_rag_experiment(
                                model_spec=model,
                                retriever_name=retr,
                                top_k=k,
                                dataset_name=ds,
                                num_questions=num_questions,
                                metric_names=metrics,
                                output_dir=output_dir,
                                config=config,
                            )
                            all_results.append(result)
                        except FileNotFoundError:
                            print(f"⚠️  Index not found: {retr}")
                            print("   Run: make index-simple")
                        except Exception as e:
                            print(f"❌ Failed: {e}")
    
    # Compare
    compare_results(output_dir)
    
    print("\n" + "=" * 70)
    print(f"✅ Study complete: {len(all_results)} experiments")
    print(f"   Results: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run study from config")
    parser.add_argument("config", type=Path, help="Path to study config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    run_study(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
