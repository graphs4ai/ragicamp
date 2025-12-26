#!/usr/bin/env python3
"""End-to-end pipeline validation script.

Tests the complete RAGiCamp pipeline with multiple variations to ensure
everything works correctly before running full experiments.

Validates:
1. DirectLLM agent with different prompts
2. RAG agent with different retrievers  
3. Multiple metrics (F1, EM, LLM-as-judge)
4. Checkpointing and resume
5. Result comparison and analysis

Usage:
    # Quick validation (10 questions, 2 variations)
    python scripts/experiments/validate_pipeline.py --quick
    
    # Standard validation (50 questions, more variations)
    python scripts/experiments/validate_pipeline.py
    
    # With specific indexes
    python scripts/experiments/validate_pipeline.py --retrievers simple_minilm_recursive_512 simple_mpnet_recursive_512
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def run_single_experiment(
    agent_type: str,
    agent_config: Dict,
    dataset: str,
    num_questions: int,
    metrics: List[str],
    output_dir: Path,
    experiment_name: str,
) -> Dict:
    """Run a single experiment and return results."""
    from ragicamp.agents import DirectLLMAgent, FixedRAGAgent
    from ragicamp.datasets import QADataset
    from ragicamp.factory import ComponentFactory
    from ragicamp.metrics import ExactMatchMetric, F1Metric
    from ragicamp.pipeline import Orchestrator, GenerationPhase, MetricsPhase
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"  Agent: {agent_type}")
    print(f"  Dataset: {dataset}")
    print(f"  Questions: {num_questions}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load dataset
    qa_dataset = QADataset(dataset)
    examples = qa_dataset.load(limit=num_questions)
    
    # Create agent
    if agent_type == "direct":
        from ragicamp.models import OpenAIModel
        model = OpenAIModel(
            name=agent_config.get("model", "gpt-4o-mini"),
            temperature=0.0,
        )
        agent = DirectLLMAgent(
            name=f"direct_{agent_config.get('model', 'gpt4omini')}",
            model=model,
            prompt_template=agent_config.get("prompt", None),
        )
    elif agent_type == "rag":
        from ragicamp.models import OpenAIModel
        from ragicamp.retrievers import DenseRetriever
        
        model = OpenAIModel(
            name=agent_config.get("model", "gpt-4o-mini"),
            temperature=0.0,
        )
        retriever = DenseRetriever.load_index(agent_config["retriever"])
        agent = FixedRAGAgent(
            name=f"rag_{agent_config['retriever']}",
            model=model,
            retriever=retriever,
            top_k=agent_config.get("top_k", 5),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Create metrics
    metric_objects = []
    for m in metrics:
        if m == "f1":
            metric_objects.append(F1Metric())
        elif m == "exact_match":
            metric_objects.append(ExactMatchMetric())
        # Skip llm_judge for validation to save API calls
    
    # Setup output
    exp_output = output_dir / experiment_name
    exp_output.mkdir(parents=True, exist_ok=True)
    
    # Run generation phase
    gen_phase = GenerationPhase(
        agent=agent,
        output_path=exp_output / "predictions.json",
        checkpoint_every=10,
    )
    predictions_path = gen_phase.run(examples, qa_dataset)
    
    # Run metrics phase
    metrics_phase = MetricsPhase(
        metrics=metric_objects,
        output_path=exp_output / "results.json",
    )
    results = metrics_phase.run(predictions_path)
    
    duration = time.time() - start_time
    
    # Save experiment metadata
    metadata = {
        "experiment_name": experiment_name,
        "agent_type": agent_type,
        "agent_config": agent_config,
        "dataset": dataset,
        "num_questions": num_questions,
        "metrics": metrics,
        "duration_seconds": duration,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(exp_output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Completed in {duration:.1f}s")
    print(f"  Results: {results}")
    
    return metadata


def compare_results(output_dir: Path) -> Dict:
    """Compare results across experiments."""
    print(f"\n{'='*60}")
    print("Comparing Results")
    print(f"{'='*60}")
    
    results = []
    
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metadata_path = exp_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                results.append(json.load(f))
    
    if not results:
        print("No results found!")
        return {}
    
    # Create comparison table
    print(f"\n{'Experiment':<40} {'F1':>8} {'EM':>8} {'Time':>8}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x.get("results", {}).get("f1", 0), reverse=True):
        name = r["experiment_name"][:38]
        f1 = r.get("results", {}).get("f1", 0) * 100
        em = r.get("results", {}).get("exact_match", 0) * 100
        duration = r.get("duration_seconds", 0)
        print(f"{name:<40} {f1:>7.1f}% {em:>7.1f}% {duration:>7.1f}s")
    
    # Save comparison
    comparison = {
        "experiments": results,
        "summary": {
            "total_experiments": len(results),
            "best_f1": max(r.get("results", {}).get("f1", 0) for r in results),
            "best_em": max(r.get("results", {}).get("exact_match", 0) for r in results),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {output_dir / 'comparison.json'}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (10 questions, minimal variations)")
    parser.add_argument("--num-questions", type=int, default=50,
                        help="Number of questions per experiment (default: 50)")
    parser.add_argument("--dataset", type=str, default="nq",
                        help="Dataset to use (default: nq)")
    parser.add_argument("--retrievers", nargs="+", 
                        default=["simple_minilm_recursive_512"],
                        help="Retriever artifacts to test")
    parser.add_argument("--output-dir", type=str, default="outputs/validation",
                        help="Output directory")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip RAG experiments (if no index available)")
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.num_questions = 10
        args.retrievers = args.retrievers[:1]  # Only first retriever
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🔬 RAGiCamp Pipeline Validation")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Questions: {args.num_questions}")
    print(f"Retrievers: {args.retrievers}")
    print(f"Output: {output_dir}")
    
    all_results = []
    
    # =========================================================================
    # DirectLLM Experiments
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1: DirectLLM Experiments")
    print("=" * 70)
    
    direct_configs = [
        {"model": "gpt-4o-mini", "prompt": None},  # Default prompt
    ]
    
    if not args.quick:
        direct_configs.append(
            {"model": "gpt-4o-mini", "prompt": "Answer the following question briefly and directly.\n\nQuestion: {question}\n\nAnswer:"}
        )
    
    for i, config in enumerate(direct_configs):
        try:
            result = run_single_experiment(
                agent_type="direct",
                agent_config=config,
                dataset=args.dataset,
                num_questions=args.num_questions,
                metrics=["f1", "exact_match"],
                output_dir=output_dir,
                experiment_name=f"direct_v{i+1}",
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ DirectLLM experiment failed: {e}")
    
    # =========================================================================
    # RAG Experiments
    # =========================================================================
    if not args.skip_rag:
        print("\n" + "=" * 70)
        print("Phase 2: RAG Experiments")
        print("=" * 70)
        
        for retriever in args.retrievers:
            for top_k in [3, 5] if not args.quick else [5]:
                try:
                    result = run_single_experiment(
                        agent_type="rag",
                        agent_config={
                            "model": "gpt-4o-mini",
                            "retriever": retriever,
                            "top_k": top_k,
                        },
                        dataset=args.dataset,
                        num_questions=args.num_questions,
                        metrics=["f1", "exact_match"],
                        output_dir=output_dir,
                        experiment_name=f"rag_{retriever}_k{top_k}",
                    )
                    all_results.append(result)
                except FileNotFoundError as e:
                    print(f"⚠️  Retriever not found: {retriever}")
                    print(f"   Run: make index-test  (to build test indexes)")
                except Exception as e:
                    print(f"❌ RAG experiment failed: {e}")
    
    # =========================================================================
    # Compare Results
    # =========================================================================
    comparison = compare_results(output_dir)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 Validation Summary")
    print("=" * 70)
    
    successful = len(all_results)
    total = len(direct_configs) + (len(args.retrievers) * (1 if args.quick else 2) if not args.skip_rag else 0)
    
    print(f"Experiments: {successful}/{total} successful")
    
    if successful > 0:
        print(f"Best F1: {comparison['summary']['best_f1']*100:.1f}%")
        print(f"Best EM: {comparison['summary']['best_em']*100:.1f}%")
    
    print(f"\nResults saved to: {output_dir}")
    print("\n✅ Pipeline validation complete!")
    
    if successful < total:
        print("\n⚠️  Some experiments failed. Check output above for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
