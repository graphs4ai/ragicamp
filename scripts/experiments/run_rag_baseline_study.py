#!/usr/bin/env python3
"""Run the RAG baseline study experiment suite.

This script orchestrates RAG experiments with variations in:
- Models (Gemma, Llama)
- Datasets (NQ, TriviaQA, HotpotQA)
- RAG-specific prompts
- Retriever parameters (top_k)
- Retriever types (dense with different embedders, sparse/BM25)

**IMPORTANT**: Requires an indexed corpus. Run `make index` first.
Each experiment runs in a SEPARATE subprocess to ensure GPU memory is released.

Usage:
    # Quick test (verify RAG pipeline works)
    python scripts/experiments/run_rag_baseline_study.py --quick
    
    # Sweep top_k values with one model/dataset
    python scripts/experiments/run_rag_baseline_study.py --sweep-topk
    
    # Full RAG study (all variations)
    python scripts/experiments/run_rag_baseline_study.py --full
    
    # Dry run (show commands without executing)
    python scripts/experiments/run_rag_baseline_study.py --dry-run
"""

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple
import json
import itertools


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Models to test (same as DirectLLM baselines for fair comparison)
MODELS = ["gemma_2b_4bit", "llama3_8b"]

# Datasets
DATASETS = ["nq", "triviaqa", "hotpotqa"]

# RAG-specific prompts (emphasize using context)
RAG_PROMPTS = ["rag_concise", "rag_sentence", "rag_extractive"]

# Retrieval parameters
TOP_K_VALUES = [1, 3, 5, 10]

# Retriever configurations (requires separate indices per embedder)
RETRIEVERS = {
    "dense": "dense",           # Default MiniLM embedder
    "dense_minilm": "dense_minilm",  # Explicit MiniLM
    # "dense_mpnet": "dense_mpnet",    # MPNet (higher quality, needs index)
    # "dense_e5": "dense_e5",          # E5 (instruction-tuned, needs index)
    # "sparse": "sparse",              # BM25 (needs sparse index)
}

# Time between runs for GPU cleanup
CLEANUP_DELAY = 5


# ============================================================================
# STUDY PROFILES (predefined experiment combinations)
# ============================================================================

STUDY_PROFILES = {
    "quick": {
        "description": "Quick test - 1 model, 1 dataset, 1 top_k",
        "models": ["gemma_2b_4bit"],
        "datasets": ["nq"],
        "prompts": ["rag_concise"],
        "top_k_values": [5],
        "retrievers": ["dense"],
        "evaluation": "quick",
    },
    "sweep_topk": {
        "description": "Sweep top_k values with one model/dataset",
        "models": ["gemma_2b_4bit"],
        "datasets": ["nq"],
        "prompts": ["rag_concise"],
        "top_k_values": [1, 3, 5, 10],
        "retrievers": ["dense"],
        "evaluation": "standard",
    },
    "sweep_prompts": {
        "description": "Compare RAG prompt styles",
        "models": ["gemma_2b_4bit"],
        "datasets": ["nq"],
        "prompts": ["rag_concise", "rag_sentence", "rag_extractive"],
        "top_k_values": [5],
        "retrievers": ["dense"],
        "evaluation": "standard",
    },
    "compare_datasets": {
        "description": "Compare performance across datasets",
        "models": ["gemma_2b_4bit"],
        "datasets": ["nq", "triviaqa", "hotpotqa"],
        "prompts": ["rag_concise"],
        "top_k_values": [5],
        "retrievers": ["dense"],
        "evaluation": "standard",
    },
    "compare_models": {
        "description": "Compare models with RAG",
        "models": ["gemma_2b_4bit", "llama3_8b"],
        "datasets": ["nq"],
        "prompts": ["rag_concise"],
        "top_k_values": [5],
        "retrievers": ["dense"],
        "evaluation": "standard",
    },
    "standard": {
        "description": "Standard RAG study - 2 models, 3 datasets, 3 prompts, 4 top_k",
        "models": ["gemma_2b_4bit", "llama3_8b"],
        "datasets": ["nq", "triviaqa", "hotpotqa"],
        "prompts": ["rag_concise", "rag_sentence", "rag_extractive"],
        "top_k_values": [1, 3, 5, 10],
        "retrievers": ["dense"],
        "evaluation": "standard",
    },
    "full": {
        "description": "Full RAG study with all variations",
        "models": MODELS,
        "datasets": DATASETS,
        "prompts": RAG_PROMPTS,
        "top_k_values": TOP_K_VALUES,
        "retrievers": list(RETRIEVERS.keys()),
        "evaluation": "standard",
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_command(cmd: List[str], dry_run: bool = False, timeout: int = 1800) -> bool:
    """Run a command in a fresh subprocess and return success status."""
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")
    
    if dry_run:
        return True
    
    try:
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        result = subprocess.run(
            cmd, 
            check=True, 
            timeout=timeout,
            env=env,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ Error: Command timed out after {timeout}s")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return False


def run_single_experiment(
    overrides: Dict[str, Any],
    dry_run: bool = False,
) -> bool:
    """Run a single RAG experiment in its own subprocess."""
    cmd = ["uv", "run", "python", "-m", "ragicamp.cli.run"]
    cmd.append("experiment=baseline_study_rag")
    
    for key, value in overrides.items():
        cmd.append(f"{key}={value}")
    
    return run_command(cmd, dry_run)


def generate_experiment_configs(
    models: List[str],
    datasets: List[str],
    prompts: List[str],
    top_k_values: List[int],
    retrievers: List[str],
    evaluation: str = "standard",
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Generate individual RAG experiment configurations.
    
    Yields (description, overrides) tuples for each experiment.
    """
    for model, dataset, prompt, top_k, retriever in itertools.product(
        models, datasets, prompts, top_k_values, retrievers
    ):
        desc = f"{model}/{dataset}/{prompt}/top_k={top_k}/{retriever}"
        overrides = {
            "model": model,
            "dataset": dataset,
            "prompt": prompt,
            "agent.top_k": top_k,
            "retriever": retriever,
            "evaluation": evaluation,
        }
        yield desc, overrides


def check_index_exists(retriever: str = "dense") -> bool:
    """Check if the required retriever index exists."""
    # Map retriever config to expected artifact path
    artifact_paths = {
        "dense": "wikipedia_small_chunked",
        "dense_minilm": "wikipedia_small_chunked",
        "dense_mpnet": "wikipedia_small_chunked_mpnet",
        "dense_e5": "wikipedia_small_chunked_e5",
        "sparse": "wikipedia_small_chunked",
    }
    
    artifact_name = artifact_paths.get(retriever, "wikipedia_small_chunked")
    index_path = Path("artifacts/retrievers") / artifact_name
    
    return index_path.exists()


def check_prerequisites(retrievers: List[str]) -> bool:
    """Check that required data and indices exist."""
    issues = []
    
    # Check datasets
    data_dir = Path("data/datasets")
    for dataset in ["natural_questions", "triviaqa", "hotpotqa"]:
        pattern = f"{dataset}*.json"
        if not list(data_dir.glob(pattern)):
            issues.append(f"Dataset not found: {dataset}")
    
    # Check retriever indices
    for retriever in retrievers:
        if not check_index_exists(retriever):
            issues.append(f"Index not found for retriever: {retriever}")
    
    if issues:
        print("\n⚠️  Prerequisites missing:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("  make download-all   # Download datasets")
        print("  make index          # Build retriever index (default embedder)")
        print("\nFor additional embedders, create indices manually:")
        print("  python scripts/data/index.py --embedding-model all-mpnet-base-v2 --name wikipedia_small_chunked_mpnet")
        return False
    
    return True


# ============================================================================
# MAIN STUDY RUNNER
# ============================================================================

def run_rag_study(
    models: List[str],
    datasets: List[str],
    prompts: List[str],
    top_k_values: List[int],
    retrievers: List[str],
    evaluation: str = "standard",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run RAG baseline study.
    
    Each experiment runs in a SEPARATE subprocess to ensure GPU memory
    is fully released between runs.
    """
    total_runs = len(models) * len(datasets) * len(prompts) * len(top_k_values) * len(retrievers)
    
    print("\n" + "="*70)
    print("RAG BASELINE STUDY (SEQUENTIAL - separate processes)")
    print("="*70)
    print(f"  Models:      {models}")
    print(f"  Datasets:    {datasets}")
    print(f"  Prompts:     {prompts}")
    print(f"  top_k:       {top_k_values}")
    print(f"  Retrievers:  {retrievers}")
    print(f"  Evaluation:  {evaluation}")
    print(f"  Total runs:  {total_runs}")
    print("="*70)
    
    results = {"passed": 0, "failed": 0, "runs": []}
    
    # Generate all experiment configs
    configs = list(generate_experiment_configs(
        models=models,
        datasets=datasets,
        prompts=prompts,
        top_k_values=top_k_values,
        retrievers=retrievers,
        evaluation=evaluation,
    ))
    
    # Run each experiment in a separate subprocess
    for i, (desc, overrides) in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"Run {i}/{total_runs}: {desc}")
        print(f"{'='*70}")
        
        success = run_single_experiment(overrides, dry_run=dry_run)
        
        results["runs"].append({"desc": desc, "success": success, **overrides})
        if success:
            results["passed"] += 1
            print(f"✅ {desc}: PASSED")
        else:
            results["failed"] += 1
            print(f"❌ {desc}: FAILED")
        
        # Wait between runs for GPU memory cleanup
        if not dry_run and i < total_runs:
            print(f"\n⏳ Waiting {CLEANUP_DELAY}s for GPU memory cleanup...")
            time.sleep(CLEANUP_DELAY)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG baseline study experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Study Profiles:
  --quick           Quick test (1 model, 1 dataset, 1 run)
  --sweep-topk      Sweep top_k values (1, 3, 5, 10)
  --sweep-prompts   Compare RAG prompt styles
  --compare-datasets Compare performance across datasets
  --compare-models  Compare models with RAG
  --standard        Standard study (2 models × 3 datasets × 3 prompts × 4 top_k)
  --full            Full study with all variations

Examples:
    # Quick verification
    python scripts/experiments/run_rag_baseline_study.py --quick
    
    # Standard RAG study
    python scripts/experiments/run_rag_baseline_study.py --standard
    
    # Custom configuration
    python scripts/experiments/run_rag_baseline_study.py \\
        --models gemma_2b_4bit \\
        --datasets nq triviaqa \\
        --top-k 3 5 10
        """,
    )
    
    # Study profile selection
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--quick", action="store_true", help="Quick test")
    profile.add_argument("--sweep-topk", action="store_true", help="Sweep top_k values")
    profile.add_argument("--sweep-prompts", action="store_true", help="Compare prompts")
    profile.add_argument("--compare-datasets", action="store_true", help="Compare datasets")
    profile.add_argument("--compare-models", action="store_true", help="Compare models")
    profile.add_argument("--standard", action="store_true", help="Standard study")
    profile.add_argument("--full", action="store_true", help="Full study")
    
    # Custom configuration (overrides profiles)
    parser.add_argument("--models", nargs="+", default=None, help="Models to test")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to test")
    parser.add_argument("--prompts", nargs="+", default=None, help="Prompt styles")
    parser.add_argument("--top-k", nargs="+", type=int, default=None, help="top_k values")
    parser.add_argument("--retrievers", nargs="+", default=None, help="Retriever configs")
    parser.add_argument("--evaluation", default=None, help="Evaluation mode")
    
    # Execution control
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--skip-check", action="store_true", help="Skip prerequisite check")
    
    args = parser.parse_args()
    
    # Determine which profile to use
    profile_name = None
    if args.quick:
        profile_name = "quick"
    elif args.sweep_topk:
        profile_name = "sweep_topk"
    elif args.sweep_prompts:
        profile_name = "sweep_prompts"
    elif args.compare_datasets:
        profile_name = "compare_datasets"
    elif args.compare_models:
        profile_name = "compare_models"
    elif args.standard:
        profile_name = "standard"
    elif args.full:
        profile_name = "full"
    else:
        # Default to standard if no profile and no custom args
        if not any([args.models, args.datasets, args.prompts, args.top_k, args.retrievers]):
            parser.print_help()
            sys.exit(1)
    
    # Get configuration from profile or custom args
    if profile_name:
        profile_config = STUDY_PROFILES[profile_name]
        models = args.models or profile_config["models"]
        datasets = args.datasets or profile_config["datasets"]
        prompts = args.prompts or profile_config["prompts"]
        top_k_values = args.top_k or profile_config["top_k_values"]
        retrievers = args.retrievers or profile_config["retrievers"]
        evaluation = args.evaluation or profile_config["evaluation"]
        print(f"\n📋 Using profile: {profile_name}")
        print(f"   {profile_config['description']}")
    else:
        models = args.models or MODELS
        datasets = args.datasets or DATASETS
        prompts = args.prompts or RAG_PROMPTS
        top_k_values = args.top_k or TOP_K_VALUES
        retrievers = args.retrievers or ["dense"]
        evaluation = args.evaluation or "standard"
    
    # Check prerequisites
    if not args.skip_check and not args.dry_run:
        if not check_prerequisites(retrievers):
            print("\nRun with --skip-check to ignore these warnings")
            sys.exit(1)
    
    # Track results
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("RAGiCamp RAG Baseline Study")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run the study
    results = run_rag_study(
        models=models,
        datasets=datasets,
        prompts=prompts,
        top_k_values=top_k_values,
        retrievers=retrievers,
        evaluation=evaluation,
        dry_run=args.dry_run,
    )
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Duration: {duration}")
    print(f"Total runs: {results['passed'] + results['failed']}")
    print(f"  ✅ Passed: {results['passed']}")
    print(f"  ❌ Failed: {results['failed']}")
    
    # Save summary to JSON
    if not args.dry_run:
        summary = {
            "study_type": "rag_baseline",
            "profile": profile_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "config": {
                "models": models,
                "datasets": datasets,
                "prompts": prompts,
                "top_k_values": top_k_values,
                "retrievers": retrievers,
                "evaluation": evaluation,
            },
            "total_passed": results["passed"],
            "total_failed": results["failed"],
            "runs": results["runs"],
        }
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        summary_path = output_dir / f"rag_baseline_study_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSummary saved to: {summary_path}")
        print("\nResults saved to: outputs/")
        print("Compare with: make compare")
        print("Generate report: make report")
    
    print("="*70)
    
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
