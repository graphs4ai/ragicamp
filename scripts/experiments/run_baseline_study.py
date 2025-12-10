#!/usr/bin/env python3
"""Run the baseline study experiment suite.

This script orchestrates the full baseline comparison:
- DirectLLM vs FixedRAG
- Multiple datasets (NQ, TriviaQA, HotpotQA)
- Multiple models (Gemma 2B, Phi-3, Llama 3 8B)
- Prompt variations
- RAG parameter sweeps

Usage:
    # Quick test (1 model, 1 dataset, 10 examples)
    python scripts/experiments/run_baseline_study.py --quick
    
    # DirectLLM only
    python scripts/experiments/run_baseline_study.py --direct-only
    
    # RAG only
    python scripts/experiments/run_baseline_study.py --rag-only
    
    # Full study
    python scripts/experiments/run_baseline_study.py --full
    
    # Dry run (show commands without executing)
    python scripts/experiments/run_baseline_study.py --dry-run
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json


# Experiment configurations
MODELS = ["gemma_2b_4bit", "phi3"]  # Add "llama3_8b" for full study
DATASETS = ["nq", "triviaqa", "hotpotqa"]
PROMPTS = ["concise", "sentence", "explained"]
TOP_K_VALUES = [1, 3, 5, 10]


def run_command(cmd: List[str], dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")
    
    if dry_run:
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False


def run_hydra_experiment(
    experiment: str,
    overrides: Dict[str, Any],
    dry_run: bool = False,
    multirun: bool = False,
) -> bool:
    """Run a Hydra experiment with overrides."""
    cmd = ["uv", "run", "python", "-m", "ragicamp.cli.run"]
    
    if multirun:
        cmd.append("--multirun")
    
    cmd.append(f"experiment={experiment}")
    
    for key, value in overrides.items():
        if isinstance(value, list):
            cmd.append(f"{key}={','.join(str(v) for v in value)}")
        else:
            cmd.append(f"{key}={value}")
    
    return run_command(cmd, dry_run)


def run_quick_test(dry_run: bool = False) -> Dict[str, Any]:
    """Run a quick test to verify setup."""
    print("\n" + "="*60)
    print("QUICK TEST: Verifying setup")
    print("="*60)
    
    results = {"passed": 0, "failed": 0, "runs": []}
    
    # Single DirectLLM run
    success = run_hydra_experiment(
        "baseline_study_direct",
        {"evaluation": "quick", "model": "gemma_2b_4bit", "dataset": "nq"},
        dry_run=dry_run,
    )
    results["runs"].append({"type": "direct", "success": success})
    if success:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print("\n" + "-"*60)
    print(f"Quick test complete: {results['passed']} passed, {results['failed']} failed")
    
    return results


def run_direct_llm_study(
    models: List[str] = None,
    datasets: List[str] = None,
    prompts: List[str] = None,
    evaluation: str = "standard",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run DirectLLM baseline study."""
    models = models or MODELS
    datasets = datasets or DATASETS
    prompts = prompts or PROMPTS
    
    print("\n" + "="*60)
    print("PHASE 1: DirectLLM Baselines")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Prompts: {prompts}")
    print(f"  Total runs: {len(models) * len(datasets) * len(prompts)}")
    print("="*60)
    
    results = {"passed": 0, "failed": 0, "runs": []}
    
    # Use multirun for efficiency
    success = run_hydra_experiment(
        "baseline_study_direct",
        {
            "model": models,
            "dataset": datasets,
            "prompt": prompts,
            "evaluation": evaluation,
        },
        dry_run=dry_run,
        multirun=True,
    )
    
    if success:
        results["passed"] = len(models) * len(datasets) * len(prompts)
    else:
        results["failed"] = 1
    
    return results


def run_rag_study(
    models: List[str] = None,
    datasets: List[str] = None,
    top_k_values: List[int] = None,
    evaluation: str = "standard",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run FixedRAG baseline study."""
    models = models or MODELS
    datasets = datasets or DATASETS
    top_k_values = top_k_values or TOP_K_VALUES
    
    print("\n" + "="*60)
    print("PHASE 2: FixedRAG Baselines")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  top_k values: {top_k_values}")
    print(f"  Total runs: {len(models) * len(datasets) * len(top_k_values)}")
    print("="*60)
    
    results = {"passed": 0, "failed": 0, "runs": []}
    
    # Use multirun for efficiency
    success = run_hydra_experiment(
        "baseline_study_rag",
        {
            "model": models,
            "dataset": datasets,
            "agent.top_k": top_k_values,
            "evaluation": evaluation,
        },
        dry_run=dry_run,
        multirun=True,
    )
    
    if success:
        results["passed"] = len(models) * len(datasets) * len(top_k_values)
    else:
        results["failed"] = 1
    
    return results


def check_prerequisites() -> bool:
    """Check that required data exists."""
    issues = []
    
    # Check datasets
    data_dir = Path("data/datasets")
    for dataset in ["natural_questions", "triviaqa", "hotpotqa"]:
        pattern = f"{dataset}*.json"
        if not list(data_dir.glob(pattern)):
            issues.append(f"Dataset not found: {dataset}")
    
    # Check index for RAG
    index_dir = Path("artifacts/retrievers")
    if not index_dir.exists() or not list(index_dir.iterdir()):
        issues.append("No retriever index found (needed for RAG experiments)")
    
    if issues:
        print("\n⚠️  Prerequisites missing:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("  make download-all   # Download datasets")
        print("  make index          # Build retriever index")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the baseline study experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick verification
    python scripts/experiments/run_baseline_study.py --quick
    
    # DirectLLM only (faster)
    python scripts/experiments/run_baseline_study.py --direct-only
    
    # Full study
    python scripts/experiments/run_baseline_study.py --full
    
    # Custom models
    python scripts/experiments/run_baseline_study.py --models gemma_2b_4bit phi3
        """,
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Quick test (verify setup)")
    mode.add_argument("--direct-only", action="store_true", help="Run DirectLLM experiments only")
    mode.add_argument("--rag-only", action="store_true", help="Run RAG experiments only")
    mode.add_argument("--full", action="store_true", help="Run complete study")
    
    # Customization
    parser.add_argument("--models", nargs="+", default=None, help="Models to test")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to test")
    parser.add_argument("--prompts", nargs="+", default=None, help="Prompt styles to test")
    parser.add_argument("--top-k", nargs="+", type=int, default=None, help="top_k values for RAG")
    parser.add_argument("--evaluation", default="standard", help="Evaluation mode")
    
    # Execution control
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--skip-check", action="store_true", help="Skip prerequisite check")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_check and not args.dry_run:
        if not check_prerequisites():
            print("\nRun with --skip-check to ignore these warnings")
            sys.exit(1)
    
    # Track results
    all_results = {}
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("RAGiCamp Baseline Study")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Run experiments based on mode
    if args.quick:
        all_results["quick_test"] = run_quick_test(args.dry_run)
        
    elif args.direct_only:
        all_results["direct_llm"] = run_direct_llm_study(
            models=args.models,
            datasets=args.datasets,
            prompts=args.prompts,
            evaluation=args.evaluation,
            dry_run=args.dry_run,
        )
        
    elif args.rag_only:
        all_results["rag"] = run_rag_study(
            models=args.models,
            datasets=args.datasets,
            top_k_values=args.top_k,
            evaluation=args.evaluation,
            dry_run=args.dry_run,
        )
        
    elif args.full:
        all_results["direct_llm"] = run_direct_llm_study(
            models=args.models,
            datasets=args.datasets,
            prompts=args.prompts,
            evaluation=args.evaluation,
            dry_run=args.dry_run,
        )
        all_results["rag"] = run_rag_study(
            models=args.models,
            datasets=args.datasets,
            top_k_values=args.top_k,
            evaluation=args.evaluation,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
        sys.exit(1)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Duration: {duration}")
    
    total_passed = sum(r.get("passed", 0) for r in all_results.values())
    total_failed = sum(r.get("failed", 0) for r in all_results.values())
    
    print(f"Total runs: {total_passed + total_failed}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    
    if not args.dry_run:
        print("\nResults saved to: outputs/")
        print("Compare with: make compare")
        print("Generate report: make report")
    
    print("="*60)
    
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
