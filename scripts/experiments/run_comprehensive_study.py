#!/usr/bin/env python3
"""Run comprehensive baseline study with structured results.

This script runs a full baseline comparison with:
- DirectLLM: No retrieval, vary models and prompts
- FixedRAG: Vary models, retrievers, top_k, prompts

Results are saved to:
- MLflow: For interactive analysis
- JSON: Structured results for programmatic analysis

Usage:
    # Quick test (verify setup)
    python scripts/experiments/run_comprehensive_study.py --quick
    
    # DirectLLM only (5-10 variations)
    python scripts/experiments/run_comprehensive_study.py --direct-only
    
    # RAG only (10-20 variations)
    python scripts/experiments/run_comprehensive_study.py --rag-only
    
    # Full study
    python scripts/experiments/run_comprehensive_study.py --full
    
    # Dry run (show commands)
    python scripts/experiments/run_comprehensive_study.py --full --dry-run
    
    # Remove 100-example cap for production run
    python scripts/experiments/run_comprehensive_study.py --full --no-limit
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

# DirectLLM Study Configurations
DIRECT_MODELS = [
    "gemma_2b_4bit",
    "llama3_8b",
]

DIRECT_PROMPTS = [
    "concise",
    "sentence",
    "explained",
]

# RAG Study Configurations
RAG_MODELS = [
    "gemma_2b_4bit",
    "llama3_8b",
]

# Retrievers: corpus × embedding × chunk_size combinations
# These must be pre-built with: make index-standard or make index-extended
# Format: {corpus}_{embedding}_{chunk_size}
RAG_RETRIEVERS = [
    "simple_minilm_512",   # Simple Wiki + MiniLM (fast)
    "simple_mpnet_512",    # Simple Wiki + MPNet (quality)
]

RAG_TOP_K = [3, 5, 10]

RAG_PROMPTS = [
    "rag_concise",
    "rag_extractive",
]

# Datasets (same for both)
DATASETS = ["nq", "triviaqa", "hotpotqa"]

# Time between runs for GPU cleanup (seconds)
CLEANUP_DELAY = 5

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_command(cmd: List[str], dry_run: bool = False, timeout: int = 3600) -> Tuple[bool, str]:
    """Run command in subprocess, return (success, output_dir)."""
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")
    
    if dry_run:
        return True, ""
    
    try:
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        result = subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Try to extract output directory from hydra output
        output_dir = ""
        for line in result.stdout.split("\n"):
            if "outputs/" in line and "/" in line:
                # Extract path-like string
                parts = line.split()
                for part in parts:
                    if "outputs/" in part:
                        output_dir = part.strip("'\"")
                        break
        
        return True, output_dir
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e.stderr[:500] if e.stderr else 'No stderr'}")
        return False, ""
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout}s")
        return False, ""


def generate_direct_configs(
    models: List[str],
    datasets: List[str],
    prompts: List[str],
    num_examples: int = 100,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Generate DirectLLM experiment configurations."""
    for model, dataset, prompt in itertools.product(models, datasets, prompts):
        run_id = f"direct_{model}_{dataset}_{prompt}"
        config = {
            "model": model,
            "dataset": dataset,
            "prompt": prompt,
            "dataset.num_examples": num_examples,
        }
        yield run_id, config


def generate_rag_configs(
    models: List[str],
    datasets: List[str],
    retrievers: List[str],
    top_k_values: List[int],
    prompts: List[str],
    num_examples: int = 100,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Generate FixedRAG experiment configurations."""
    for model, dataset, retriever, top_k, prompt in itertools.product(
        models, datasets, retrievers, top_k_values, prompts
    ):
        run_id = f"rag_{model}_{dataset}_{retriever}_k{top_k}_{prompt}"
        config = {
            "model": model,
            "dataset": dataset,
            "retriever": retriever,
            "agent.top_k": top_k,
            "prompt": prompt,
            "dataset.num_examples": num_examples,
        }
        yield run_id, config


def run_experiment(
    experiment: str,
    run_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return result."""
    cmd = ["uv", "run", "python", "-m", "ragicamp.cli.run"]
    cmd.append(f"experiment={experiment}")
    
    for key, value in config.items():
        cmd.append(f"{key}={value}")
    
    start_time = datetime.now()
    success, output_dir = run_command(cmd, dry_run)
    end_time = datetime.now()
    
    result = {
        "run_id": run_id,
        "experiment": experiment,
        "config": config,
        "success": success,
        "output_dir": output_dir,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
    }
    
    # Try to load metrics from output
    if success and output_dir:
        results_path = Path(output_dir) / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                result["metrics"] = json.load(f)
    
    return result


# ============================================================================
# STUDY RUNNERS
# ============================================================================

def run_direct_study(
    models: List[str] = None,
    datasets: List[str] = None,
    prompts: List[str] = None,
    num_examples: int = 100,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run DirectLLM baseline study."""
    models = models or DIRECT_MODELS
    datasets = datasets or DATASETS
    prompts = prompts or DIRECT_PROMPTS
    
    configs = list(generate_direct_configs(models, datasets, prompts, num_examples))
    total = len(configs)
    
    print("\n" + "=" * 70)
    print("📊 DIRECTLLM STUDY")
    print("=" * 70)
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Prompts: {prompts}")
    print(f"  Examples per run: {num_examples}")
    print(f"  Total runs: {total}")
    print("=" * 70)
    
    results = {"runs": [], "passed": 0, "failed": 0}
    
    for i, (run_id, config) in enumerate(configs, 1):
        print(f"\n[{i}/{total}] {run_id}")
        
        result = run_experiment("comprehensive_direct", run_id, config, dry_run)
        results["runs"].append(result)
        
        if result["success"]:
            results["passed"] += 1
            print(f"  ✅ PASSED")
            if "metrics" in result:
                metrics = result["metrics"]
                print(f"     F1: {metrics.get('f1', 'N/A'):.3f}")
                print(f"     EM: {metrics.get('exact_match', 'N/A'):.3f}")
                print(f"     LLM Judge: {metrics.get('llm_judge_qa', 'N/A'):.3f}")
        else:
            results["failed"] += 1
            print(f"  ❌ FAILED")
        
        # GPU cleanup between runs
        if not dry_run and i < total:
            time.sleep(CLEANUP_DELAY)
    
    return results


def run_rag_study(
    models: List[str] = None,
    datasets: List[str] = None,
    retrievers: List[str] = None,
    top_k_values: List[int] = None,
    prompts: List[str] = None,
    num_examples: int = 100,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run FixedRAG baseline study."""
    models = models or RAG_MODELS
    datasets = datasets or DATASETS
    retrievers = retrievers or RAG_RETRIEVERS
    top_k_values = top_k_values or RAG_TOP_K
    prompts = prompts or RAG_PROMPTS
    
    configs = list(generate_rag_configs(
        models, datasets, retrievers, top_k_values, prompts, num_examples
    ))
    total = len(configs)
    
    print("\n" + "=" * 70)
    print("📊 FIXEDRAG STUDY")
    print("=" * 70)
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Retrievers: {retrievers}")
    print(f"  Top-K values: {top_k_values}")
    print(f"  Prompts: {prompts}")
    print(f"  Examples per run: {num_examples}")
    print(f"  Total runs: {total}")
    print("=" * 70)
    
    results = {"runs": [], "passed": 0, "failed": 0}
    
    for i, (run_id, config) in enumerate(configs, 1):
        print(f"\n[{i}/{total}] {run_id}")
        
        result = run_experiment("comprehensive_rag", run_id, config, dry_run)
        results["runs"].append(result)
        
        if result["success"]:
            results["passed"] += 1
            print(f"  ✅ PASSED")
            if "metrics" in result:
                metrics = result["metrics"]
                print(f"     F1: {metrics.get('f1', 'N/A'):.3f}")
                print(f"     EM: {metrics.get('exact_match', 'N/A'):.3f}")
                print(f"     LLM Judge: {metrics.get('llm_judge_qa', 'N/A'):.3f}")
        else:
            results["failed"] += 1
            print(f"  ❌ FAILED")
        
        # GPU cleanup between runs
        if not dry_run and i < total:
            time.sleep(CLEANUP_DELAY)
    
    return results


def run_quick_test(dry_run: bool = False) -> Dict[str, Any]:
    """Quick verification test."""
    print("\n" + "=" * 70)
    print("🧪 QUICK TEST")
    print("=" * 70)
    
    # Single DirectLLM run
    result = run_experiment(
        "comprehensive_direct",
        "quick_test_direct",
        {"model": "gemma_2b_4bit", "dataset": "nq", "prompt": "concise", "dataset.num_examples": 10},
        dry_run,
    )
    
    return {
        "runs": [result],
        "passed": 1 if result["success"] else 0,
        "failed": 0 if result["success"] else 1,
    }


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def save_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save structured results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_study_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {filepath}")
    return filepath


def generate_summary_table(results: Dict[str, Any]) -> str:
    """Generate a markdown summary table."""
    lines = [
        "# Comprehensive Study Results",
        "",
        f"**Date**: {results.get('end_time', 'N/A')}",
        f"**Duration**: {results.get('duration_seconds', 0) / 60:.1f} minutes",
        f"**Total Runs**: {results.get('total_passed', 0) + results.get('total_failed', 0)}",
        f"**Passed**: {results.get('total_passed', 0)}",
        f"**Failed**: {results.get('total_failed', 0)}",
        "",
    ]
    
    # DirectLLM results table
    if "direct_llm" in results:
        lines.append("## DirectLLM Results")
        lines.append("")
        lines.append("| Model | Dataset | Prompt | F1 | EM | LLM Judge |")
        lines.append("|-------|---------|--------|----|----|-----------|")
        
        for run in results["direct_llm"].get("runs", []):
            if run["success"] and "metrics" in run:
                m = run["metrics"]
                cfg = run["config"]
                lines.append(
                    f"| {cfg['model']} | {cfg['dataset']} | {cfg['prompt']} | "
                    f"{m.get('f1', 0):.3f} | {m.get('exact_match', 0):.3f} | "
                    f"{m.get('llm_judge_qa', 0):.3f} |"
                )
        lines.append("")
    
    # RAG results table
    if "rag" in results:
        lines.append("## FixedRAG Results")
        lines.append("")
        lines.append("| Model | Dataset | Retriever | Top-K | F1 | EM | LLM Judge |")
        lines.append("|-------|---------|-----------|-------|----|----|-----------|")
        
        for run in results["rag"].get("runs", []):
            if run["success"] and "metrics" in run:
                m = run["metrics"]
                cfg = run["config"]
                lines.append(
                    f"| {cfg['model']} | {cfg['dataset']} | {cfg['retriever']} | "
                    f"{cfg['agent.top_k']} | {m.get('f1', 0):.3f} | "
                    f"{m.get('exact_match', 0):.3f} | {m.get('llm_judge_qa', 0):.3f} |"
                )
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive baseline study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Quick test (verify setup)")
    mode.add_argument("--direct-only", action="store_true", help="DirectLLM only")
    mode.add_argument("--rag-only", action="store_true", help="RAG only")
    mode.add_argument("--full", action="store_true", help="Full study (Direct + RAG)")
    
    # Options
    parser.add_argument("--dry-run", action="store_true", help="Show commands only")
    parser.add_argument("--no-limit", action="store_true", help="Remove 100-example cap")
    parser.add_argument("--num-examples", type=int, default=100, help="Examples per run")
    parser.add_argument("--output-dir", default="outputs/studies", help="Output directory")
    
    args = parser.parse_args()
    
    num_examples = None if args.no_limit else args.num_examples
    output_dir = Path(args.output_dir)
    
    start_time = datetime.now()
    all_results = {
        "start_time": start_time.isoformat(),
        "num_examples": num_examples,
        "args": vars(args),
    }
    
    print("\n" + "=" * 70)
    print("🚀 COMPREHENSIVE BASELINE STUDY")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Examples per run: {num_examples or 'FULL DATASET'}")
    print("=" * 70)
    
    # Run studies
    if args.quick:
        all_results["quick_test"] = run_quick_test(args.dry_run)
    elif args.direct_only:
        all_results["direct_llm"] = run_direct_study(
            num_examples=num_examples, dry_run=args.dry_run
        )
    elif args.rag_only:
        all_results["rag"] = run_rag_study(
            num_examples=num_examples, dry_run=args.dry_run
        )
    elif args.full:
        all_results["direct_llm"] = run_direct_study(
            num_examples=num_examples, dry_run=args.dry_run
        )
        all_results["rag"] = run_rag_study(
            num_examples=num_examples, dry_run=args.dry_run
        )
    else:
        parser.print_help()
        sys.exit(1)
    
    # Finalize
    end_time = datetime.now()
    all_results["end_time"] = end_time.isoformat()
    all_results["duration_seconds"] = (end_time - start_time).total_seconds()
    
    # Aggregate counts
    total_passed = sum(r.get("passed", 0) for r in all_results.values() if isinstance(r, dict))
    total_failed = sum(r.get("failed", 0) for r in all_results.values() if isinstance(r, dict))
    all_results["total_passed"] = total_passed
    all_results["total_failed"] = total_failed
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Duration: {(end_time - start_time)}")
    print(f"Total runs: {total_passed + total_failed}")
    print(f"  ✅ Passed: {total_passed}")
    print(f"  ❌ Failed: {total_failed}")
    
    # Save results
    if not args.dry_run:
        save_results(all_results, output_dir)
        
        # Generate summary markdown
        summary_md = generate_summary_table(all_results)
        summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(summary_path, "w") as f:
            f.write(summary_md)
        print(f"📄 Summary: {summary_path}")
    
    print("\n💡 Next steps:")
    print("   make mlflow-ui    # View results in MLflow")
    print("   make compare      # Compare all runs")
    print("=" * 70)
    
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
