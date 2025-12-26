#!/usr/bin/env python3
"""Compare experiment results.

Usage:
    python scripts/eval/compare.py outputs/
    python scripts/eval/compare.py outputs/ --sort f1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_hydra_config(result_dir: Path) -> Dict[str, Any]:
    """Load Hydra config from .hydra/config.yaml if available."""
    config_path = result_dir / ".hydra" / "config.yaml"
    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception:
            pass
    return {}


def load_results(path: Path) -> Optional[List[Dict[str, Any]]]:
    """Load results from a JSON file. Returns list of result dicts."""
    with open(path) as f:
        data = json.load(f)

    # Handle comparison.json format (aggregated from run_study.py)
    if "experiments" in data:
        results = []
        for exp in data["experiments"]:
            metrics = exp.get("results", {})
            results.append({
                "path": str(path),
                "name": exp.get("name", "?"),
                "metrics": metrics,
                "num_examples": exp.get("num_questions", "?"),
                "dataset": exp.get("dataset", "?"),
                "model": exp.get("model", "?").replace("openai:", "").replace("hf:", ""),
                "prompt": exp.get("prompt", "?"),
                "agent": exp.get("type", "?"),
                "timestamp": exp.get("timestamp", "?"),
            })
        return results

    # Skip orchestration logs
    if "total_passed" in data and "results" in data:
        return None

    # Extract metrics from results.json or legacy summary files
    metrics = {}
    if "overall_metrics" in data:
        for key, value in data["overall_metrics"].items():
            if isinstance(value, (int, float)) and key not in ["num_successful", "num_failures"]:
                metrics[key] = value

    for key in ["exact_match", "f1", "bertscore_f1", "bertscore_precision", "bertscore_recall", "bleurt", "llm_judge_qa"]:
        if key in data and key not in metrics:
            metrics[key] = data[key]

    if "metrics" in data and isinstance(data["metrics"], dict):
        metrics.update(data["metrics"])

    if not metrics:
        return None

    # Load Hydra config for full experiment info
    hydra_config = load_hydra_config(path.parent)

    # Extract experiment parameters
    model_name = "?"
    prompt_style = "?"
    dataset = "?"

    if hydra_config:
        model_cfg = hydra_config.get("model", {})
        model_name = model_cfg.get("model_name", "?")
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        if len(model_name) > 20:
            model_name = model_name[:17] + "..."

        prompt_cfg = hydra_config.get("prompt", {})
        prompt_style = prompt_cfg.get("style", "?")
    else:
        # Try to parse from metadata.json (new format)
        metadata_path = path.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            model_name = metadata.get("model", "?").replace("openai:", "").replace("hf:", "")
            prompt_style = metadata.get("prompt", "?")
            dataset = metadata.get("dataset", "?")
        else:
            # Fallback: try to parse from directory name or agent_name
            dir_name = path.parent.name
            agent_name = data.get("agent_name", dir_name)
            
            if "gemma" in agent_name.lower():
                model_name = "gemma-2b-it"
            elif "llama" in agent_name.lower():
                model_name = "llama3"
            elif "phi" in agent_name.lower():
                model_name = "phi3"
            elif "gpt4o" in agent_name.lower() or "gpt-4o" in agent_name.lower():
                model_name = "gpt-4o-mini"
            
            if "rag" in agent_name.lower():
                prompt_style = "rag"
            elif "default" in agent_name.lower():
                prompt_style = "default"
            elif "concise" in agent_name.lower():
                prompt_style = "concise"
            elif "detailed" in agent_name.lower():
                prompt_style = "detailed"
            
            # Parse dataset from dir name
            for ds in ["nq", "triviaqa", "hotpotqa"]:
                if ds in agent_name.lower():
                    dataset = ds
                    break

    if dataset == "?":
        dataset = data.get("dataset_name", "?")

    name = f"{dataset[:8]}/{model_name[:12]}/{prompt_style}"

    return [{
        "path": str(path),
        "name": name,
        "metrics": metrics,
        "num_examples": data.get("num_examples", "?"),
        "dataset": dataset,
        "model": model_name,
        "prompt": prompt_style,
        "agent": data.get("agent_name", "?"),
        "timestamp": path.parent.name,
    }]


def find_result_files(base_path: Path, pattern: str = "*summary*.json") -> List[Path]:
    """Find result files recursively."""
    if base_path.is_file():
        return [base_path]

    files = []
    
    # First check for comparison.json (preferred - aggregated results)
    comparison_file = base_path / "comparison.json"
    if comparison_file.exists():
        return [comparison_file]
    
    # Then look for results.json files (new format from run_study.py)
    for f in base_path.rglob("results.json"):
        files.append(f)
    
    # Also look for legacy summary files
    for f in base_path.rglob(pattern):
        if "baseline_study_summary" in f.name or "rag_baseline_study" in f.name:
            continue
        if f not in files:
            files.append(f)

    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def compare_results(results: List[Dict[str, Any]], sort_by: str = None, group_by: str = None):
    """Print comparison table."""
    if not results:
        print("No results found with metrics")
        return

    # Collect all metrics
    all_metrics = set()
    for r in results:
        all_metrics.update(r["metrics"].keys())

    # Priority order
    priority = ["exact_match", "f1", "bertscore_f1", "bleurt", "llm_judge_qa"]
    display_metrics = [m for m in priority if m in all_metrics][:5]

    # Sort results
    if sort_by and sort_by in all_metrics:
        results = sorted(results, key=lambda r: r["metrics"].get(sort_by, 0), reverse=True)

    # Print header
    print("\n" + "=" * 100)
    print("Experiment Comparison")
    print("=" * 100)

    # Column headers
    header = f"{'Dataset':<12} | {'Model':<15} | {'Prompt':<10} | {'N':>4}"
    for metric in display_metrics:
        short = metric.replace("bertscore_", "bs_").replace("llm_judge_", "llm_")[:8]
        header += f" | {short:>8}"
    print(header)
    print("-" * len(header))

    # Print rows
    for r in results:
        dataset = r["dataset"][:12] if r["dataset"] else "?"
        model = r["model"][:15] if r["model"] else "?"
        prompt = r["prompt"][:10] if r["prompt"] else "?"

        row = f"{dataset:<12} | {model:<15} | {prompt:<10} | {str(r['num_examples']):>4}"
        for metric in display_metrics:
            value = r["metrics"].get(metric)
            if value is not None and isinstance(value, float):
                row += f" | {value:>8.3f}"
            elif value is not None:
                row += f" | {str(value):>8}"
            else:
                row += f" | {'-':>8}"
        print(row)

    print("-" * len(header))

    # Best results per dataset
    if len(results) > 1:
        print("\n🏆 Best per dataset:")
        datasets = set(r["dataset"] for r in results if r["dataset"] != "?")
        for dataset in sorted(datasets):
            dataset_results = [r for r in results if r["dataset"] == dataset]
            if dataset_results:
                # Best by F1
                best = max(dataset_results, key=lambda r: r["metrics"].get("f1", 0))
                f1 = best["metrics"].get("f1", 0)
                print(f"  {dataset}: {best['model'][:15]}/{best['prompt']} (F1={f1:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("paths", nargs="+", type=Path, help="Result files or directories")
    parser.add_argument("--sort", "-s", default="f1", help="Sort by metric (default: f1)")
    parser.add_argument("--pattern", "-p", default="*summary*.json", help="File pattern")
    parser.add_argument("--all", "-a", action="store_true", help="Show all files found")

    args = parser.parse_args()

    result_files = []
    for path in args.paths:
        result_files.extend(find_result_files(path, args.pattern))

    if not result_files:
        print("No result files found")
        sys.exit(1)

    if args.all:
        print(f"Found {len(result_files)} files")

    results = []
    for path in result_files:
        try:
            r = load_results(path)
            if r:
                # load_results now returns a list
                results.extend(r)
        except Exception as e:
            if args.all:
                print(f"Warning: {path}: {e}")

    compare_results(results, sort_by=args.sort)
    print()


if __name__ == "__main__":
    main()
