#!/usr/bin/env python3
"""
Compare baseline study results and generate visualizations.

Usage:
    python scripts/analysis/compare_baseline.py outputs/multirun/2025-12-10/
    python scripts/analysis/compare_baseline.py --output report.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def find_summary_files(base_dir: Path) -> List[Path]:
    """Find all summary JSON files in the output directory."""
    summaries = []
    for path in base_dir.rglob("*_summary.json"):
        summaries.append(path)
    return sorted(summaries)


def load_results(summary_files: List[Path]) -> List[Dict[str, Any]]:
    """Load and parse all summary files."""
    results = []
    for path in summary_files:
        try:
            with open(path) as f:
                data = json.load(f)

            # Extract experiment info from Hydra config if available
            hydra_dir = path.parent / ".hydra"
            config = {}
            if hydra_dir.exists():
                config_file = hydra_dir / "config.yaml"
                if config_file.exists():
                    import yaml

                    with open(config_file) as f:
                        config = yaml.safe_load(f)

            results.append(
                {
                    "path": str(path),
                    "run_dir": str(path.parent),
                    "model": config.get("model", {}).get("model_name", "unknown"),
                    "dataset": config.get("dataset", {}).get("name", "unknown"),
                    "prompt": config.get("prompt", {}).get("style", "unknown"),
                    "agent": config.get("agent", {}).get("type", "unknown"),
                    **data,
                }
            )
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")

    return results


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return

    # Determine available metrics
    metric_keys = set()
    for r in results:
        for key in r.keys():
            if key in ["exact_match", "f1", "llm_judge_qa", "bertscore_f1", "bleurt"]:
                metric_keys.add(key)

    metric_keys = sorted(metric_keys)

    # Header
    print("\n" + "=" * 100)
    print("BASELINE STUDY RESULTS")
    print("=" * 100)

    # Table header
    header = f"{'Model':<30} {'Dataset':<15} {'Prompt':<12}"
    for m in metric_keys:
        header += f" {m:<12}"
    print(header)
    print("-" * 100)

    # Sort results
    results_sorted = sorted(
        results, key=lambda x: (x.get("model", ""), x.get("dataset", ""), x.get("prompt", ""))
    )

    # Table rows
    for r in results_sorted:
        model_short = r.get("model", "unknown").split("/")[-1][:28]
        row = f"{model_short:<30} {r.get('dataset', '?'):<15} {r.get('prompt', '?'):<12}"
        for m in metric_keys:
            val = r.get(m, None)
            if val is not None:
                row += f" {val:.4f}      "
            else:
                row += f" {'N/A':<12}"
        print(row)

    print("=" * 100)


def generate_summary_stats(results: List[Dict[str, Any]]):
    """Print summary statistics."""
    if not results:
        return

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Group by model
    by_model = {}
    for r in results:
        model = r.get("model", "unknown").split("/")[-1]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    for model, runs in by_model.items():
        print(f"\n📊 {model}")

        # Compute averages for key metrics
        for metric in ["exact_match", "f1", "llm_judge_qa"]:
            values = [r.get(metric) for r in runs if r.get(metric) is not None]
            if values:
                avg = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                print(f"   {metric}: avg={avg:.4f}, min={min_val:.4f}, max={max_val:.4f}")

    # Best configurations
    print("\n🏆 BEST CONFIGURATIONS")

    for metric in ["exact_match", "f1", "llm_judge_qa"]:
        values = [(r, r.get(metric)) for r in results if r.get(metric) is not None]
        if values:
            best = max(values, key=lambda x: x[1])
            r, val = best
            print(
                f"   {metric}: {val:.4f} ({r.get('model', '?').split('/')[-1]}, {r.get('dataset')}, {r.get('prompt')})"
            )


def save_csv(results: List[Dict[str, Any]], output_path: Path):
    """Save results to CSV."""
    import csv

    if not results:
        return

    # Collect all keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    # Remove path-like keys for cleaner CSV
    skip_keys = {"path", "run_dir", "per_question_scores"}
    keys = sorted([k for k in all_keys if k not in skip_keys])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            # Shorten model names
            if "model" in r:
                r = dict(r)
                r["model"] = r["model"].split("/")[-1]
            writer.writerow(r)

    print(f"\n✓ CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline study results")
    parser.add_argument("results_dir", nargs="?", default="outputs", help="Directory with results")
    parser.add_argument("--csv", type=str, help="Save results to CSV")
    parser.add_argument("--json", type=str, help="Save results to JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        sys.exit(1)

    print(f"🔍 Searching for results in: {results_dir}")

    summary_files = find_summary_files(results_dir)
    print(f"📁 Found {len(summary_files)} result files")

    if not summary_files:
        print("No summary files found. Run experiments first.")
        sys.exit(1)

    results = load_results(summary_files)
    print(f"📊 Loaded {len(results)} results")

    # Display results
    print_comparison_table(results)
    generate_summary_stats(results)

    # Save outputs
    if args.csv:
        save_csv(results, Path(args.csv))

    if args.json:
        output_path = Path(args.json)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
