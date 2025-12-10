#!/usr/bin/env python3
"""Compare experiment results.

Usage:
    python scripts/eval/compare.py outputs/run1.json outputs/run2.json
    python scripts/eval/compare.py outputs/
    python scripts/eval/compare.py outputs/ --metric f1 --sort
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_results(path: Path) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    # Extract metrics - handle different formats
    metrics = {}
    
    # Format 1: Direct metrics
    for key in ["exact_match", "f1", "bertscore_f1", "llm_judge_score", "faithfulness"]:
        if key in data:
            metrics[key] = data[key]
    
    # Format 2: Nested under "metrics"
    if "metrics" in data:
        metrics.update(data["metrics"])
    
    # Format 3: Summary file
    if "summary" in data:
        metrics.update(data["summary"])
    
    return {
        "path": str(path),
        "name": path.stem,
        "metrics": metrics,
        "num_examples": data.get("num_examples", data.get("total_examples", "?")),
    }


def compare_results(results: List[Dict[str, Any]], sort_by: str = None):
    """Print comparison table."""
    if not results:
        print("No results to compare")
        return
    
    # Collect all metrics
    all_metrics = set()
    for r in results:
        all_metrics.update(r["metrics"].keys())
    
    # Sort metrics for consistent display
    metrics_order = ["exact_match", "f1", "bertscore_f1", "llm_judge_score", 
                     "faithfulness", "answer_relevancy", "context_precision"]
    sorted_metrics = [m for m in metrics_order if m in all_metrics]
    sorted_metrics.extend([m for m in sorted(all_metrics) if m not in sorted_metrics])
    
    # Sort results if requested
    if sort_by and sort_by in all_metrics:
        results = sorted(
            results,
            key=lambda r: r["metrics"].get(sort_by, 0),
            reverse=True,
        )
    
    # Print header
    print("\n" + "=" * 80)
    print("Experiment Comparison")
    print("=" * 80)
    
    # Print table header
    name_width = max(len(r["name"]) for r in results)
    name_width = max(name_width, 20)
    
    header = f"{'Experiment':<{name_width}} | {'N':>5}"
    for metric in sorted_metrics:
        short_name = metric[:10]
        header += f" | {short_name:>10}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for r in results:
        row = f"{r['name']:<{name_width}} | {str(r['num_examples']):>5}"
        for metric in sorted_metrics:
            value = r["metrics"].get(metric)
            if value is not None:
                if isinstance(value, float):
                    row += f" | {value:>10.4f}"
                else:
                    row += f" | {str(value):>10}"
            else:
                row += f" | {'-':>10}"
        print(row)
    
    print("-" * len(header))
    
    # Print best for each metric
    if len(results) > 1:
        print("\n🏆 Best Results:")
        for metric in sorted_metrics:
            values = [(r["name"], r["metrics"].get(metric)) for r in results if r["metrics"].get(metric) is not None]
            if values:
                best = max(values, key=lambda x: x[1])
                print(f"  {metric}: {best[0]} ({best[1]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval/compare.py outputs/run1.json outputs/run2.json
  python scripts/eval/compare.py outputs/
  python scripts/eval/compare.py outputs/ --sort f1
        """,
    )
    
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Result files or directories to compare",
    )
    parser.add_argument(
        "--sort", "-s",
        default=None,
        help="Sort by metric (e.g., f1, exact_match)",
    )
    parser.add_argument(
        "--pattern", "-p",
        default="*summary*.json",
        help="Pattern for finding files in directories (default: *summary*.json)",
    )
    
    args = parser.parse_args()
    
    # Collect all result files
    result_files = []
    for path in args.paths:
        if path.is_file():
            result_files.append(path)
        elif path.is_dir():
            result_files.extend(path.glob(args.pattern))
    
    if not result_files:
        print("No result files found")
        sys.exit(1)
    
    # Load results
    results = []
    for path in result_files:
        try:
            results.append(load_results(path))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    
    # Compare
    compare_results(results, sort_by=args.sort)
    print()


if __name__ == "__main__":
    main()
