#!/usr/bin/env python3
"""
Compare baseline study results and generate analysis.

Usage:
    python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline
    python scripts/analysis/compare_baseline.py --csv results.csv
    python scripts/analysis/compare_baseline.py --mlflow  # Log to MLflow
    python scripts/analysis/compare_baseline.py --pivot model dataset  # Pivot table

Examples:
    # Basic comparison by model
    python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline

    # Compare by different dimensions
    python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline --group-by prompt

    # Export to CSV
    python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline --csv results.csv

    # Log to MLflow for tracking
    python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline --mlflow

    # Create pivot table: rows=model, cols=dataset
    python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline --pivot model dataset
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.analysis import (
    ResultsLoader,
    compare_results,
    best_by,
    pivot_results,
    summarize_results,
    format_comparison_table,
)


def print_summary(results):
    """Print high-level summary."""
    summary = summarize_results(results)

    print("\n" + "=" * 70)
    print(" EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Total experiments: {summary['count']}")
    print(f"  Models: {', '.join(summary['models'])}")
    print(f"  Datasets: {', '.join(summary['datasets'])}")
    print(f"  Prompts: {', '.join(summary['prompts'])}")
    print(f"  Types: {', '.join(summary['types'])}")
    print(f"  Total duration: {summary['total_duration_hours']:.1f} hours")
    print()
    print(f"  Average F1: {summary['avg_f1']:.4f}")
    print(f"  Average EM: {summary['avg_exact_match']:.4f}")
    print(f"  Average Throughput: {summary['avg_throughput_qps']:.2f} QPS")

    print("\n" + "-" * 70)
    print(" BEST CONFIGURATIONS")
    print("-" * 70)
    for metric in ["f1", "exact_match", "bertscore_f1"]:
        best = summary.get(f"best_{metric}", {})
        print(
            f"  {metric:<15}: {best.get('value', 0):.4f} "
            f"({best.get('model', '?')}, {best.get('dataset', '?')}, {best.get('prompt', '?')})"
        )


def print_comparison(results, group_by, metric):
    """Print comparison table."""
    stats = compare_results(results, group_by=group_by, metric=metric)
    print(format_comparison_table(stats, title=f"Comparison by {group_by.upper()}", metric=metric))


def print_pivot_table(results, rows, cols, metric):
    """Print pivot table."""
    pivot = pivot_results(results, rows=rows, cols=cols, metric=metric)

    # Get all column values
    all_cols = set()
    for row_data in pivot.values():
        all_cols.update(row_data.keys())
    all_cols = sorted(all_cols)

    print("\n" + "=" * 80)
    print(f" PIVOT TABLE: {rows} x {cols} ({metric})")
    print("=" * 80)

    # Header
    header = f"{'':25}"
    for col in all_cols:
        header += f" {col[:12]:>12}"
    print(header)
    print("-" * 80)

    # Rows
    for row_name, row_data in sorted(pivot.items()):
        row = f"{row_name[:24]:25}"
        for col in all_cols:
            val = row_data.get(col, None)
            if val is not None:
                row += f" {val:>12.4f}"
            else:
                row += f" {'—':>12}"
        print(row)

    print("=" * 80)


def print_top_n(results, n, metric):
    """Print top N results."""
    top = best_by(results, metric=metric, n=n)

    print("\n" + "=" * 100)
    print(f" TOP {n} BY {metric.upper()}")
    print("=" * 100)
    print(f"{'Rank':<5} {'Name':<50} {metric:>10} {'Model':>15} {'Dataset':>10}")
    print("-" * 100)

    for i, r in enumerate(top, 1):
        val = getattr(r, metric, 0)
        print(f"{i:<5} {r.name[:49]:<50} {val:>10.4f} {r.model_short[:14]:>15} {r.dataset:>10}")

    print("=" * 100)


def export_csv(results, path):
    """Export results to CSV."""
    fieldnames = [
        "name",
        "type",
        "model",
        "dataset",
        "prompt",
        "quantization",
        "retriever",
        "top_k",
        "f1",
        "exact_match",
        "bertscore_f1",
        "bleurt",
        "duration",
        "throughput_qps",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    print(f"\n✓ Exported {len(results)} results to: {path}")


def log_to_mlflow(results, experiment_name):
    """Log results to MLflow."""
    try:
        from ragicamp.analysis import MLflowTracker
    except ImportError:
        print("❌ MLflow not installed. Run: pip install mlflow")
        return

    tracker = MLflowTracker(experiment_name)
    logged = tracker.backfill_from_results(results, skip_existing=True)
    print(f"\n✓ Logged {logged} experiments to MLflow")
    print(f"  View at: mlflow ui  (then open http://localhost:5000)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare and analyze RAGiCamp experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="outputs/comprehensive_baseline",
        help="Directory with results (default: outputs/comprehensive_baseline)",
    )
    parser.add_argument(
        "--group-by",
        "-g",
        default="model",
        choices=["model", "dataset", "prompt", "retriever", "quantization", "type"],
        help="Dimension to group by",
    )
    parser.add_argument("--metric", "-m", default="f1", help="Metric to compare (default: f1)")
    parser.add_argument("--top", "-n", type=int, default=10, help="Show top N results")
    parser.add_argument(
        "--pivot",
        nargs=2,
        metavar=("ROWS", "COLS"),
        help="Create pivot table with ROWS and COLS dimensions",
    )
    parser.add_argument("--csv", type=str, help="Export to CSV file")
    parser.add_argument("--json", type=str, help="Export to JSON file")
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument(
        "--mlflow-experiment",
        default="ragicamp",
        help="MLflow experiment name (default: ragicamp)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Load results
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        sys.exit(1)

    print(f"🔍 Loading results from: {results_dir}")
    loader = ResultsLoader(results_dir)
    results = loader.load_all()

    if not results:
        print("❌ No results found.")
        sys.exit(1)

    print(f"📊 Loaded {len(results)} experiments")

    # Generate output
    if not args.quiet:
        print_summary(results)
        print_comparison(results, args.group_by, args.metric)

        if args.pivot:
            print_pivot_table(results, args.pivot[0], args.pivot[1], args.metric)

        print_top_n(results, args.top, args.metric)

    # Exports
    if args.csv:
        export_csv(results, args.csv)

    if args.json:
        data = [r.to_dict() for r in results]
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Exported to JSON: {args.json}")

    if args.mlflow:
        log_to_mlflow(results, args.mlflow_experiment)


if __name__ == "__main__":
    main()
