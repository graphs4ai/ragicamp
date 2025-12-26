#!/usr/bin/env python3
"""Compare experiment results in a formatted table.

Usage:
    python scripts/experiments/compare_results.py outputs/simple
    python scripts/experiments/compare_results.py outputs/simple --sort llm_judge
    python scripts/experiments/compare_results.py outputs/simple --filter direct
"""

import argparse
import json
import sys
from pathlib import Path


def load_comparison(path: Path) -> dict:
    """Load comparison.json from experiment directory."""
    comparison_path = path / "comparison.json"
    if not comparison_path.exists():
        print(f"❌ No comparison.json found at {path}")
        print("   Run experiments first or check the path")
        sys.exit(1)
    
    with open(comparison_path) as f:
        return json.load(f)


def print_table(experiments: list, sort_by: str = "llm_judge_qa"):
    """Print formatted comparison table."""
    if not experiments:
        print("No experiments found!")
        return
    
    # Sort
    experiments.sort(
        key=lambda x: x["results"].get(sort_by, 0), 
        reverse=True
    )
    
    # Header
    print()
    print("{:<52} {:>7} {:>5} {:>7} {:>7} {:>7}".format(
        "Experiment", "F1", "EM", "BERT", "BLEURT", "Judge"
    ))
    print("-" * 92)
    
    # Rows
    for exp in experiments:
        name = exp["name"][:50]
        r = exp["results"]
        f1 = r.get("f1", 0) * 100
        em = r.get("exact_match", 0) * 100
        bert = r.get("bertscore_f1", 0) * 100
        bleurt = r.get("bleurt", 0) * 100
        judge = r.get("llm_judge_qa", 0) * 100
        
        print("{:<52} {:>6.1f}% {:>4.0f}% {:>6.1f}% {:>6.1f}% {:>6.0f}%".format(
            name, f1, em, bert, bleurt, judge
        ))
    
    # Summary
    print("-" * 92)
    print()
    print("Summary ({} experiments):".format(len(experiments)))
    print("  Best F1:      {:.1f}%".format(max(e["results"].get("f1", 0) for e in experiments) * 100))
    print("  Best EM:      {:.1f}%".format(max(e["results"].get("exact_match", 0) for e in experiments) * 100))
    print("  Best BERT:    {:.1f}%".format(max(e["results"].get("bertscore_f1", 0) for e in experiments) * 100))
    print("  Best BLEURT:  {:.1f}%".format(max(e["results"].get("bleurt", 0) for e in experiments) * 100))
    print("  Best Judge:   {:.0f}%".format(max(e["results"].get("llm_judge_qa", 0) for e in experiments) * 100))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results in a table"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to experiment output directory (containing comparison.json)"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="llm_judge_qa",
        choices=["f1", "exact_match", "bertscore_f1", "bleurt", "llm_judge_qa"],
        help="Metric to sort by (default: llm_judge_qa)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter experiments by type (e.g., 'direct', 'rag', 'openai', 'gemma')"
    )
    
    args = parser.parse_args()
    
    # Load data
    data = load_comparison(args.path)
    experiments = data.get("experiments", [])
    
    # Filter if requested
    if args.filter:
        experiments = [
            e for e in experiments 
            if args.filter.lower() in e["name"].lower()
        ]
        print(f"Filtered to {len(experiments)} experiments matching '{args.filter}'")
    
    # Print table
    print_table(experiments, sort_by=args.sort)


if __name__ == "__main__":
    main()
