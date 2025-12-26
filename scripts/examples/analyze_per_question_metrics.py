#!/usr/bin/env python3
"""Example: Analyzing per-question metrics for model evaluation.

This script demonstrates how to load and analyze per-question metrics
to gain insights into model performance.

Usage:
    python examples/analyze_per_question_metrics.py
"""

import json
from pathlib import Path

import pandas as pd


def analyze_metrics(json_path: str):
    """Analyze per-question metrics from a JSON file.

    Args:
        json_path: Path to *_per_question.json file
    """
    print("=" * 70)
    print("PER-QUESTION METRICS ANALYSIS")
    print("=" * 70)

    # Load data
    with open(json_path) as f:
        data = json.load(f)

    metrics_list = data["per_question_metrics"]
    df = pd.DataFrame(metrics_list)

    print(f"\nDataset: {data['dataset_name']}")
    print(f"Agent: {data['agent_name']}")
    print(f"Total questions: {data['num_examples']}")

    # Overall statistics
    print("\n" + "-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)

    metric_cols = [col for col in df.columns if col not in ["question_index", "question"]]

    for metric in metric_cols:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            print(f"\n{metric}:")
            print(f"  Mean:   {df[metric].mean():.4f}")
            print(f"  Median: {df[metric].median():.4f}")
            print(f"  Std:    {df[metric].std():.4f}")
            print(f"  Min:    {df[metric].min():.4f}")
            print(f"  Max:    {df[metric].max():.4f}")

    # Find hardest questions
    print("\n" + "-" * 70)
    print("HARDEST QUESTIONS (by F1 score)")
    print("-" * 70)

    if "f1" in df.columns:
        hardest = df.nsmallest(5, "f1")
        for idx, row in hardest.iterrows():
            print(f"\n{row['question_index']}. {row['question']}")
            for metric in metric_cols:
                if metric in row:
                    print(f"   {metric}: {row[metric]:.4f}")

    # Find easiest questions
    print("\n" + "-" * 70)
    print("EASIEST QUESTIONS (by F1 score)")
    print("-" * 70)

    if "f1" in df.columns:
        easiest = df.nlargest(5, "f1")
        for idx, row in easiest.iterrows():
            print(f"\n{row['question_index']}. {row['question']}")
            for metric in metric_cols:
                if metric in row:
                    print(f"   {metric}: {row[metric]:.4f}")

    # Performance distribution
    print("\n" + "-" * 70)
    print("PERFORMANCE DISTRIBUTION")
    print("-" * 70)

    if "exact_match" in df.columns:
        perfect = (df["exact_match"] == 1.0).sum()
        wrong = (df["exact_match"] == 0.0).sum()
        partial = len(df) - perfect - wrong

        print(f"\nExact Match Distribution:")
        print(f"  Perfect (EM = 1.0):  {perfect:3d} ({perfect/len(df)*100:.1f}%)")
        print(f"  Partial (0 < EM < 1): {partial:3d} ({partial/len(df)*100:.1f}%)")
        print(f"  Wrong (EM = 0.0):     {wrong:3d} ({wrong/len(df)*100:.1f}%)")

    # Metric correlations
    if len(metric_cols) > 1:
        print("\n" + "-" * 70)
        print("METRIC CORRELATIONS")
        print("-" * 70)

        numeric_cols = [col for col in metric_cols if pd.api.types.is_numeric_dtype(df[col])]

        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            print("\n", corr.to_string())

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def compare_models(json_paths: list):
    """Compare per-question performance across models.

    Args:
        json_paths: List of paths to *_per_question.json files
    """
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    models = {}
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        model_name = data["agent_name"]
        models[model_name] = pd.DataFrame(data["per_question_metrics"])

    # Compare overall performance
    print("\nOverall Performance:")
    print("-" * 70)

    for model_name, df in models.items():
        print(f"\n{model_name}:")
        for metric in ["exact_match", "f1", "bertscore_f1"]:
            if metric in df.columns:
                print(f"  {metric:20s}: {df[metric].mean():.4f}")

    # Find questions where models differ most
    print("\n" + "-" * 70)
    print("QUESTIONS WITH LARGEST PERFORMANCE DIFFERENCES")
    print("-" * 70)

    if len(models) == 2:
        model_names = list(models.keys())
        df1, df2 = models[model_names[0]], models[model_names[1]]

        if "f1" in df1.columns and "f1" in df2.columns:
            # Merge on question index
            merged = df1.merge(df2, on="question_index", suffixes=("_1", "_2"))
            merged["f1_diff"] = abs(merged["f1_1"] - merged["f1_2"])

            biggest_diff = merged.nlargest(5, "f1_diff")

            for idx, row in biggest_diff.iterrows():
                print(f"\n{row['question_index']}. {row['question_1']}")
                print(f"  {model_names[0]:20s}: F1 = {row['f1_1']:.4f}")
                print(f"  {model_names[1]:20s}: F1 = {row['f1_2']:.4f}")
                print(f"  Difference: {row['f1_diff']:.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_per_question_metrics.py <json_file> [json_file2 ...]")
        print("\nExample:")
        print(
            "  python analyze_per_question_metrics.py outputs/gemma2b_baseline_results_per_question.json"
        )
        print("\nOr compare multiple models:")
        print(
            "  python analyze_per_question_metrics.py model1_per_question.json model2_per_question.json"
        )
        sys.exit(1)

    json_files = sys.argv[1:]

    # Analyze first file
    analyze_metrics(json_files[0])

    # If multiple files, compare them
    if len(json_files) > 1:
        print("\n\n")
        compare_models(json_files)
