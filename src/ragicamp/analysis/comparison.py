"""Compare and analyze experiment results.

Provides utilities for:
- Grouping results by different dimensions (model, dataset, prompt, etc.)
- Finding best configurations
- Creating pivot tables
- Generating summary statistics
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

from ragicamp.analysis.loader import ExperimentResult


def compare_results(
    results: List[ExperimentResult],
    group_by: str = "model",
    metric: str = "f1",
    sort: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compare results grouped by a dimension.

    Args:
        results: List of experiment results
        group_by: Dimension to group by (model, dataset, prompt, retriever, quantization, type)
        metric: Metric to compare (f1, exact_match, bertscore_f1, bleurt)
        sort: Sort by metric value descending

    Returns:
        Dict mapping group values to stats: {group: {mean, min, max, count, values}}

    Example:
        >>> compare_results(results, group_by="model", metric="f1")
        {
            "gemma-2b-it": {"mean": 0.15, "min": 0.07, "max": 0.21, "count": 24},
            "Llama-3.2-3B": {"mean": 0.18, "min": 0.10, "max": 0.25, "count": 24},
        }
    """
    groups: Dict[str, List[float]] = defaultdict(list)

    for r in results:
        # Get group key
        if group_by == "model":
            key = r.model_short
        elif group_by == "retriever":
            key = r.retriever_short
        elif group_by == "model_full":
            key = r.model
        else:
            key = str(getattr(r, group_by, "unknown"))

        # Get metric value
        value = getattr(r, metric, None)
        if value is not None:
            groups[key].append(value)

    # Compute statistics
    stats = {}
    for key, values in groups.items():
        if values:
            stats[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "std": _std(values),
            }

    # Sort by mean descending
    if sort:
        stats = dict(sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True))

    return stats


def best_by(
    results: List[ExperimentResult],
    metric: str = "f1",
    n: int = 10,
    filter_fn: Optional[Callable[[ExperimentResult], bool]] = None,
) -> List[ExperimentResult]:
    """Get the top N results by a metric.

    Args:
        results: List of experiment results
        metric: Metric to rank by
        n: Number of top results to return
        filter_fn: Optional filter function

    Returns:
        List of top N results sorted by metric descending

    Example:
        >>> best = best_by(results, metric="exact_match", n=5)
        >>> for r in best:
        ...     print(f"{r.name}: {r.exact_match:.2%}")
    """
    filtered = results
    if filter_fn:
        filtered = [r for r in results if filter_fn(r)]

    sorted_results = sorted(filtered, key=lambda r: getattr(r, metric, 0), reverse=True)
    return sorted_results[:n]


def pivot_results(
    results: List[ExperimentResult],
    rows: str = "model",
    cols: str = "dataset",
    metric: str = "f1",
    agg: str = "mean",
) -> Dict[str, Dict[str, float]]:
    """Create a pivot table of results.

    Args:
        results: List of experiment results
        rows: Dimension for rows (model, prompt, quantization, etc.)
        cols: Dimension for columns (dataset, retriever, etc.)
        metric: Metric to aggregate
        agg: Aggregation function (mean, max, min)

    Returns:
        Nested dict: {row_key: {col_key: value}}

    Example:
        >>> pivot = pivot_results(results, rows="model", cols="dataset", metric="f1")
        >>> # Pretty print as table
        >>> for model, datasets in pivot.items():
        ...     print(f"{model}: NQ={datasets.get('nq', 0):.2f}, HotpotQA={datasets.get('hotpotqa', 0):.2f}")
    """
    # Collect values per cell
    cells: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        row_key = r.model_short if rows == "model" else str(getattr(r, rows, "unknown"))
        col_key = str(getattr(r, cols, "unknown"))
        value = getattr(r, metric, None)

        if value is not None:
            cells[row_key][col_key].append(value)

    # Aggregate
    agg_fn = {"mean": lambda x: sum(x) / len(x), "max": max, "min": min}.get(agg, lambda x: x[0])

    pivot = {}
    for row_key, col_dict in cells.items():
        pivot[row_key] = {}
        for col_key, values in col_dict.items():
            if values:
                pivot[row_key][col_key] = agg_fn(values)

    return pivot


def summarize_results(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Generate summary statistics for a set of results.

    Args:
        results: List of experiment results

    Returns:
        Summary dict with counts, best configs, and averages

    Example:
        >>> summary = summarize_results(results)
        >>> print(f"Total: {summary['count']} experiments")
        >>> print(f"Best F1: {summary['best_f1']['name']} ({summary['best_f1']['value']:.2%})")
    """
    if not results:
        return {"count": 0}

    # Basic counts
    summary = {
        "count": len(results),
        "models": list(set(r.model_short for r in results)),
        "datasets": list(set(r.dataset for r in results)),
        "prompts": list(set(r.prompt for r in results)),
        "types": list(set(r.type for r in results)),
    }

    # Best by each metric
    for metric in ["f1", "exact_match", "bertscore_f1", "bleurt"]:
        best = max(results, key=lambda r: getattr(r, metric, 0))
        summary[f"best_{metric}"] = {
            "name": best.name,
            "value": getattr(best, metric, 0),
            "model": best.model_short,
            "dataset": best.dataset,
            "prompt": best.prompt,
        }

    # Averages
    for metric in ["f1", "exact_match", "bertscore_f1", "bleurt", "throughput_qps"]:
        values = [getattr(r, metric, 0) for r in results]
        summary[f"avg_{metric}"] = sum(values) / len(values) if values else 0

    # Total duration
    summary["total_duration_hours"] = sum(r.duration for r in results) / 3600

    return summary


def format_comparison_table(
    stats: Dict[str, Dict[str, Any]],
    title: str = "Comparison",
    metric: str = "f1",
) -> str:
    """Format comparison stats as a text table.

    Args:
        stats: Output from compare_results()
        title: Table title
        metric: Metric name for header

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f" {title}")
    lines.append("=" * 70)
    lines.append(f"{'Group':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'N':>6}")
    lines.append("-" * 70)

    for group, s in stats.items():
        lines.append(
            f"{group[:29]:<30} {s['mean']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['count']:>6}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def _std(values: List[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5

