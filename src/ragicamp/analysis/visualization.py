"""Visualization utilities for experiment results.

Provides matplotlib/seaborn plots for comparing experiments.
All functions return matplotlib Figure objects for notebook integration.

Example:
    from ragicamp.analysis import ResultsLoader
    from ragicamp.analysis.visualization import plot_comparison, plot_heatmap

    results = ResultsLoader("outputs/comprehensive_baseline").load_all()
    
    fig = plot_comparison(results, group_by="model", metric="f1")
    fig.savefig("model_comparison.png")
"""

from typing import Any, Dict, List, Optional, Tuple

from ragicamp.analysis.comparison import compare_results, pivot_results
from ragicamp.analysis.loader import ExperimentResult

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Check for seaborn
try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")


def plot_comparison(
    results: List[ExperimentResult],
    group_by: str = "model",
    metric: str = "f1",
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    show_error_bars: bool = True,
    color_palette: Optional[List[str]] = None,
) -> "plt.Figure":
    """Bar chart comparing groups by a metric.

    Args:
        results: List of experiment results
        group_by: Dimension to group by
        metric: Metric to compare
        figsize: Figure size
        title: Plot title
        show_error_bars: Show min/max as error bars
        color_palette: Custom colors

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    stats = compare_results(results, group_by=group_by, metric=metric)

    groups = list(stats.keys())
    means = [stats[g]["mean"] for g in groups]
    mins = [stats[g]["min"] for g in groups]
    maxs = [stats[g]["max"] for g in groups]

    fig, ax = plt.subplots(figsize=figsize)

    colors = color_palette or plt.cm.Set2.colors[: len(groups)]
    bars = ax.bar(groups, means, color=colors, edgecolor="black", linewidth=0.5)

    if show_error_bars:
        # Error bars showing range
        yerr_lower = [m - mn for m, mn in zip(means, mins)]
        yerr_upper = [mx - m for m, mx in zip(means, maxs)]
        ax.errorbar(
            groups,
            means,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="gray",
            capsize=5,
            capthick=1,
        )

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel(group_by.title(), fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title or f"{metric.upper()} by {group_by.title()}", fontsize=14)

    # Rotate x labels if needed
    if len(groups) > 4 or any(len(g) > 15 for g in groups):
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_heatmap(
    results: List[ExperimentResult],
    rows: str = "model",
    cols: str = "dataset",
    metric: str = "f1",
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    cmap: str = "YlGnBu",
    annotate: bool = True,
) -> "plt.Figure":
    """Heatmap of metric values across two dimensions.

    Args:
        results: List of experiment results
        rows: Dimension for rows
        cols: Dimension for columns
        metric: Metric to show
        figsize: Figure size
        title: Plot title
        cmap: Colormap name
        annotate: Show values in cells

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    pivot = pivot_results(results, rows=rows, cols=cols, metric=metric)

    # Convert to matrix
    row_keys = sorted(pivot.keys())
    col_keys = sorted(set(k for row in pivot.values() for k in row.keys()))

    matrix = []
    for row_key in row_keys:
        row_data = pivot.get(row_key, {})
        matrix.append([row_data.get(col, 0) for col in col_keys])

    fig, ax = plt.subplots(figsize=figsize)

    if SEABORN_AVAILABLE:
        import pandas as pd

        df = pd.DataFrame(matrix, index=row_keys, columns=col_keys)
        sns.heatmap(df, annot=annotate, fmt=".3f", cmap=cmap, ax=ax)
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(col_keys)))
        ax.set_yticks(range(len(row_keys)))
        ax.set_xticklabels(col_keys)
        ax.set_yticklabels(row_keys)
        plt.colorbar(im, ax=ax)

        if annotate:
            for i, row in enumerate(matrix):
                for j, val in enumerate(row):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    ax.set_xlabel(cols.title(), fontsize=12)
    ax.set_ylabel(rows.title(), fontsize=12)
    ax.set_title(title or f"{metric.upper()}: {rows.title()} × {cols.title()}", fontsize=14)

    plt.tight_layout()
    return fig


def plot_multi_metric(
    results: List[ExperimentResult],
    group_by: str = "model",
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
) -> "plt.Figure":
    """Grouped bar chart comparing multiple metrics.

    Args:
        results: List of experiment results
        group_by: Dimension to group by
        metrics: Metrics to compare (default: f1, exact_match, bertscore_f1)
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()
    import numpy as np

    metrics = metrics or ["f1", "exact_match", "bertscore_f1"]

    # Get stats for each metric
    all_stats = {}
    for metric in metrics:
        all_stats[metric] = compare_results(results, group_by=group_by, metric=metric)

    groups = list(all_stats[metrics[0]].keys())
    n_groups = len(groups)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    colors = plt.cm.Set2.colors[:n_metrics]

    for i, metric in enumerate(metrics):
        stats = all_stats[metric]
        values = [stats.get(g, {}).get("mean", 0) for g in groups]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.upper(), color=colors[i])

    ax.set_xlabel(group_by.title(), fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title or f"Metrics by {group_by.title()}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45 if n_groups > 4 else 0, ha="right")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_scatter(
    results: List[ExperimentResult],
    x_metric: str = "f1",
    y_metric: str = "throughput_qps",
    color_by: str = "model",
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
) -> "plt.Figure":
    """Scatter plot of two metrics, colored by a dimension.

    Args:
        results: List of experiment results
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        color_by: Dimension to color by
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Group by color dimension
    groups = {}
    for r in results:
        key = r.model_short if color_by == "model" else str(getattr(r, color_by, "unknown"))
        if key not in groups:
            groups[key] = {"x": [], "y": [], "names": []}
        groups[key]["x"].append(getattr(r, x_metric, 0))
        groups[key]["y"].append(getattr(r, y_metric, 0))
        groups[key]["names"].append(r.name)

    colors = plt.cm.Set2.colors[: len(groups)]
    for i, (group, data) in enumerate(sorted(groups.items())):
        ax.scatter(data["x"], data["y"], label=group, color=colors[i], s=80, alpha=0.7)

    ax.set_xlabel(x_metric.upper(), fontsize=12)
    ax.set_ylabel(y_metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{y_metric.title()} vs {x_metric.upper()}", fontsize=14)
    ax.legend(title=color_by.title())

    plt.tight_layout()
    return fig


def plot_distribution(
    results: List[ExperimentResult],
    metric: str = "f1",
    group_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> "plt.Figure":
    """Box plot or violin plot of metric distribution.

    Args:
        results: List of experiment results
        metric: Metric to plot
        group_by: Optional grouping dimension
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if group_by:
        # Group data
        groups = {}
        for r in results:
            key = r.model_short if group_by == "model" else str(getattr(r, group_by, "unknown"))
            if key not in groups:
                groups[key] = []
            groups[key].append(getattr(r, metric, 0))

        if SEABORN_AVAILABLE:
            import pandas as pd

            data = []
            for group, values in groups.items():
                for v in values:
                    data.append({group_by: group, metric: v})
            df = pd.DataFrame(data)
            sns.boxplot(x=group_by, y=metric, data=df, ax=ax)
        else:
            labels = list(groups.keys())
            data = [groups[k] for k in labels]
            ax.boxplot(data, labels=labels)
    else:
        values = [getattr(r, metric, 0) for r in results]
        ax.hist(values, bins=20, edgecolor="black")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Count")

    ax.set_title(title or f"{metric.upper()} Distribution", fontsize=14)

    if group_by and len(groups) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig


def create_summary_dashboard(
    results: List[ExperimentResult],
    figsize: Tuple[int, int] = (16, 12),
) -> "plt.Figure":
    """Create a multi-panel summary dashboard.

    Args:
        results: List of experiment results
        figsize: Figure size

    Returns:
        matplotlib Figure with 4 subplots
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. F1 by model
    stats = compare_results(results, group_by="model", metric="f1")
    groups = list(stats.keys())
    means = [stats[g]["mean"] for g in groups]
    axes[0, 0].bar(groups, means, color=plt.cm.Set2.colors[: len(groups)])
    axes[0, 0].set_title("F1 by Model", fontsize=12)
    axes[0, 0].set_ylabel("F1")
    for i, (g, m) in enumerate(zip(groups, means)):
        axes[0, 0].annotate(f"{m:.3f}", xy=(i, m), ha="center", va="bottom")

    # 2. Heatmap: model x dataset
    pivot = pivot_results(results, rows="model", cols="dataset", metric="f1")
    row_keys = sorted(pivot.keys())
    col_keys = sorted(set(k for row in pivot.values() for k in row.keys()))
    matrix = [[pivot.get(r, {}).get(c, 0) for c in col_keys] for r in row_keys]

    im = axes[0, 1].imshow(matrix, cmap="YlGnBu", aspect="auto")
    axes[0, 1].set_xticks(range(len(col_keys)))
    axes[0, 1].set_yticks(range(len(row_keys)))
    axes[0, 1].set_xticklabels(col_keys)
    axes[0, 1].set_yticklabels([r[:15] for r in row_keys])
    axes[0, 1].set_title("F1: Model × Dataset", fontsize=12)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            axes[0, 1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    # 3. Prompt comparison
    stats = compare_results(results, group_by="prompt", metric="f1")
    groups = list(stats.keys())
    means = [stats[g]["mean"] for g in groups]
    axes[1, 0].bar(groups, means, color=plt.cm.Set3.colors[: len(groups)])
    axes[1, 0].set_title("F1 by Prompt Style", fontsize=12)
    axes[1, 0].set_ylabel("F1")

    # 4. Type comparison (direct vs RAG)
    stats = compare_results(results, group_by="type", metric="f1")
    groups = list(stats.keys())
    means = [stats[g]["mean"] for g in groups]
    axes[1, 1].bar(groups, means, color=["#ff9999", "#99ccff"])
    axes[1, 1].set_title("F1: Direct LLM vs RAG", fontsize=12)
    axes[1, 1].set_ylabel("F1")
    for i, (g, m) in enumerate(zip(groups, means)):
        axes[1, 1].annotate(f"{m:.3f}", xy=(i, m), ha="center", va="bottom")

    plt.suptitle("Experiment Summary Dashboard", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

