"""Analysis module for RAGiCamp experiment results.

Provides tools for loading, comparing, and visualizing experiment results.

Example:
    from ragicamp.analysis import ResultsLoader, compare_results

    # Load all results from a study
    loader = ResultsLoader("outputs/comprehensive_baseline")
    results = loader.load_all()

    # Compare by different dimensions
    compare_results(results, group_by="model", metric="f1")
    compare_results(results, group_by="retriever", metric="exact_match")

    # Log to MLflow
    from ragicamp.analysis import MLflowTracker
    tracker = MLflowTracker("my_study")
    tracker.backfill_from_results(results)
"""

from ragicamp.analysis.comparison import (
    best_by,
    compare_results,
    format_comparison_table,
    pivot_results,
    summarize_results,
)
from ragicamp.analysis.loader import ExperimentResult, ResultsLoader

# MLflow is optional
try:
    from ragicamp.analysis.mlflow_tracker import MLflowTracker, log_to_mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLflowTracker = None  # type: ignore
    log_to_mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

# Visualization is optional (requires matplotlib)
try:
    from ragicamp.analysis.visualization import (
        create_summary_dashboard,
        plot_comparison,
        plot_distribution,
        plot_heatmap,
        plot_multi_metric,
        plot_scatter,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    plot_comparison = None  # type: ignore
    plot_heatmap = None  # type: ignore
    plot_multi_metric = None  # type: ignore
    plot_scatter = None  # type: ignore
    plot_distribution = None  # type: ignore
    create_summary_dashboard = None  # type: ignore
    VISUALIZATION_AVAILABLE = False

__all__ = [
    # Loader
    "ResultsLoader",
    "ExperimentResult",
    # Comparison
    "compare_results",
    "best_by",
    "pivot_results",
    "summarize_results",
    "format_comparison_table",
    # MLflow
    "MLflowTracker",
    "log_to_mlflow",
    "MLFLOW_AVAILABLE",
    # Visualization
    "plot_comparison",
    "plot_heatmap",
    "plot_multi_metric",
    "plot_scatter",
    "plot_distribution",
    "create_summary_dashboard",
    "VISUALIZATION_AVAILABLE",
]

