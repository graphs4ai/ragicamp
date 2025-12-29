"""MLflow integration for experiment tracking.

Provides utilities for:
- Logging experiment results to MLflow
- Comparing runs in MLflow UI
- Backfilling existing results to MLflow

Example:
    from ragicamp.analysis import MLflowTracker

    # Log new experiment
    tracker = MLflowTracker(experiment_name="comprehensive_baseline")
    with tracker.start_run("my_experiment"):
        tracker.log_params({"model": "gemma-2b", "dataset": "nq"})
        tracker.log_metrics({"f1": 0.15, "exact_match": 0.07})

    # Backfill existing results
    tracker.backfill_from_results(results)
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)

# Check if mlflow is available
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class MLflowTracker:
    """MLflow experiment tracker.

    Wraps MLflow for convenient logging of RAGiCamp experiments.
    Handles:
    - Experiment creation
    - Run management
    - Parameter and metric logging
    - Artifact logging (predictions, configs)

    Example:
        tracker = MLflowTracker("my_study")

        # Log a single experiment
        tracker.log_experiment(experiment_result)

        # Or use context manager for manual logging
        with tracker.start_run("experiment_name") as run:
            tracker.log_params(params)
            result = run_experiment()
            tracker.log_metrics(result.metrics)
    """

    def __init__(
        self,
        experiment_name: str = "ragicamp",
        tracking_uri: Optional[str] = None,
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (default: ./mlruns)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow not installed. Run: pip install mlflow")

        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif not os.environ.get("MLFLOW_TRACKING_URI"):
            # Default to local mlruns directory
            mlflow.set_tracking_uri("file:./mlruns")

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info("Created MLflow experiment: %s", experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            logger.debug("Using existing MLflow experiment: %s", experiment_name)

        mlflow.set_experiment(experiment_name)

        self._active_run = None

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Optional tags

        Yields:
            MLflow run object
        """
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            self._active_run = run
            try:
                yield run
            finally:
                self._active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the active run.

        Args:
            params: Parameter dict
        """
        # MLflow params must be strings <=500 chars
        cleaned = {}
        for k, v in params.items():
            if v is None:
                continue
            str_val = str(v)[:500]
            cleaned[k] = str_val

        mlflow.log_params(cleaned)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the active run.

        Args:
            metrics: Metric dict (name -> value)
            step: Optional step number
        """
        # Filter to numeric values
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log an artifact file.

        Args:
            path: Local path to artifact
            artifact_path: Destination path in artifact store
        """
        mlflow.log_artifact(str(path), artifact_path)

    def log_experiment(
        self,
        result: "ExperimentResult",  # noqa: F821
        log_predictions: bool = False,
        predictions_path: Optional[Path] = None,
    ) -> str:
        """Log a complete experiment result to MLflow.

        Args:
            result: ExperimentResult from analysis.loader
            log_predictions: Whether to log predictions as artifact
            predictions_path: Path to predictions file

        Returns:
            MLflow run ID
        """
        from ragicamp.analysis.loader import ExperimentResult

        with self.start_run(run_name=result.name) as run:
            # Log parameters
            params = {
                "type": result.type,
                "model": result.model,
                "dataset": result.dataset,
                "prompt": result.prompt,
                "quantization": result.quantization,
                "retriever": result.retriever or "none",
                "top_k": result.top_k,
                "batch_size": result.batch_size,
                "num_questions": result.num_questions,
            }
            self.log_params(params)

            # Log metrics
            metrics = {
                "f1": result.f1,
                "exact_match": result.exact_match,
                "bertscore_f1": result.bertscore_f1,
                "bleurt": result.bleurt,
                "duration_seconds": result.duration,
                "throughput_qps": result.throughput_qps,
            }
            self.log_metrics(metrics)

            # Log predictions artifact if available
            if log_predictions and predictions_path and predictions_path.exists():
                self.log_artifact(predictions_path, "predictions")

            return run.info.run_id

    def backfill_from_results(
        self,
        results: List["ExperimentResult"],  # noqa: F821
        skip_existing: bool = True,
    ) -> int:
        """Backfill existing results into MLflow.

        Args:
            results: List of ExperimentResult objects
            skip_existing: Skip runs that already exist (by name)

        Returns:
            Number of experiments logged
        """
        # Get existing run names if skipping
        existing_names = set()
        if skip_existing:
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
            if not runs.empty:
                existing_names = set(runs["tags.mlflow.runName"].dropna())

        logged = 0
        for result in results:
            if result.name in existing_names:
                logger.debug("Skipping existing run: %s", result.name)
                continue

            try:
                self.log_experiment(result)
                logged += 1
            except Exception as e:
                logger.warning("Failed to log %s: %s", result.name, e)

        logger.info("Logged %d experiments to MLflow", logged)
        return logged

    @staticmethod
    def get_best_run(
        experiment_name: str,
        metric: str = "f1",
        ascending: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get the best run from an experiment.

        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            ascending: If True, lower is better

        Returns:
            Best run info dict or None
        """
        if not MLFLOW_AVAILABLE:
            return None

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None

        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if runs.empty:
            return None

        return runs.iloc[0].to_dict()


def log_to_mlflow(
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
) -> Optional[str]:
    """Convenience function to log a single experiment.

    Args:
        experiment_name: MLflow experiment name
        run_name: Run name
        params: Parameters dict
        metrics: Metrics dict

    Returns:
        Run ID or None if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping tracking")
        return None

    tracker = MLflowTracker(experiment_name)
    with tracker.start_run(run_name) as run:
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        return run.info.run_id

