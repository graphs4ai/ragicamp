"""MLflow integration utilities for experiment tracking.

This module provides helper functions for logging experiments, metrics,
and artifacts to MLflow.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mlflow
    from mlflow import MlflowClient
    
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None


class MLflowTracker:
    """Wrapper for MLflow tracking with graceful fallback.
    
    This class provides a consistent interface for MLflow tracking
    that gracefully degrades if MLflow is not installed.
    
    Example:
        >>> tracker = MLflowTracker(enabled=True, experiment_name="rag_eval")
        >>> with tracker.start_run(run_name="gemma_baseline"):
        ...     tracker.log_params({"model": "gemma-2b", "top_k": 5})
        ...     tracker.log_metrics({"f1": 0.85, "exact_match": 0.72})
        ...     tracker.log_artifact("outputs/predictions.json")
    """
    
    def __init__(
        self,
        enabled: bool = True,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow tracker.
        
        Args:
            enabled: Whether to enable MLflow tracking
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            run_name: Run name for this experiment
            tags: Tags to add to the run
        """
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self._run_id = None
        
        if not MLFLOW_AVAILABLE and enabled:
            print("⚠️  MLflow not installed. Tracking disabled.")
            print("   Install with: pip install mlflow")
            self.enabled = False
        
        if self.enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            if experiment_name:
                mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start MLflow run (context manager).
        
        Args:
            run_name: Optional run name override
            tags: Optional tags override
            
        Returns:
            Context manager for the run
        """
        if not self.enabled:
            return _DummyContext()
        
        run_name = run_name or self.run_name
        tags = {**self.tags, **(tags or {})}
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        self._run_id = run.info.run_id
        return run
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        if self.enabled:
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"⚠️  Failed to log param {key}: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.enabled:
            return
        
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        
        try:
            mlflow.log_params(flat_params)
        except Exception as e:
            print(f"⚠️  Failed to log params: {e}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.enabled:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                print(f"⚠️  Failed to log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled:
            return
        
        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"⚠️  Failed to log metrics: {e}")
    
    def log_artifact(self, path: str):
        """Log an artifact file.
        
        Args:
            path: Path to artifact file
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            print(f"⚠️  Failed to log artifact {path}: {e}")
    
    def log_artifacts(self, dir_path: str):
        """Log all artifacts in a directory.
        
        Args:
            dir_path: Path to directory containing artifacts
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifacts(dir_path)
        except Exception as e:
            print(f"⚠️  Failed to log artifacts from {dir_path}: {e}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_dict(dictionary, artifact_file)
        except Exception as e:
            print(f"⚠️  Failed to log dict as {artifact_file}: {e}")
    
    def log_text(self, text: str, artifact_file: str):
        """Log text as an artifact.
        
        Args:
            text: Text to log
            artifact_file: Filename for the artifact
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_text(text, artifact_file)
        except Exception as e:
            print(f"⚠️  Failed to log text as {artifact_file}: {e}")
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run.
        
        Args:
            key: Tag key
            value: Tag value
        """
        if self.enabled:
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                print(f"⚠️  Failed to set tag {key}: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags on the current run.
        
        Args:
            tags: Dictionary of tags
        """
        if self.enabled:
            try:
                mlflow.set_tags(tags)
            except Exception as e:
                print(f"⚠️  Failed to set tags: {e}")
    
    def get_run_id(self) -> Optional[str]:
        """Get the current MLflow run ID."""
        return self._run_id
    
    def end_run(self):
        """End the current MLflow run."""
        if self.enabled:
            mlflow.end_run()
            self._run_id = None
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to string if not a simple type
                if not isinstance(v, (str, int, float, bool, type(None))):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)


class _DummyContext:
    """Dummy context manager when MLflow is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def create_mlflow_tracker(config: Dict[str, Any]) -> MLflowTracker:
    """Create MLflow tracker from experiment config.
    
    Args:
        config: Experiment configuration with mlflow section
        
    Returns:
        MLflowTracker instance
    """
    mlflow_config = config.get("mlflow", {})
    
    return MLflowTracker(
        enabled=mlflow_config.get("enabled", True),
        experiment_name=mlflow_config.get("experiment_name"),
        tracking_uri=mlflow_config.get("tracking_uri"),
        run_name=mlflow_config.get("run_name"),
        tags=mlflow_config.get("tags", {}),
    )
