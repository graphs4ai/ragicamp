"""PredictionWriter - Consistent saving and loading of predictions.

This module provides a single point of control for all prediction I/O.
Uses the data contracts from core.schemas to ensure consistency.

Usage:
    from ragicamp.utils.prediction_writer import PredictionWriter
    
    writer = PredictionWriter(output_dir / "predictions.json")
    
    # Load existing or create new
    predictions = writer.load()
    
    # Add new predictions
    writer.append(new_predictions)
    
    # Save atomically
    writer.save()
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ragicamp.core.schemas import PredictionRecord, RetrievedDoc


class PredictionWriter:
    """Handles reading and writing of prediction files.
    
    Ensures all predictions follow the PredictionRecord schema and
    provides atomic saves to prevent data corruption.
    
    Attributes:
        path: Path to the predictions.json file
        experiment_name: Name of the experiment
        predictions: List of prediction records
    """
    
    def __init__(self, path: Path, experiment_name: str = ""):
        """Initialize the writer.
        
        Args:
            path: Path to predictions.json
            experiment_name: Name of the experiment
        """
        self.path = Path(path)
        self.experiment_name = experiment_name
        self._predictions: List[PredictionRecord] = []
        self._aggregate_metrics: Dict[str, float] = {}
        self._completed_indices: Set[int] = set()
    
    @property
    def predictions(self) -> List[PredictionRecord]:
        """Get all predictions."""
        return self._predictions
    
    @property
    def completed_indices(self) -> Set[int]:
        """Get indices of completed predictions."""
        return self._completed_indices
    
    @property
    def count(self) -> int:
        """Number of predictions."""
        return len(self._predictions)
    
    def load(self) -> List[PredictionRecord]:
        """Load predictions from file.
        
        Returns:
            List of PredictionRecord objects
        """
        if not self.path.exists():
            self._predictions = []
            self._completed_indices = set()
            return self._predictions
        
        with open(self.path) as f:
            data = json.load(f)
        
        self.experiment_name = data.get("experiment", self.experiment_name)
        self._aggregate_metrics = data.get("aggregate_metrics", {})
        
        self._predictions = []
        for p in data.get("predictions", []):
            self._predictions.append(self._dict_to_record(p))
        
        self._completed_indices = {p.idx for p in self._predictions}
        return self._predictions
    
    def append(self, predictions: List[Dict[str, Any]]) -> None:
        """Append new predictions.
        
        Args:
            predictions: List of prediction dicts (from executor)
        """
        for p in predictions:
            record = self._dict_to_record(p)
            self._predictions.append(record)
            self._completed_indices.add(record.idx)
    
    def append_record(self, record: PredictionRecord) -> None:
        """Append a single PredictionRecord."""
        self._predictions.append(record)
        self._completed_indices.add(record.idx)
    
    def update_metrics(self, aggregate_metrics: Dict[str, float]) -> None:
        """Update aggregate metrics.
        
        Args:
            aggregate_metrics: Dict of metric name -> score
        """
        self._aggregate_metrics.update(aggregate_metrics)
    
    def set_per_item_metric(self, idx: int, metric_name: str, value: float) -> None:
        """Set a metric value for a specific prediction.
        
        Args:
            idx: Prediction index
            metric_name: Name of the metric
            value: Metric value
        """
        for pred in self._predictions:
            if pred.idx == idx:
                pred.metrics[metric_name] = value
                return
    
    def save(self) -> None:
        """Save predictions atomically.
        
        Uses a temp file and atomic rename to prevent corruption.
        """
        data = {
            "experiment": self.experiment_name,
            "predictions": [self._record_to_dict(p) for p in self._predictions],
        }
        if self._aggregate_metrics:
            data["aggregate_metrics"] = self._aggregate_metrics
        
        # Atomic save: write to temp, then rename
        temp_path = self.path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.path)
    
    def _dict_to_record(self, d: Dict[str, Any]) -> PredictionRecord:
        """Convert dict to PredictionRecord.
        
        Handles both old format (retrieved_context as string) and
        new format (retrieved_docs as list).
        """
        retrieved_docs = None
        
        # Handle new format: retrieved_docs as list of dicts
        if "retrieved_docs" in d and d["retrieved_docs"]:
            retrieved_docs = [
                RetrievedDoc.from_dict(doc) if isinstance(doc, dict) else doc
                for doc in d["retrieved_docs"]
            ]
        
        # Handle old format: retrieved_context as string (convert to single doc)
        elif "retrieved_context" in d and d["retrieved_context"]:
            # Old format compatibility
            retrieved_docs = [
                RetrievedDoc(rank=1, content=str(d["retrieved_context"]))
            ]
        
        return PredictionRecord(
            idx=d.get("idx", 0),
            question=d.get("question", d.get("query", "")),
            prediction=d.get("prediction", ""),
            expected=d.get("expected", []),
            prompt=d.get("prompt", ""),
            retrieved_docs=retrieved_docs,
            metrics=d.get("metrics", {}),
            error=d.get("error"),
        )
    
    def _record_to_dict(self, record: PredictionRecord) -> Dict[str, Any]:
        """Convert PredictionRecord to dict for JSON serialization."""
        d = {
            "idx": record.idx,
            "question": record.question,
            "prediction": record.prediction,
            "expected": record.expected,
            "prompt": record.prompt,
            "metrics": record.metrics,
        }
        
        if record.retrieved_docs:
            d["retrieved_docs"] = [doc.to_dict() for doc in record.retrieved_docs]
        
        if record.error:
            d["error"] = record.error
        
        return d


def create_prediction_from_result(
    idx: int,
    question: str,
    prediction: str,
    expected: List[str],
    prompt: Optional[str] = None,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> PredictionRecord:
    """Factory function to create a PredictionRecord from executor results.
    
    Args:
        idx: Index in dataset
        question: The question asked
        prediction: Model's prediction
        expected: Expected answers
        prompt: Full prompt used
        retrieved_docs: Retrieved documents (for RAG)
        error: Error message if failed
    
    Returns:
        PredictionRecord instance
    """
    docs = None
    if retrieved_docs:
        docs = [
            RetrievedDoc.from_dict(d) if isinstance(d, dict) else d
            for d in retrieved_docs
        ]
    
    return PredictionRecord(
        idx=idx,
        question=question,
        prediction=prediction,
        expected=expected,
        prompt=prompt or "",
        retrieved_docs=docs,
        error=error,
    )
