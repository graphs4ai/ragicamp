"""Experiment state management for resumable multi-phase experiments.

This module provides robust state tracking for experiments, allowing:
- Resume from any completed phase
- Resume from any question within a phase (question-level checkpointing)
- Rerun specific phases (e.g., only metrics)
- Track individual metric completion (resume after OOM)
- Track intermediate outputs
- Atomic state updates
"""

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PhaseStatus(str, Enum):
    """Status of an experiment phase."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseState:
    """State of a single experiment phase.
    
    Supports both phase-level and item-level checkpointing:
    - For generation: tracks which question index we're at
    - For metrics: tracks which individual metrics are complete
    """
    
    name: str
    status: PhaseStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    # Question-level checkpointing for generation phase
    checkpoint_idx: int = 0  # Resume from this index
    total_items: int = 0  # Total items to process
    checkpoint_file: Optional[str] = None  # Path to checkpoint predictions
    
    # Metric-level checkpointing for metrics phase
    completed_metrics: List[str] = None  # Which metrics are done
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.completed_metrics is None:
            self.completed_metrics = []
    
    def mark_started(self, total_items: int = 0):
        """Mark phase as started.
        
        Args:
            total_items: Total number of items to process (for progress tracking)
        """
        self.status = PhaseStatus.IN_PROGRESS
        self.started_at = datetime.now().isoformat()
        self.total_items = total_items
    
    def mark_completed(self, output_path: Optional[str] = None, **metadata):
        """Mark phase as completed."""
        self.status = PhaseStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        if output_path:
            self.output_path = output_path
        self.metadata.update(metadata)
    
    def mark_failed(self, error: str):
        """Mark phase as failed."""
        self.status = PhaseStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.error = error
    
    def update_checkpoint(self, idx: int, checkpoint_file: Optional[str] = None):
        """Update checkpoint index for resumable processing.
        
        Args:
            idx: Current item index (will resume from idx+1)
            checkpoint_file: Path to intermediate checkpoint file
        """
        self.checkpoint_idx = idx
        if checkpoint_file:
            self.checkpoint_file = checkpoint_file
    
    def mark_metric_complete(self, metric_name: str):
        """Mark a specific metric as completed (for metrics phase).
        
        Args:
            metric_name: Name of the completed metric
        """
        if metric_name not in self.completed_metrics:
            self.completed_metrics.append(metric_name)
    
    def get_pending_metrics(self, all_metrics: List[str]) -> List[str]:
        """Get list of metrics that still need to be computed.
        
        Args:
            all_metrics: List of all metric names to compute
            
        Returns:
            List of metric names not yet completed
        """
        return [m for m in all_metrics if m not in self.completed_metrics]
    
    def can_resume(self) -> bool:
        """Check if this phase can be resumed from a checkpoint.
        
        Returns:
            True if there's a checkpoint to resume from
        """
        return (
            self.status == PhaseStatus.IN_PROGRESS 
            and self.checkpoint_idx > 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a hash of the config for change detection.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Hash string (first 16 chars of SHA256)
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class ExperimentState:
    """Complete state of an experiment.
    
    Tracks all phases, outputs, and allows resumption from any point.
    Supports both phase-level and item-level (question) checkpointing.
    
    Example:
        >>> state = ExperimentState.create("rag_eval", ["generation", "metrics"], config)
        >>> 
        >>> # Start generation with 1000 questions
        >>> state.start_phase("generation", total_items=1000)
        >>> for i, example in enumerate(examples):
        ...     # Process example
        ...     if i % 10 == 0:  # Checkpoint every 10
        ...         state.update_phase_checkpoint("generation", i, "checkpoint.json")
        ...         state.save()
        >>> state.complete_phase("generation", output_path="outputs/preds.json")
        >>> 
        >>> # If it fails at question 50, resume:
        >>> state = ExperimentState.load("outputs/rag_eval_state.json")
        >>> start_idx = state.get_checkpoint_idx("generation")  # Returns 50
        >>> # Continue from question 50
        >>> 
        >>> # For metrics, resume individual metrics after OOM:
        >>> pending = state.get_pending_metrics("metrics", ["exact_match", "f1", "bertscore"])
        >>> for metric in pending:  # Only runs bertscore if others done
        ...     compute_metric(metric)
        ...     state.mark_metric_complete("metrics", metric)
        ...     state.save()
    """
    
    name: str
    phases: Dict[str, PhaseState]
    config: Dict[str, Any]
    config_hash: str  # For detecting config changes
    created_at: str
    updated_at: str
    mlflow_run_id: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        name: str,
        phase_names: List[str],
        config: Dict[str, Any],
        mlflow_run_id: Optional[str] = None,
    ) -> "ExperimentState":
        """Create a new experiment state.
        
        Args:
            name: Experiment name
            phase_names: List of phase names in order
            config: Experiment configuration
            mlflow_run_id: Optional MLflow run ID
            
        Returns:
            New ExperimentState
        """
        now = datetime.now().isoformat()
        phases = {
            phase_name: PhaseState(name=phase_name, status=PhaseStatus.PENDING)
            for phase_name in phase_names
        }
        
        return cls(
            name=name,
            phases=phases,
            config=config,
            config_hash=compute_config_hash(config),
            created_at=now,
            updated_at=now,
            mlflow_run_id=mlflow_run_id,
        )
    
    def start_phase(self, phase_name: str, total_items: int = 0):
        """Mark a phase as started.
        
        Args:
            phase_name: Name of the phase to start
            total_items: Total number of items to process (for progress tracking)
        """
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phases[phase_name].mark_started(total_items=total_items)
        self.updated_at = datetime.now().isoformat()
    
    def complete_phase(
        self,
        phase_name: str,
        output_path: Optional[str] = None,
        **metadata
    ):
        """Mark a phase as completed.
        
        Args:
            phase_name: Name of the phase
            output_path: Path to phase output
            **metadata: Additional metadata to store
        """
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phases[phase_name].mark_completed(output_path, **metadata)
        self.updated_at = datetime.now().isoformat()
    
    def fail_phase(self, phase_name: str, error: str):
        """Mark a phase as failed."""
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phases[phase_name].mark_failed(error)
        self.updated_at = datetime.now().isoformat()
    
    def skip_phase(self, phase_name: str):
        """Mark a phase as skipped."""
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phases[phase_name].status = PhaseStatus.SKIPPED
        self.updated_at = datetime.now().isoformat()
    
    def update_phase_checkpoint(
        self, 
        phase_name: str, 
        idx: int, 
        checkpoint_file: Optional[str] = None
    ):
        """Update checkpoint index for a phase (for question-level resume).
        
        Args:
            phase_name: Name of the phase
            idx: Current item index (will resume from idx+1)
            checkpoint_file: Path to intermediate checkpoint file
        """
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phases[phase_name].update_checkpoint(idx, checkpoint_file)
        self.updated_at = datetime.now().isoformat()
    
    def get_checkpoint_idx(self, phase_name: str) -> int:
        """Get the checkpoint index for a phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Checkpoint index (0 if no checkpoint)
        """
        if phase_name not in self.phases:
            return 0
        return self.phases[phase_name].checkpoint_idx
    
    def get_checkpoint_file(self, phase_name: str) -> Optional[str]:
        """Get the checkpoint file path for a phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Checkpoint file path or None
        """
        if phase_name not in self.phases:
            return None
        return self.phases[phase_name].checkpoint_file
    
    def can_resume_phase(self, phase_name: str) -> bool:
        """Check if a phase can be resumed from checkpoint.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            True if there's a checkpoint to resume from
        """
        if phase_name not in self.phases:
            return False
        return self.phases[phase_name].can_resume()
    
    def mark_metric_complete(self, phase_name: str, metric_name: str):
        """Mark a specific metric as completed.
        
        Args:
            phase_name: Name of the metrics phase
            metric_name: Name of the completed metric
        """
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phases[phase_name].mark_metric_complete(metric_name)
        self.updated_at = datetime.now().isoformat()
    
    def get_pending_metrics(self, phase_name: str, all_metrics: List[str]) -> List[str]:
        """Get list of metrics that still need to be computed.
        
        Args:
            phase_name: Name of the metrics phase
            all_metrics: List of all metric names to compute
            
        Returns:
            List of metric names not yet completed
        """
        if phase_name not in self.phases:
            return all_metrics
        return self.phases[phase_name].get_pending_metrics(all_metrics)
    
    def config_changed(self, new_config: Dict[str, Any]) -> bool:
        """Check if config has changed since state was created.
        
        Args:
            new_config: New configuration to compare
            
        Returns:
            True if config has changed
        """
        new_hash = compute_config_hash(new_config)
        return new_hash != self.config_hash
    
    def should_run_phase(self, phase_name: str, force_rerun: bool = False) -> bool:
        """Check if a phase should be run.
        
        Args:
            phase_name: Name of the phase
            force_rerun: If True, run even if completed
            
        Returns:
            True if phase should run
        """
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        phase = self.phases[phase_name]
        
        if force_rerun:
            return True
        
        # Run if pending or failed
        return phase.status in [PhaseStatus.PENDING, PhaseStatus.FAILED]
    
    def get_phase_output(self, phase_name: str) -> Optional[str]:
        """Get output path from a completed phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Output path or None
        """
        if phase_name not in self.phases:
            return None
        
        phase = self.phases[phase_name]
        if phase.status == PhaseStatus.COMPLETED:
            return phase.output_path
        return None
    
    def get_completed_phases(self) -> List[str]:
        """Get list of completed phase names."""
        return [
            name for name, phase in self.phases.items()
            if phase.status == PhaseStatus.COMPLETED
        ]
    
    def get_pending_phases(self) -> List[str]:
        """Get list of pending phase names."""
        return [
            name for name, phase in self.phases.items()
            if phase.status == PhaseStatus.PENDING
        ]
    
    def is_complete(self) -> bool:
        """Check if all phases are completed."""
        return all(
            phase.status in [PhaseStatus.COMPLETED, PhaseStatus.SKIPPED]
            for phase in self.phases.values()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "phases": {name: phase.to_dict() for name, phase in self.phases.items()},
            "config": self.config,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "mlflow_run_id": self.mlflow_run_id,
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save state to disk (atomic write).
        
        Args:
            path: Optional path to save to (default: outputs/{name}_state.json)
            
        Returns:
            Path where state was saved
        """
        if path is None:
            path = Path("outputs") / f"{self.name}_state.json"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write: write to temp file, then rename
        temp_path = path.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Atomic rename
        shutil.move(str(temp_path), str(path))
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentState":
        """Load state from disk.
        
        Args:
            path: Path to state file
            
        Returns:
            Loaded ExperimentState
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct PhaseState objects with new fields
        phases = {}
        for name, phase_data in data["phases"].items():
            # Handle backward compatibility - add new fields if missing
            if "checkpoint_idx" not in phase_data:
                phase_data["checkpoint_idx"] = 0
            if "total_items" not in phase_data:
                phase_data["total_items"] = 0
            if "checkpoint_file" not in phase_data:
                phase_data["checkpoint_file"] = None
            if "completed_metrics" not in phase_data:
                phase_data["completed_metrics"] = []
            phases[name] = PhaseState(**phase_data)
        
        # Handle backward compatibility for config_hash
        config_hash = data.get("config_hash")
        if config_hash is None:
            config_hash = compute_config_hash(data["config"])
        
        return cls(
            name=data["name"],
            phases=phases,
            config=data["config"],
            config_hash=config_hash,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            mlflow_run_id=data.get("mlflow_run_id"),
        )
    
    @classmethod
    def load_or_create(
        cls,
        path: Path,
        name: str,
        phase_names: List[str],
        config: Dict[str, Any],
        mlflow_run_id: Optional[str] = None,
        invalidate_on_config_change: bool = True,
    ) -> "ExperimentState":
        """Load existing state or create new one.
        
        Args:
            path: Path to state file
            name: Experiment name (for creation)
            phase_names: Phase names (for creation)
            config: Config (for creation)
            mlflow_run_id: MLflow run ID (for creation)
            invalidate_on_config_change: If True, create new state if config changed
            
        Returns:
            Loaded or new ExperimentState
        """
        path = Path(path)
        if path.exists():
            print(f"📂 Loading experiment state from {path}")
            state = cls.load(path)
            
            # Check if config changed
            if state.config_changed(config):
                if invalidate_on_config_change:
                    print(f"⚠️  Config changed - creating new state (old checkpoint invalidated)")
                    return cls.create(name, phase_names, config, mlflow_run_id)
                else:
                    print(f"⚠️  Config changed but continuing with existing state")
            
            return state
        else:
            print(f"📝 Creating new experiment state")
            return cls.create(name, phase_names, config, mlflow_run_id)
    
    def summary(self) -> str:
        """Get human-readable summary of state."""
        lines = [
            f"Experiment: {self.name}",
            f"Config Hash: {self.config_hash}",
            f"Created: {self.created_at}",
            f"Updated: {self.updated_at}",
            f"MLflow Run: {self.mlflow_run_id or 'N/A'}",
            "",
            "Phases:",
        ]
        
        for name, phase in self.phases.items():
            status_emoji = {
                PhaseStatus.PENDING: "⏸️",
                PhaseStatus.IN_PROGRESS: "🔄",
                PhaseStatus.COMPLETED: "✅",
                PhaseStatus.FAILED: "❌",
                PhaseStatus.SKIPPED: "⏭️",
            }
            emoji = status_emoji.get(phase.status, "❓")
            
            # Show progress for in-progress phases
            if phase.status == PhaseStatus.IN_PROGRESS and phase.total_items > 0:
                progress = f" ({phase.checkpoint_idx}/{phase.total_items})"
            else:
                progress = ""
            
            lines.append(f"  {emoji} {name}: {phase.status.value}{progress}")
            
            if phase.output_path:
                lines.append(f"     Output: {phase.output_path}")
            if phase.checkpoint_file:
                lines.append(f"     Checkpoint: {phase.checkpoint_file}")
            if phase.completed_metrics:
                lines.append(f"     Completed metrics: {', '.join(phase.completed_metrics)}")
            if phase.error:
                lines.append(f"     Error: {phase.error}")
        
        return "\n".join(lines)
