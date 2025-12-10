"""Experiment state management for resumable multi-phase experiments.

This module provides robust state tracking for experiments, allowing:
- Resume from any completed phase
- Rerun specific phases (e.g., only metrics)
- Track intermediate outputs
- Atomic state updates
"""

import json
import shutil
from dataclasses import asdict, dataclass
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
    """State of a single experiment phase."""
    
    name: str
    status: PhaseStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def mark_started(self):
        """Mark phase as started."""
        self.status = PhaseStatus.IN_PROGRESS
        self.started_at = datetime.now().isoformat()
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentState:
    """Complete state of an experiment.
    
    Tracks all phases, outputs, and allows resumption from any point.
    
    Example:
        >>> state = ExperimentState(name="rag_eval", phases=["generation", "metrics"])
        >>> state.start_phase("generation")
        >>> # ... do generation ...
        >>> state.complete_phase("generation", output_path="outputs/preds.json")
        >>> state.save()
        >>> 
        >>> # Later, resume
        >>> state = ExperimentState.load("outputs/rag_eval_state.json")
        >>> if state.should_run_phase("metrics"):
        ...     # Run metrics
    """
    
    name: str
    phases: Dict[str, PhaseState]
    config: Dict[str, Any]
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
            name: PhaseState(name=name, status=PhaseStatus.PENDING)
            for name in phase_names
        }
        
        return cls(
            name=name,
            phases=phases,
            config=config,
            created_at=now,
            updated_at=now,
            mlflow_run_id=mlflow_run_id,
        )
    
    def start_phase(self, phase_name: str):
        """Mark a phase as started."""
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        self.phases[phase_name].mark_started()
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
        
        # Reconstruct PhaseState objects
        phases = {
            name: PhaseState(**phase_data)
            for name, phase_data in data["phases"].items()
        }
        
        return cls(
            name=data["name"],
            phases=phases,
            config=data["config"],
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
    ) -> "ExperimentState":
        """Load existing state or create new one.
        
        Args:
            path: Path to state file
            name: Experiment name (for creation)
            phase_names: Phase names (for creation)
            config: Config (for creation)
            mlflow_run_id: MLflow run ID (for creation)
            
        Returns:
            Loaded or new ExperimentState
        """
        path = Path(path)
        if path.exists():
            print(f"📂 Loading experiment state from {path}")
            return cls.load(path)
        else:
            print(f"📝 Creating new experiment state")
            return cls.create(name, phase_names, config, mlflow_run_id)
    
    def summary(self) -> str:
        """Get human-readable summary of state."""
        lines = [
            f"Experiment: {self.name}",
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
            lines.append(f"  {emoji} {name}: {phase.status.value}")
            if phase.output_path:
                lines.append(f"     Output: {phase.output_path}")
            if phase.error:
                lines.append(f"     Error: {phase.error}")
        
        return "\n".join(lines)
