"""Experiment state management for phased execution.

This module provides state tracking for experiments, enabling:
- Resume from any phase after crash/interruption
- Health checks to detect incomplete experiments
- Partial re-execution (e.g., recompute only metrics)

Phases:
    INIT -> GENERATING -> GENERATED -> COMPUTING_METRICS -> COMPLETE
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentPhase(Enum):
    """Experiment lifecycle phases."""

    INIT = "init"  # Config saved, questions exported
    GENERATING = "generating"  # Predictions in progress
    GENERATED = "generated"  # All predictions complete
    COMPUTING_METRICS = "computing_metrics"  # Metrics being computed
    COMPLETE = "complete"  # All done
    FAILED = "failed"  # Error occurred


# Phase ordering for comparison
_PHASE_ORDER = {
    ExperimentPhase.INIT: 0,
    ExperimentPhase.GENERATING: 1,
    ExperimentPhase.GENERATED: 2,
    ExperimentPhase.COMPUTING_METRICS: 3,
    ExperimentPhase.COMPLETE: 4,
    ExperimentPhase.FAILED: -1,
}


@dataclass
class ExperimentState:
    """Persistent state for an experiment.

    Saved to state.json in the experiment output directory.
    """

    phase: ExperimentPhase
    started_at: str
    updated_at: str
    total_questions: int = 0
    predictions_complete: int = 0
    metrics_computed: List[str] = field(default_factory=list)
    metrics_requested: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def save(self, path: Path) -> None:
        """Save state to JSON file atomically."""
        self.updated_at = datetime.now().isoformat()
        data = {
            "phase": self.phase.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "total_questions": self.total_questions,
            "predictions_complete": self.predictions_complete,
            "metrics_computed": self.metrics_computed,
            "metrics_requested": self.metrics_requested,
            "error": self.error,
        }
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ExperimentState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            phase=ExperimentPhase(data["phase"]),
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            total_questions=data.get("total_questions", 0),
            predictions_complete=data.get("predictions_complete", 0),
            metrics_computed=data.get("metrics_computed", []),
            metrics_requested=data.get("metrics_requested", []),
            error=data.get("error"),
        )

    @classmethod
    def new(cls, total_questions: int = 0, metrics: Optional[List[str]] = None) -> "ExperimentState":
        """Create a new experiment state."""
        now = datetime.now().isoformat()
        return cls(
            phase=ExperimentPhase.INIT,
            started_at=now,
            updated_at=now,
            total_questions=total_questions,
            metrics_requested=metrics or [],
        )

    def advance_to(self, phase: ExperimentPhase) -> None:
        """Advance to a new phase."""
        self.phase = phase
        self.updated_at = datetime.now().isoformat()

    def set_error(self, error: str) -> None:
        """Mark experiment as failed with error."""
        self.error = error
        self.phase = ExperimentPhase.FAILED
        self.updated_at = datetime.now().isoformat()

    def is_at_least(self, phase: ExperimentPhase) -> bool:
        """Check if we're at or past the given phase (phase is at least started)."""
        return _PHASE_ORDER.get(self.phase, -1) >= _PHASE_ORDER.get(phase, 0)

    def is_past(self, phase: ExperimentPhase) -> bool:
        """Check if we've completed the given phase and moved past it."""
        return _PHASE_ORDER.get(self.phase, -1) > _PHASE_ORDER.get(phase, 0)


@dataclass
class ExperimentHealth:
    """Health check result for an experiment."""

    phase: ExperimentPhase
    is_complete: bool
    is_healthy: bool  # No errors, consistent state
    total_questions: int
    predictions_complete: int
    missing_predictions: List[int]  # Question indices missing predictions
    metrics_computed: List[str]
    metrics_missing: List[str]
    error: Optional[str] = None

    @property
    def can_resume(self) -> bool:
        """Check if experiment can be resumed (has partial work to continue).

        Returns False for fresh experiments (INIT with no predictions) - those
        should be 'started', not 'resumed'.
        """
        if self.is_complete or self.phase == ExperimentPhase.FAILED:
            return False
        # Fresh experiment - nothing to resume, start fresh
        if self.phase == ExperimentPhase.INIT and self.predictions_complete == 0:
            return False
        return True

    @property
    def resume_phase(self) -> Optional[ExperimentPhase]:
        """Determine which phase to resume from."""
        if self.is_complete:
            return None
        if self.phase == ExperimentPhase.FAILED:
            return None

        # If we haven't started yet (INIT phase with no predictions), start from beginning
        if self.phase == ExperimentPhase.INIT:
            return ExperimentPhase.INIT

        # If there are missing predictions, we need to generate
        if self.missing_predictions:
            return ExperimentPhase.GENERATING

        # If no predictions were made at all, we need to generate (not compute metrics)
        if self.predictions_complete == 0:
            return ExperimentPhase.GENERATING

        # If we have predictions but missing metrics, compute them
        if self.metrics_missing:
            return ExperimentPhase.COMPUTING_METRICS

        return self.phase

    @property
    def needs_generation(self) -> bool:
        """Check if predictions are incomplete."""
        return len(self.missing_predictions) > 0

    @property
    def needs_metrics(self) -> bool:
        """Check if metrics need to be computed."""
        return len(self.metrics_missing) > 0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_complete:
            return f"✓ Complete ({self.predictions_complete} predictions, {len(self.metrics_computed)} metrics)"
        if self.phase == ExperimentPhase.FAILED:
            return f"✗ Failed: {self.error}"
        parts = []
        if self.needs_generation:
            parts.append(f"predictions: {self.predictions_complete}/{self.total_questions}")
        if self.needs_metrics:
            parts.append(f"missing metrics: {', '.join(self.metrics_missing)}")
        return f"○ {self.phase.value} - {'; '.join(parts)}"


def _validate_state_artifacts(state: ExperimentState, exp_dir: Path) -> ExperimentState:
    """Validate that required artifacts exist for the claimed phase.

    If artifacts are missing, adjust the phase to match actual state on disk.
    This handles cases where state.json claims a phase but files are missing.
    """
    predictions_path = exp_dir / "predictions.json"
    questions_path = exp_dir / "questions.json"
    results_path = exp_dir / "results.json"

    original_phase = state.phase

    # COMPLETE phase requires results.json
    if state.phase == ExperimentPhase.COMPLETE:
        if not results_path.exists():
            # Downgrade to COMPUTING_METRICS if predictions exist
            if predictions_path.exists():
                state.phase = ExperimentPhase.COMPUTING_METRICS
            elif questions_path.exists():
                state.phase = ExperimentPhase.GENERATING
            else:
                state.phase = ExperimentPhase.INIT

    # COMPUTING_METRICS phase requires predictions.json
    if state.phase == ExperimentPhase.COMPUTING_METRICS:
        if not predictions_path.exists():
            # Downgrade to GENERATING if questions exist
            if questions_path.exists():
                state.phase = ExperimentPhase.GENERATING
                state.predictions_complete = 0
            else:
                state.phase = ExperimentPhase.INIT
                state.predictions_complete = 0

    # GENERATED phase requires predictions.json
    if state.phase == ExperimentPhase.GENERATED:
        if not predictions_path.exists():
            if questions_path.exists():
                state.phase = ExperimentPhase.GENERATING
                state.predictions_complete = 0
            else:
                state.phase = ExperimentPhase.INIT
                state.predictions_complete = 0

    # GENERATING phase requires questions.json
    if state.phase == ExperimentPhase.GENERATING:
        if not questions_path.exists():
            state.phase = ExperimentPhase.INIT
            state.predictions_complete = 0

    # Log if we adjusted the phase
    if state.phase != original_phase:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "State mismatch detected: state.json claims %s but artifacts missing. "
            "Adjusted to %s",
            original_phase.value,
            state.phase.value,
        )

    return state


def detect_state(exp_dir: Path, requested_metrics: Optional[List[str]] = None) -> ExperimentState:
    """Detect experiment state from files on disk.

    Args:
        exp_dir: Experiment output directory
        requested_metrics: Metrics that should be computed

    Returns:
        ExperimentState inferred from files
    """
    requested_metrics = requested_metrics or []
    state_path = exp_dir / "state.json"
    results_path = exp_dir / "results.json"
    predictions_path = exp_dir / "predictions.json"
    questions_path = exp_dir / "questions.json"
    metadata_path = exp_dir / "metadata.json"

    # If state.json exists, load it but VERIFY artifacts match the claimed phase
    if state_path.exists():
        state = ExperimentState.load(state_path)
        # Update requested metrics if provided
        if requested_metrics:
            state.metrics_requested = requested_metrics

        # Validate that required artifacts exist for the claimed phase
        # If artifacts are missing, adjust phase accordingly
        state = _validate_state_artifacts(state, exp_dir)
        return state

    # Otherwise, detect from files
    now = datetime.now().isoformat()

    # Check for completion first
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        metrics = list(data.get("metrics", {}).keys())
        num = data.get("num_examples", 0)
        return ExperimentState(
            phase=ExperimentPhase.COMPLETE,
            started_at=now,
            updated_at=now,
            total_questions=num,
            predictions_complete=num,
            metrics_computed=metrics,
            metrics_requested=requested_metrics or metrics,
        )

    # Check for predictions
    if predictions_path.exists():
        with open(predictions_path) as f:
            data = json.load(f)
        preds = data.get("predictions", [])
        total = len(preds)

        # Determine if generation is complete
        # (heuristic: if we have aggregate_metrics, generation is done)
        if data.get("aggregate_metrics"):
            # Check which metrics are computed
            computed = list(data.get("aggregate_metrics", {}).keys())
            return ExperimentState(
                phase=ExperimentPhase.COMPUTING_METRICS,
                started_at=now,
                updated_at=now,
                total_questions=total,
                predictions_complete=total,
                metrics_computed=computed,
                metrics_requested=requested_metrics,
            )
        else:
            return ExperimentState(
                phase=ExperimentPhase.GENERATED,
                started_at=now,
                updated_at=now,
                total_questions=total,
                predictions_complete=total,
                metrics_requested=requested_metrics,
            )

    # Check for questions
    if questions_path.exists():
        with open(questions_path) as f:
            data = json.load(f)
        total = data.get("count", len(data.get("questions", [])))
        return ExperimentState(
            phase=ExperimentPhase.INIT,
            started_at=now,
            updated_at=now,
            total_questions=total,
            metrics_requested=requested_metrics,
        )

    # Check for metadata only
    if metadata_path.exists():
        return ExperimentState(
            phase=ExperimentPhase.INIT,
            started_at=now,
            updated_at=now,
            metrics_requested=requested_metrics,
        )

    # Nothing exists - new experiment
    return ExperimentState.new(metrics=requested_metrics)


def check_health(
    exp_dir: Path,
    requested_metrics: Optional[List[str]] = None,
) -> ExperimentHealth:
    """Check health of an experiment.

    Args:
        exp_dir: Experiment output directory
        requested_metrics: Metrics that should be computed

    Returns:
        ExperimentHealth with detailed status
    """
    requested_metrics = requested_metrics or []
    state = detect_state(exp_dir, requested_metrics)

    # Determine missing predictions
    missing_predictions = []
    predictions_path = exp_dir / "predictions.json"
    questions_path = exp_dir / "questions.json"

    total_questions = state.total_questions
    predictions_complete = state.predictions_complete

    if questions_path.exists():
        with open(questions_path) as f:
            q_data = json.load(f)
        q_indices = set(q["idx"] for q in q_data.get("questions", []))
        total_questions = len(q_indices)

        if predictions_path.exists():
            with open(predictions_path) as f:
                p_data = json.load(f)
            p_indices = {p.get("idx", i) for i, p in enumerate(p_data.get("predictions", []))}
            missing_predictions = sorted(q_indices - p_indices)
            predictions_complete = len(p_indices)
        else:
            # No predictions at all - all questions are missing
            missing_predictions = sorted(q_indices)
            predictions_complete = 0

    # Determine missing metrics
    metrics_computed = state.metrics_computed
    metrics_missing = [m for m in requested_metrics if m not in metrics_computed]

    return ExperimentHealth(
        phase=state.phase,
        is_complete=state.phase == ExperimentPhase.COMPLETE,
        is_healthy=state.error is None and state.phase != ExperimentPhase.FAILED,
        total_questions=total_questions,
        predictions_complete=predictions_complete,
        missing_predictions=missing_predictions,
        metrics_computed=metrics_computed,
        metrics_missing=metrics_missing,
        error=state.error,
    )

