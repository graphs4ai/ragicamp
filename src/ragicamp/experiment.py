"""Unified Experiment abstraction for RAGiCamp.

This module provides a single, clean interface for running experiments:

    from ragicamp import Experiment

    exp = Experiment.from_config("conf/study/my_study.yaml")
    results = exp.run()

Or programmatically:

    exp = Experiment(
        name="my_experiment",
        model=model,
        agent=agent,
        dataset=dataset,
        metrics=metrics,
    )
    results = exp.run(batch_size=8)

Phases:
    INIT -> GENERATING -> GENERATED -> COMPUTING_METRICS -> COMPLETE
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ragicamp.agents.base import RAGAgent
from ragicamp.core.logging import get_logger
from ragicamp.datasets.base import QADataset
from ragicamp.execution import ResilientExecutor
from ragicamp.execution.phases import (
    ExecutionContext,
    GenerationHandler,
    InitHandler,
    MetricsHandler,
    PhaseHandler,
)
from ragicamp.state import (
    ExperimentHealth,
    ExperimentPhase,
    ExperimentState,
    check_health,
    detect_state,
)
from ragicamp.metrics.base import Metric
from ragicamp.models.base import LanguageModel
from ragicamp.utils.paths import ensure_dir
from ragicamp.utils.resource_manager import ResourceManager

# TYPE_CHECKING import to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


@dataclass(frozen=True)
class _MinimalSpec:
    """Minimal spec adapter for use with phase handlers.
    
    The Experiment class creates this to pass to handlers that
    expect an ExperimentSpec-like object.
    """
    name: str


class _MetricsIncompleteError(Exception):
    """Internal exception raised when some metrics fail to compute.

    This prevents the experiment from being marked complete, allowing
    the user to fix the issue (e.g., set API keys) and re-run.
    """

    def __init__(self, missing_metrics: List[str]):
        self.missing_metrics = missing_metrics
        super().__init__(f"Metrics incomplete: {missing_metrics}")


@dataclass
class ExperimentCallbacks:
    """Callbacks for monitoring experiment progress.

    All callbacks are optional. Set them to receive notifications at key points:

    Example:
        callbacks = ExperimentCallbacks(
            on_batch_start=lambda i, n: print(f"Starting batch {i}/{n}"),
            on_complete=lambda r: print(f"Done! F1={r.f1:.3f}"),
        )
        exp.run(callbacks=callbacks)
    """

    on_phase_start: Optional[Callable[[ExperimentPhase], None]] = None
    """Called when entering a new phase. Args: (phase)"""

    on_phase_end: Optional[Callable[[ExperimentPhase], None]] = None
    """Called when completing a phase. Args: (phase)"""

    on_batch_start: Optional[Callable[[int, int], None]] = None
    """Called before each batch. Args: (batch_index, total_batches)"""

    on_batch_end: Optional[Callable[[int, int, List[str]], None]] = None
    """Called after each batch. Args: (batch_index, total_batches, predictions)"""

    on_checkpoint: Optional[Callable[[int, Path], None]] = None
    """Called when checkpoint is saved. Args: (num_predictions, checkpoint_path)"""

    on_complete: Optional[Callable[["ExperimentResult"], None]] = None
    """Called when experiment completes. Args: (result)"""


@dataclass
class ExperimentResult:
    """Result of running an experiment."""

    name: str
    metrics: Dict[str, float]
    num_examples: int
    duration_seconds: float
    output_path: Optional[Path] = None
    predictions_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def f1(self) -> float:
        return self.metrics.get("f1", 0.0)

    @property
    def exact_match(self) -> float:
        return self.metrics.get("exact_match", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "num_examples": self.num_examples,
            "duration_seconds": self.duration_seconds,
            "output_path": str(self.output_path) if self.output_path else None,
            "predictions_path": str(self.predictions_path) if self.predictions_path else None,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        ensure_dir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        return f"ExperimentResult(name={self.name!r}, f1={self.f1:.3f}, em={self.exact_match:.3f})"


@dataclass
class Experiment:
    """Unified experiment abstraction with phased execution.

    Combines model, agent, dataset, and metrics into a single runnable unit.
    Handles checkpointing, resource management, and result saving automatically.

    Phases:
        1. INIT: Save metadata, export questions
        2. GENERATING: Generate predictions with checkpointing
        3. GENERATED: All predictions complete
        4. COMPUTING_METRICS: Compute all metrics
        5. COMPLETE: Save final results

    Each phase produces artifacts that enable resume from any point.
    """

    name: str
    agent: RAGAgent
    dataset: QADataset
    metrics: List[Metric] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Optional references for cleanup and metadata
    _model: Optional[LanguageModel] = field(default=None, repr=False)
    _spec: Optional["ExperimentSpec"] = field(default=None, repr=False)

    # Runtime state (not serialized)
    _state: Optional[ExperimentState] = field(default=None, repr=False)
    _callbacks: Optional[ExperimentCallbacks] = field(default=None, repr=False)
    _batch_size: int = field(default=8, repr=False)
    _min_batch_size: int = field(default=1, repr=False)
    _checkpoint_every: int = field(default=50, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    
    # Phase handlers (lazy initialized)
    _handlers: Optional[Dict[ExperimentPhase, PhaseHandler]] = field(default=None, repr=False)

    # =========================================================================
    # Public API
    # =========================================================================

    def check_health(self) -> ExperimentHealth:
        """Check experiment state and detect issues.

        Returns:
            ExperimentHealth with current phase, missing predictions/metrics, etc.
        """
        metric_names = [m.name for m in self.metrics]
        return check_health(self.output_path, metric_names)

    @property
    def output_path(self) -> Path:
        """Path to experiment output directory."""
        return self.output_dir / self.name

    @property
    def state_path(self) -> Path:
        return self.output_path / "state.json"

    @property
    def questions_path(self) -> Path:
        return self.output_path / "questions.json"

    @property
    def predictions_path(self) -> Path:
        return self.output_path / "predictions.json"

    @property
    def results_path(self) -> Path:
        return self.output_path / "results.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_path / "metadata.json"

    def run(
        self,
        batch_size: int = 8,
        min_batch_size: int = 1,
        checkpoint_every: int = 50,
        resume: bool = True,
        callbacks: Optional[ExperimentCallbacks] = None,
        phases: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Run the experiment with phase-aware execution.

        Args:
            batch_size: Number of examples to process in parallel
            min_batch_size: Minimum batch size to reduce to on CUDA errors (default: 1).
                On CUDA/OOM errors, batch size is halved until it reaches this floor.
            checkpoint_every: Save checkpoint every N examples
            resume: Resume from checkpoint if available
            callbacks: Optional callbacks for progress monitoring
            phases: Optional list of phases to run (default: all remaining)
            **kwargs: Additional arguments passed to agent.answer()

        Returns:
            ExperimentResult with metrics and metadata
        """
        self._callbacks = callbacks or ExperimentCallbacks()
        self._batch_size = batch_size
        self._min_batch_size = min_batch_size
        self._checkpoint_every = checkpoint_every
        self._start_time = time.time()
        self._kwargs = kwargs

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Check current state
        health = self.check_health()

        if health.is_complete and resume:
            logger.info("Experiment %s already complete. Loading results...", self.name)
            return self._load_result()

        # Determine which phases to run
        if phases:
            # Explicit phases requested
            target_phases = [ExperimentPhase(p) for p in phases]
        elif resume and health.can_resume:
            # Resume from current phase
            logger.info("Resuming %s from phase: %s", self.name, health.resume_phase.value)
            target_phases = self._phases_from(health.resume_phase)
        else:
            # Start fresh
            target_phases = list(ExperimentPhase)[:5]  # All except FAILED

        # Load or create state (use detect_state to validate artifacts)
        metric_names = [m.name for m in self.metrics]
        if self.state_path.exists() and resume:
            self._state = detect_state(self.output_path, metric_names)
        else:
            self._state = ExperimentState.new(metrics=metric_names)

        # Note: GPU memory cleared by caller before experiment starts
        logger.info("Running: %s", self.name)

        try:
            # Execute phases
            for phase in target_phases:
                if phase == ExperimentPhase.FAILED:
                    continue
                if self._state.is_past(phase) and phase != ExperimentPhase.COMPUTING_METRICS:
                    # Skip already completed phases (except metrics which can be re-run)
                    continue

                self._run_phase(phase)

            return self._load_result()

        except _MetricsIncompleteError as e:
            # Metrics incomplete - don't mark as failed, allow retry
            logger.warning("Experiment incomplete: %s", e)
            # State already saved in _phase_compute_metrics, just save partial results
            self._save_partial_result()
            raise

        except Exception as e:
            logger.error("Experiment failed: %s", e)
            self._state.set_error(str(e))
            self._state.save(self.state_path)
            raise

    def run_phase(self, phase: Union[str, ExperimentPhase]) -> None:
        """Run a specific phase only.

        Useful for recomputing just metrics or resuming generation.

        Args:
            phase: Phase to run (string or ExperimentPhase)
        """
        if isinstance(phase, str):
            phase = ExperimentPhase(phase)

        self._callbacks = ExperimentCallbacks()
        self._start_time = time.time()
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Load state
        if self.state_path.exists():
            self._state = ExperimentState.load(self.state_path)
        else:
            metric_names = [m.name for m in self.metrics]
            self._state = ExperimentState.new(metrics=metric_names)

        self._run_phase(phase)

    # =========================================================================
    # Phase Implementations
    # =========================================================================

    def _get_handlers(self) -> Dict[ExperimentPhase, PhaseHandler]:
        """Get or create phase handlers."""
        if self._handlers is None:
            self._handlers = {
                ExperimentPhase.INIT: InitHandler(),
                ExperimentPhase.GENERATING: GenerationHandler(),
                ExperimentPhase.COMPUTING_METRICS: MetricsHandler(),
            }
        return self._handlers

    def _create_context(self) -> ExecutionContext:
        """Create execution context for phase handlers."""
        return ExecutionContext(
            output_path=self.output_path,
            agent=self.agent,
            dataset=self.dataset,
            metrics=self.metrics,
            callbacks=self._callbacks,
            batch_size=self._batch_size,
            min_batch_size=self._min_batch_size,
            checkpoint_every=self._checkpoint_every,
            kwargs=getattr(self, "_kwargs", {}),
        )

    def _run_phase(self, phase: ExperimentPhase) -> None:
        """Execute a single phase."""
        if self._callbacks.on_phase_start:
            self._callbacks.on_phase_start(phase)

        self._state.advance_to(phase)
        self._state.save(self.state_path)

        # Use handlers for supported phases
        handlers = self._get_handlers()
        if phase in handlers:
            spec = _MinimalSpec(name=self.name)
            context = self._create_context()
            self._state = handlers[phase].execute(spec, self._state, context)
            
            # Post-handler validation for COMPUTING_METRICS
            if phase == ExperimentPhase.COMPUTING_METRICS:
                missing = set(self._state.metrics_requested) - set(self._state.metrics_computed)
                if missing:
                    logger.warning(
                        "Some metrics failed to compute: %s. Experiment will not be marked complete.",
                        list(missing),
                    )
                    raise _MetricsIncompleteError(list(missing))
        elif phase == ExperimentPhase.GENERATED:
            self._phase_generated()
        elif phase == ExperimentPhase.COMPLETE:
            self._phase_complete()

        self._state.save(self.state_path)

        if self._callbacks.on_phase_end:
            self._callbacks.on_phase_end(phase)

    def _phase_generated(self) -> None:
        """Phase 3: Mark generation as complete, unload model."""
        logger.info("Phase: GENERATED - cleaning up model")
        self._unload_model()

    def _phase_complete(self) -> None:
        """Phase 5: Create final summary and results."""
        logger.info("Phase: COMPLETE - saving results")

        # Load predictions for final metrics
        with open(self.predictions_path) as f:
            data = json.load(f)

        metrics = data.get("aggregate_metrics", {})
        duration = time.time() - self._start_time

        # Build result
        result = ExperimentResult(
            name=self.name,
            metrics=metrics,
            num_examples=len(data.get("predictions", [])),
            duration_seconds=duration,
            output_path=self.output_path,
            predictions_path=self.predictions_path,
            metadata={
                "agent": self.agent.name,
                "dataset": self.dataset.name,
                "batch_size": self._batch_size,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Save results
        with open(self.results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Log summary
        metrics_parts = []
        for key, val in result.metrics.items():
            if isinstance(val, float):
                if key in ("f1", "exact_match", "bertscore_f1", "bleurt", "llm_judge_qa"):
                    metrics_parts.append(f"{key}={val*100:.1f}%")
                else:
                    metrics_parts.append(f"{key}={val:.3f}")
        metrics_str = " ".join(metrics_parts) if metrics_parts else "no metrics"
        logger.info("Done! %s (%.1fs)", metrics_str, duration)

        if self._callbacks.on_complete:
            self._callbacks.on_complete(result)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _save_predictions(self, data: Dict[str, Any]) -> None:
        """Save predictions atomically."""
        temp_path = self.predictions_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.predictions_path)

        if self._callbacks.on_checkpoint:
            n = len(data.get("predictions", []))
            self._callbacks.on_checkpoint(n, self.predictions_path)

    def _unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if self._model and hasattr(self._model, "unload"):
            self._model.unload()
        elif hasattr(self.agent, "model") and hasattr(self.agent.model, "unload"):
            self.agent.model.unload()
        ResourceManager.clear_gpu_memory()

    def _load_result(self) -> ExperimentResult:
        """Load result from results.json."""
        with open(self.results_path) as f:
            data = json.load(f)
        return ExperimentResult(
            name=self.name,
            metrics=data.get("metrics", {}),
            num_examples=data.get("num_examples", 0),
            duration_seconds=data.get("duration_seconds", 0),
            output_path=self.output_path,
            predictions_path=self.predictions_path if self.predictions_path.exists() else None,
            metadata=data.get("metadata", {}),
        )

    def _save_partial_result(self) -> None:
        """Save partial results when metrics are incomplete.

        This creates a results.json with available metrics so the experiment
        can be loaded, but leaves state in COMPUTING_METRICS for retry.
        """
        if not self.predictions_path.exists():
            return

        with open(self.predictions_path) as f:
            data = json.load(f)

        metrics = data.get("aggregate_metrics", {})
        duration = time.time() - self._start_time

        result_data = {
            "name": self.name,
            "metrics": metrics,
            "num_examples": len(data.get("predictions", [])),
            "duration_seconds": duration,
            "completed_at": datetime.now().isoformat(),
            "partial": True,  # Mark as incomplete
            "metrics_computed": self._state.metrics_computed,
            "metrics_missing": list(
                set(self._state.metrics_requested) - set(self._state.metrics_computed)
            ),
        }

        with open(self.results_path, "w") as f:
            json.dump(result_data, f, indent=2)

    def _phases_from(self, phase: ExperimentPhase) -> List[ExperimentPhase]:
        """Get list of phases starting from the given phase."""
        all_phases = [
            ExperimentPhase.INIT,
            ExperimentPhase.GENERATING,
            ExperimentPhase.GENERATED,
            ExperimentPhase.COMPUTING_METRICS,
            ExperimentPhase.COMPLETE,
        ]
        try:
            idx = all_phases.index(phase)
            return all_phases[idx:]
        except ValueError:
            return all_phases

    @classmethod
    def from_spec(
        cls,
        spec: "ExperimentSpec",
        output_dir: Union[str, Path],
        limit: Optional[int] = None,
        judge_model: Any = None,
    ) -> "Experiment":
        """Create a fully-configured Experiment from an ExperimentSpec.
        
        This is the preferred way to create experiments as it handles all
        component creation automatically based on the spec.

        Args:
            spec: Experiment specification with all configuration
            output_dir: Base output directory (experiment creates subdir)
            limit: Optional limit on dataset examples
            judge_model: Optional LLM judge model for metrics

        Returns:
            Fully configured Experiment ready to run
            
        Example:
            >>> from ragicamp.spec import ExperimentSpec
            >>> spec = ExperimentSpec.from_dict({...})
            >>> exp = Experiment.from_spec(spec, "outputs/")
            >>> result = exp.run()
        """
        from ragicamp.factory import AgentFactory, DatasetFactory, MetricFactory, ModelFactory
        from ragicamp.spec import ExperimentSpec
        
        # Create model
        model_config = ModelFactory.parse_spec(spec.model, quantization=spec.quant)
        model = ModelFactory.create(model_config)
        
        # Create dataset
        dataset_config = DatasetFactory.parse_spec(spec.dataset, limit=limit)
        dataset = DatasetFactory.create(dataset_config)
        
        # Create agent (AgentFactory.from_spec handles all the wiring)
        agent = AgentFactory.from_spec(spec, model)
        
        # Create metrics
        metrics = MetricFactory.create(spec.metrics, judge_model=judge_model)
        
        return cls(
            name=spec.name,
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            output_dir=Path(output_dir),
            _model=model,
            _spec=spec,
        )
    
    @classmethod
    def from_components(
        cls,
        name: str,
        model: LanguageModel,
        agent: RAGAgent,
        dataset: QADataset,
        metrics: List[Metric],
        output_dir: Union[str, Path] = "outputs",
    ) -> "Experiment":
        """Create experiment from pre-built components.
        
        Use this when you've already created components manually.
        For most cases, prefer from_spec().

        Args:
            name: Experiment name
            model: Language model
            agent: RAG agent
            dataset: Evaluation dataset
            metrics: List of metrics
            output_dir: Output directory

        Returns:
            Configured Experiment
        """
        return cls(
            name=name,
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            output_dir=Path(output_dir),
            _model=model,
        )


def run_experiments(
    experiments: List[Experiment],
    skip_existing: bool = True,
    **kwargs: Any,
) -> List[ExperimentResult]:
    """Run multiple experiments with proper resource management.

    Args:
        experiments: List of experiments to run
        skip_existing: Skip experiments with existing results
        **kwargs: Arguments passed to each experiment's run()

    Returns:
        List of ExperimentResults
    """
    results = []

    for i, exp in enumerate(experiments, 1):
        logger.info("[%d/%d] %s", i, len(experiments), exp.name)

        health = exp.check_health()

        if skip_existing and health.is_complete:
            logger.info("Skipping %s (complete)", exp.name)
            results.append(exp._load_result())
            continue

        try:
            result = exp.run(**kwargs)
            results.append(result)
        except Exception as e:
            logger.error("Experiment %s failed: %s", exp.name, e)
            results.append(
                ExperimentResult(
                    name=exp.name,
                    metrics={},
                    num_examples=0,
                    duration_seconds=0,
                    metadata={"error": str(e)},
                )
            )

        ResourceManager.clear_gpu_memory()

    return results
