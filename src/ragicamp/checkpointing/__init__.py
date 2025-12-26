"""Checkpointing module for resumable experiments.

This module provides state management for experiments, allowing:
- Resume from any completed phase
- Resume from any question within a phase (question-level checkpointing)
- Track individual metric completion (resume after OOM)
- Detect config changes and invalidate stale checkpoints

Example:
    >>> from ragicamp.checkpointing import ExperimentState, PhaseStatus
    >>>
    >>> # Create or load state
    >>> state = ExperimentState.load_or_create(
    ...     path="outputs/my_exp_state.json",
    ...     name="my_experiment",
    ...     phase_names=["generation", "metrics"],
    ...     config={"model": "gemma-2b", "dataset": "nq"}
    ... )
    >>>
    >>> # Generation phase with checkpointing
    >>> if state.should_run_phase("generation"):
    ...     state.start_phase("generation", total_items=1000)
    ...     start_idx = state.get_checkpoint_idx("generation")
    ...
    ...     for i, example in enumerate(examples[start_idx:], start=start_idx):
    ...         prediction = generate(example)
    ...         predictions.append(prediction)
    ...
    ...         if i % 10 == 0:  # Checkpoint every 10
    ...             save_checkpoint(predictions, f"checkpoint_{i}.json")
    ...             state.update_phase_checkpoint("generation", i, f"checkpoint_{i}.json")
    ...             state.save()
    ...
    ...     state.complete_phase("generation", output_path="predictions.json")
    ...     state.save()
    >>>
    >>> # Metrics phase with per-metric checkpointing
    >>> if state.should_run_phase("metrics"):
    ...     pending = state.get_pending_metrics("metrics", ["exact_match", "f1", "bertscore"])
    ...
    ...     for metric_name in pending:
    ...         score = compute_metric(metric_name, predictions)
    ...         state.mark_metric_complete("metrics", metric_name)
    ...         state.save()  # Can resume if BERTScore OOMs
    ...
    ...     state.complete_phase("metrics", output_path="results.json")
    ...     state.save()
"""

from ragicamp.utils.experiment_state import (
    ExperimentState,
    PhaseState,
    PhaseStatus,
    compute_config_hash,
)

__all__ = [
    "ExperimentState",
    "PhaseState",
    "PhaseStatus",
    "compute_config_hash",
]
