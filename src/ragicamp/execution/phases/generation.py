"""Generation phase handler - generates predictions using the agent."""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, TYPE_CHECKING

from ragicamp.core.logging import get_logger
from ragicamp.execution.phases.base import ExecutionContext, PhaseHandler
from ragicamp.experiment_state import ExperimentPhase, ExperimentState

if TYPE_CHECKING:
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


class GenerationHandler(PhaseHandler):
    """Handler for the GENERATING phase.

    Generates predictions using the ResilientExecutor with:
    - Automatic batch size reduction on OOM
    - Checkpointing for resume capability
    - Progress tracking
    """

    def can_handle(self, phase: ExperimentPhase) -> bool:
        """Check if this handler processes GENERATING phase."""
        return phase == ExperimentPhase.GENERATING

    def execute(
        self,
        spec: "ExperimentSpec",
        state: ExperimentState,
        context: ExecutionContext,
    ) -> ExperimentState:
        """Generate predictions using ResilientExecutor.

        Loads questions, generates predictions in batches, and checkpoints
        progress to enable resume on interruption.
        """
        from ragicamp.execution import ResilientExecutor

        logger.info("Phase: GENERATING - generating predictions")

        questions_path = context.output_path / "questions.json"
        predictions_path = context.output_path / "predictions.json"
        state_path = context.output_path / "state.json"

        # Load questions
        with open(questions_path) as f:
            q_data = json.load(f)
        questions = q_data["questions"]

        # Load existing predictions if resuming
        predictions_data: Dict[str, Any] = {"experiment": spec.name, "predictions": []}
        completed_indices: Set[int] = set()

        if predictions_path.exists():
            with open(predictions_path) as f:
                predictions_data = json.load(f)
            completed_indices = {
                p.get("idx", i) for i, p in enumerate(predictions_data["predictions"])
            }
            logger.info(
                "Resuming: %d/%d predictions complete", len(completed_indices), len(questions)
            )

        # Find pending questions - format: (idx, question, expected_answers)
        pending = [
            (q["idx"], q["question"], q["expected"])
            for q in questions
            if q["idx"] not in completed_indices
        ]

        if not pending:
            logger.info("All predictions already complete")
            return state

        logger.info("Generating %d predictions...", len(pending))

        # Create executor with auto batch size reduction
        executor = ResilientExecutor(
            agent=context.agent,
            batch_size=context.batch_size,
            min_batch_size=context.min_batch_size,
        )

        # Checkpoint callback
        def on_checkpoint(results: List[Dict]) -> None:
            # Convert executor results to predictions format
            for r in results:
                if r["idx"] not in completed_indices:
                    pred_item = {
                        "idx": r["idx"],
                        "question": r["query"],
                        "prediction": r["prediction"],
                        "expected": r["expected"],
                        "prompt": r.get("prompt"),
                        "metrics": {},
                    }
                    # Include retrieved docs for RAG experiments
                    if "retrieved_docs" in r:
                        pred_item["retrieved_docs"] = r["retrieved_docs"]
                    predictions_data["predictions"].append(pred_item)
                    completed_indices.add(r["idx"])

            state.predictions_complete = len(predictions_data["predictions"])
            self._save_predictions(predictions_data, predictions_path)
            state.save(state_path)

        # Execute with resilient batching
        results = executor.execute(
            queries=pending,
            progress=True,
            checkpoint_every=context.checkpoint_every,
            checkpoint_callback=on_checkpoint if context.checkpoint_every else None,
            **context.kwargs,
        )

        # Add results to predictions (if not already added via checkpoint)
        for r in results:
            if r["idx"] not in completed_indices:
                pred_item = {
                    "idx": r["idx"],
                    "question": r["query"],
                    "prediction": r["prediction"],
                    "expected": r["expected"],
                    "prompt": r.get("prompt"),
                    "metrics": {},
                }
                # Include retrieved context for RAG experiments
                if "retrieved_context" in r:
                    pred_item["retrieved_context"] = r["retrieved_context"]
                predictions_data["predictions"].append(pred_item)

        # Final save
        state.predictions_complete = len(predictions_data["predictions"])
        self._save_predictions(predictions_data, predictions_path)
        logger.info("Generated %d predictions", len(predictions_data["predictions"]))

        return state

    def _save_predictions(self, data: Dict[str, Any], path: Path) -> None:
        """Save predictions atomically."""
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(path)
