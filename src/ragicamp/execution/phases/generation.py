"""Generation phase handler - generates predictions using the agent."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragicamp.core.logging import get_logger
from ragicamp.execution.phases.base import ExecutionContext, PhaseHandler
from ragicamp.experiment_state import ExperimentPhase, ExperimentState
from ragicamp.utils.experiment_io import ExperimentIO

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
        
        When VLLM_SEQUENTIAL_MODELS is enabled, uses prefetch retrieval mode:
        1. Prefetch all retrievals (embedder uses full GPU)
        2. Unload embedder
        3. Load generator (uses full GPU)
        4. Generate answers using cached retrievals
        """
        from ragicamp.core.constants import Defaults
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
        predictions_data: dict[str, Any] = {"experiment": spec.name, "predictions": []}
        completed_indices: set[int] = set()

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
        
        # Sequential model loading: prefetch all retrievals, then unload embedder
        # This allows each model to use full GPU memory
        if Defaults.VLLM_SEQUENTIAL_MODELS and hasattr(context.agent, 'prefetch_retrievals'):
            pending_queries = [q[1] for q in pending]  # Extract just the query strings
            logger.info("Sequential mode: prefetching %d retrievals...", len(pending_queries))
            context.agent.prefetch_retrievals(pending_queries, show_progress=True)
            
            # Unload embedder to free GPU for generator
            if hasattr(context.agent, 'unload_embedder'):
                logger.info("Unloading embedder to free GPU for generator...")
                context.agent.unload_embedder()

        # Create executor with auto batch size reduction
        executor = ResilientExecutor(
            agent=context.agent,
            batch_size=context.batch_size,
            min_batch_size=context.min_batch_size,
        )

        # Checkpoint callback
        def on_checkpoint(results: list[dict]) -> None:
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
                    # Include metadata (agent_type, retrieved_docs with scores, etc.)
                    if "metadata" in r and r["metadata"]:
                        pred_item["metadata"] = r["metadata"]
                    # Include intermediate steps (for iterative/self-rag agents)
                    if "intermediate_steps" in r and r["intermediate_steps"]:
                        pred_item["intermediate_steps"] = r["intermediate_steps"]
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
                # Include retrieved context for RAG experiments (legacy)
                if "retrieved_context" in r:
                    pred_item["retrieved_context"] = r["retrieved_context"]
                # Include metadata (agent_type, retrieved_docs with scores, etc.)
                if "metadata" in r and r["metadata"]:
                    pred_item["metadata"] = r["metadata"]
                # Include intermediate steps (for iterative/self-rag agents)
                if "intermediate_steps" in r and r["intermediate_steps"]:
                    pred_item["intermediate_steps"] = r["intermediate_steps"]
                predictions_data["predictions"].append(pred_item)

        # Final save
        state.predictions_complete = len(predictions_data["predictions"])
        self._save_predictions(predictions_data, predictions_path)
        logger.info("Generated %d predictions", len(predictions_data["predictions"]))

        return state

    def _save_predictions(self, data: dict[str, Any], path: Path) -> None:
        """Save predictions atomically using ExperimentIO."""
        io = ExperimentIO(path.parent)
        io.save_predictions(data)
