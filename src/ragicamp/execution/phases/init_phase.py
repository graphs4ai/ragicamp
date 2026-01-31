"""Init phase handler - exports questions and metadata."""

import json
from typing import TYPE_CHECKING

from ragicamp.core.logging import get_logger
from ragicamp.execution.phases.base import ExecutionContext, PhaseHandler
from ragicamp.experiment_state import ExperimentPhase, ExperimentState

if TYPE_CHECKING:
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


class InitHandler(PhaseHandler):
    """Handler for the INIT phase.

    Exports questions and saves experiment metadata.
    This is the first phase of experiment execution.
    """

    def can_handle(self, phase: ExperimentPhase) -> bool:
        """Check if this handler processes INIT phase."""
        return phase == ExperimentPhase.INIT

    def execute(
        self,
        spec: "ExperimentSpec",
        state: ExperimentState,
        context: ExecutionContext,
    ) -> ExperimentState:
        """Export questions and metadata.

        Creates:
        - questions.json: List of all questions to be answered
        - metadata.json: Experiment configuration metadata
        """
        logger.info("Phase: INIT - exporting questions and metadata")

        questions_path = context.output_path / "questions.json"
        metadata_path = context.output_path / "metadata.json"

        # Export questions
        examples = list(context.dataset)
        questions_data = {
            "experiment": spec.name,
            "dataset": context.dataset.name,
            "count": len(examples),
            "questions": [
                {"idx": i, "question": ex.question, "expected": ex.answers}
                for i, ex in enumerate(examples)
            ],
        }
        with open(questions_path, "w") as f:
            json.dump(questions_data, f, indent=2)

        # Save metadata
        metadata = {
            "name": spec.name,
            "agent": context.agent.name,
            "dataset": context.dataset.name,
            "metrics": [m.name for m in context.metrics] if context.metrics else [],
            "started_at": state.started_at,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update state
        state.total_questions = len(examples)
        logger.info("Exported %d questions", len(examples))

        return state
