"""Pipeline orchestrator - coordinates experiment phases.

The orchestrator provides a clean, modular way to run experiments:
1. Define phases (generation, metrics, etc.)
2. Chain them together
3. Handle errors and resource cleanup automatically

Example:
    orchestrator = ExperimentOrchestrator()
    orchestrator.add_phase(GenerationPhase(...))
    orchestrator.add_phase(MetricsPhase(...))
    results = orchestrator.run(initial_inputs)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ragicamp.pipeline.phases import Phase, PhaseResult
from ragicamp.utils.resource_manager import ResourceManager


@dataclass
class PipelineResult:
    """Result from running the complete pipeline."""

    success: bool
    phase_results: Dict[str, PhaseResult]
    final_outputs: Dict[str, Any]
    error: Optional[str] = None


class ExperimentOrchestrator:
    """Orchestrates experiment phases with proper resource management.

    Features:
    - Sequential phase execution
    - Automatic data passing between phases
    - Memory cleanup between phases
    - Error handling and recovery

    Example:
        # Define the pipeline
        orchestrator = ExperimentOrchestrator()

        # Add phases
        orchestrator.add_phase(GenerationPhase(
            model_factory=lambda: HuggingFaceModel("google/gemma-2-2b-it"),
            agent_factory=lambda m, r: DirectLLMAgent("baseline", m),
        ))

        orchestrator.add_phase(MetricsPhase(
            metrics=[ExactMatchMetric(), F1Metric()],
        ))

        # Run
        result = orchestrator.run({
            "dataset": dataset,
            "output_path": "outputs/predictions.json",
        })

        print(result.final_outputs["results"])
    """

    def __init__(self, name: str = "experiment"):
        """Initialize orchestrator.

        Args:
            name: Name for logging
        """
        self.name = name
        self.phases: List[Phase] = []

    def add_phase(self, phase: Phase) -> "ExperimentOrchestrator":
        """Add a phase to the pipeline.

        Args:
            phase: Phase to add

        Returns:
            Self for chaining
        """
        self.phases.append(phase)
        return self

    def run(self, initial_inputs: Dict[str, Any]) -> PipelineResult:
        """Run all phases in sequence.

        Args:
            initial_inputs: Initial data to pass to first phase

        Returns:
            PipelineResult with all outputs
        """
        print(f"\n{'='*70}")
        print(f"🚀 STARTING PIPELINE: {self.name}")
        print(f"   Phases: {' → '.join(p.name for p in self.phases)}")
        print(f"{'='*70}")

        ResourceManager.print_memory_status("pipeline start")

        phase_results = {}
        current_inputs = initial_inputs.copy()

        for i, phase in enumerate(self.phases, 1):
            print(f"\n[{i}/{len(self.phases)}] Running phase: {phase.name}")

            try:
                result = phase.run(current_inputs)
                phase_results[phase.name] = result

                if not result.success:
                    return PipelineResult(
                        success=False,
                        phase_results=phase_results,
                        final_outputs={},
                        error=f"Phase '{phase.name}' failed: {result.error}",
                    )

                # Pass outputs to next phase
                current_inputs.update(result.data)
                if result.output_path:
                    current_inputs["last_output_path"] = result.output_path

                print(f"✓ Phase '{phase.name}' completed")

            except Exception as e:
                print(f"✗ Phase '{phase.name}' failed with exception: {e}")
                return PipelineResult(
                    success=False,
                    phase_results=phase_results,
                    final_outputs={},
                    error=str(e),
                )

            # Cleanup between phases
            ResourceManager.clear_gpu_memory()

        print(f"\n{'='*70}")
        print(f"✅ PIPELINE COMPLETED: {self.name}")
        print(f"{'='*70}")
        ResourceManager.print_memory_status("pipeline end")

        return PipelineResult(
            success=True,
            phase_results=phase_results,
            final_outputs=current_inputs,
        )


def create_rag_pipeline(
    model_factory,
    agent_factory,
    retriever,
    dataset,
    metrics,
    output_path: str,
    judge_model=None,
) -> PipelineResult:
    """Convenience function to create and run a RAG evaluation pipeline.

    Args:
        model_factory: Callable that creates the model
        agent_factory: Callable(model, retriever) that creates the agent
        retriever: Retriever for RAG
        dataset: Dataset to evaluate on
        metrics: List of metrics
        output_path: Where to save predictions
        judge_model: Optional judge model for LLM metrics

    Returns:
        PipelineResult

    Example:
        from ragicamp.pipeline import create_rag_pipeline

        result = create_rag_pipeline(
            model_factory=lambda: HuggingFaceModel("google/gemma-2-2b-it", load_in_4bit=True),
            agent_factory=lambda m, r: FixedRAGAgent("rag", m, r, top_k=5),
            retriever=DenseRetriever.load_index("wikipedia_small"),
            dataset=NaturalQuestionsDataset(split="validation", num_examples=20),
            metrics=[ExactMatchMetric(), F1Metric()],
            output_path="outputs/rag_predictions.json",
        )

        print(result.final_outputs["results"])
    """
    from ragicamp.pipeline.phases import GenerationPhase, MetricsPhase

    orchestrator = ExperimentOrchestrator(name="RAG Evaluation")

    orchestrator.add_phase(
        GenerationPhase(
            model_factory=model_factory,
            agent_factory=agent_factory,
            retriever=retriever,
        )
    )

    orchestrator.add_phase(
        MetricsPhase(
            metrics=metrics,
            judge_model=judge_model,
        )
    )

    return orchestrator.run(
        {
            "dataset": dataset,
            "output_path": output_path,
        }
    )
