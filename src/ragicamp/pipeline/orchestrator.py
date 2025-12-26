"""Pipeline orchestrator - coordinates experiment phases.

The orchestrator provides a clean, modular way to run experiments:
1. Define phases (generation, metrics, etc.)
2. Chain them together
3. Handle errors and resource cleanup automatically
4. Manage experiment state for checkpoint/resume

Example:
    orchestrator = ExperimentOrchestrator()
    orchestrator.add_phase(GenerationPhase(...))
    orchestrator.add_phase(MetricsPhase(...))
    results = orchestrator.run(initial_inputs)
    
With checkpointing:
    orchestrator = ExperimentOrchestrator(name="my_exp", state_path="outputs/state.json")
    orchestrator.add_phase(GenerationPhase(..., checkpoint_every=10))
    orchestrator.add_phase(MetricsPhase(...))
    
    # First run - might fail at question 50
    results = orchestrator.run({"dataset": ds, "output_path": "outputs/pred.json"})
    
    # Second run - automatically resumes from checkpoint
    results = orchestrator.run({"dataset": ds, "output_path": "outputs/pred.json"})
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragicamp.pipeline.phases import Phase, PhaseResult
from ragicamp.utils.experiment_state import ExperimentState
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
    - Checkpoint/resume via ExperimentState

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
        
    With checkpointing:
        orchestrator = ExperimentOrchestrator(
            name="my_experiment",
            state_path="outputs/my_experiment_state.json",
        )
        # If interrupted, re-running will resume from checkpoint
    """

    def __init__(
        self, 
        name: str = "experiment",
        state_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize orchestrator.

        Args:
            name: Name for logging and state management
            state_path: Path to save experiment state (enables checkpointing)
            config: Experiment configuration (stored in state for reproducibility)
        """
        self.name = name
        self.phases: List[Phase] = []
        self.state_path = Path(state_path) if state_path else None
        self.config = config or {}

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
        """Run all phases in sequence with optional checkpointing.

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
        
        # Initialize or load experiment state
        state = None
        if self.state_path:
            phase_names = [p.name for p in self.phases]
            state = ExperimentState.load_or_create(
                path=self.state_path,
                name=self.name,
                phase_names=phase_names,
                config=self.config,
            )
            print(f"\n{state.summary()}\n")

        phase_results = {}
        current_inputs = initial_inputs.copy()
        
        # Pass state to phases
        if state:
            current_inputs["experiment_state"] = state

        for i, phase in enumerate(self.phases, 1):
            # Check if phase should be skipped (already completed)
            if state and not state.should_run_phase(phase.name):
                print(f"\n[{i}/{len(self.phases)}] ⏭️  Skipping completed phase: {phase.name}")
                
                # Load output from previous run
                output_path = state.get_phase_output(phase.name)
                if output_path:
                    current_inputs["last_output_path"] = output_path
                continue
            
            print(f"\n[{i}/{len(self.phases)}] Running phase: {phase.name}")

            try:
                result = phase.run(current_inputs)
                phase_results[phase.name] = result

                if not result.success:
                    if state:
                        state.fail_phase(phase.name, result.error or "Unknown error")
                        state.save()
                    
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
                
                if state:
                    state.fail_phase(phase.name, str(e))
                    state.save()
                
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
        
        # Print final state summary
        if state:
            print(f"\n{state.summary()}")

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
    experiment_name: str = "RAG Evaluation",
    checkpoint_every: int = 10,
    enable_checkpointing: bool = True,
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
        experiment_name: Name for the experiment
        checkpoint_every: Save checkpoint every N questions
        enable_checkpointing: Enable checkpoint/resume (default: True)

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
            checkpoint_every=10,  # Save every 10 questions
        )

        print(result.final_outputs["results"])
    """
    from ragicamp.pipeline.phases import GenerationPhase, MetricsPhase

    # Determine state path from output path
    state_path = None
    if enable_checkpointing:
        state_path = str(Path(output_path).with_name(
            Path(output_path).stem + "_state.json"
        ))
    
    orchestrator = ExperimentOrchestrator(
        name=experiment_name,
        state_path=state_path,
    )

    orchestrator.add_phase(
        GenerationPhase(
            model_factory=model_factory,
            agent_factory=agent_factory,
            retriever=retriever,
            checkpoint_every=checkpoint_every,
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
