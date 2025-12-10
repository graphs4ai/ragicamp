"""Pipeline module - modular experiment orchestration with resource management.

This module provides clean abstractions for running experiments:
- ResourceManager: GPU/RAM lifecycle management
- Phases: Modular building blocks (Generation, Metrics, etc.)
- Orchestrator: Coordinates phases with automatic cleanup

Quick Start:
    from ragicamp.pipeline import create_rag_pipeline, ResourceManager

    # Simple one-liner
    result = create_rag_pipeline(
        model_factory=lambda: HuggingFaceModel("google/gemma-2-2b-it", load_in_4bit=True),
        agent_factory=lambda m, r: FixedRAGAgent("rag", m, r, top_k=5),
        retriever=retriever,
        dataset=dataset,
        metrics=[ExactMatchMetric(), F1Metric()],
        output_path="outputs/predictions.json",
    )

Advanced Usage:
    from ragicamp.pipeline import ExperimentOrchestrator, GenerationPhase, MetricsPhase

    # Build custom pipeline
    orchestrator = ExperimentOrchestrator("my_experiment")
    orchestrator.add_phase(GenerationPhase(...))
    orchestrator.add_phase(MetricsPhase(...))
    result = orchestrator.run(inputs)
"""

from ragicamp.pipeline.orchestrator import (
    ExperimentOrchestrator,
    PipelineResult,
    create_rag_pipeline,
)
from ragicamp.pipeline.phases import (
    GenerationPhase,
    MetricsPhase,
    Phase,
    PhaseResult,
)
from ragicamp.utils.resource_manager import (
    ResourceManager,
    gpu_memory_scope,
    managed_model,
)

__all__ = [
    # Resource management
    "ResourceManager",
    "gpu_memory_scope",
    "managed_model",
    # Phases
    "Phase",
    "PhaseResult",
    "GenerationPhase",
    "MetricsPhase",
    # Orchestration
    "ExperimentOrchestrator",
    "PipelineResult",
    "create_rag_pipeline",
]
