"""RAGiCamp: A modular framework for experimenting with RAG approaches.

Key packages:
- spec: Experiment specifications (ExperimentSpec, build_specs, naming)
- state: State management (ExperimentState, ExperimentPhase, check_health)
- factory: Component factories (ModelFactory, DatasetFactory, etc.)
- execution: Batch execution (ResilientExecutor, phase handlers)
- agents: RAG agents (DirectLLM, FixedRAG)
- models: Language model interfaces (HuggingFace, OpenAI)
- retrievers: Document retrieval (Dense, Sparse)
- datasets: QA datasets (NQ, HotpotQA, TriviaQA)
- metrics: Evaluation metrics (F1, EM, BERTScore, LLM-as-judge)
- indexes: Index building and management

Quick start:
    from ragicamp import Experiment, ComponentFactory

    # Create components
    model = ComponentFactory.create_model({"type": "huggingface", "model_name": "..."})
    agent = ComponentFactory.create_agent({"type": "direct_llm", "name": "baseline"}, model)
    dataset = ComponentFactory.create_dataset({"name": "nq", "split": "validation"})

    # Run experiment
    exp = Experiment(name="my_exp", agent=agent, dataset=dataset, metrics=[...])
    result = exp.run(batch_size=8)
"""

# ============================================================================
# CRITICAL: Configure TensorFlow BEFORE any library imports!
#
# TensorFlow is transitively imported by transformers/sentence-transformers.
# By default, TF allocates ALL GPU memory on import, leaving no room for
# PyTorch models like BERTScore or the main LLM.
#
# This MUST happen before any other imports to be effective.
# ============================================================================
import os as _os

if "TF_FORCE_GPU_ALLOW_GROWTH" not in _os.environ:
    _os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
if "TF_CPP_MIN_LOG_LEVEL" not in _os.environ:
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning logs

__version__ = "0.4.0"

from ragicamp.execution import ResilientExecutor
from ragicamp.experiment import (
    Experiment,
    ExperimentCallbacks,
    ExperimentResult,
    run_experiments,
)
from ragicamp.factory import ComponentFactory

# Import from canonical locations (state/ package)
from ragicamp.state import (
    ExperimentHealth,
    ExperimentPhase,
    ExperimentState,
    check_health,
    detect_state,
)

__all__ = [
    # Core
    "Experiment",
    "ExperimentCallbacks",
    "ExperimentResult",
    "run_experiments",
    # State management
    "ExperimentPhase",
    "ExperimentState",
    "ExperimentHealth",
    "check_health",
    "detect_state",
    # Execution
    "ResilientExecutor",
    # Factory
    "ComponentFactory",
    # Version
    "__version__",
]
