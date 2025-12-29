"""RAGiCamp: A modular framework for experimenting with RAG approaches.

Key modules:
- experiment: Unified experiment abstraction
- agents: RAG agents (DirectLLM, FixedRAG)
- models: Language model interfaces (HuggingFace, OpenAI)
- retrievers: Document retrieval (Dense, Sparse)
- datasets: QA datasets (NQ, HotpotQA, TriviaQA)
- metrics: Evaluation metrics (F1, EM, BERTScore, LLM-as-judge)
- corpus: Document corpus and chunking
- factory: Component creation from configs

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

__version__ = "0.3.0"

from ragicamp.experiment import (
    Experiment,
    ExperimentCallbacks,
    ExperimentResult,
    run_experiments,
)
from ragicamp.factory import ComponentFactory

__all__ = [
    # Core
    "Experiment",
    "ExperimentCallbacks",
    "ExperimentResult",
    "run_experiments",
    # Factory
    "ComponentFactory",
    # Version
    "__version__",
]
