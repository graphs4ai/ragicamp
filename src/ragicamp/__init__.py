"""RAGiCamp: A modular framework for experimenting with RAG approaches.

Key modules:
- checkpointing: Experiment state and checkpoint management
- pipeline: Phase-based experiment orchestration
- training: Online/offline training with trajectory storage
- metrics: Evaluation metrics including async LLM-as-judge
- agents: RAG agents (DirectLLM, FixedRAG, BanditRAG, MDPRAG)
- models: Language model interfaces (HuggingFace, OpenAI)
- retrievers: Document retrieval (Dense, Sparse)
- datasets: QA datasets (NQ, HotpotQA, TriviaQA)
"""

__version__ = "0.3.0"

from ragicamp.factory import ComponentFactory
from ragicamp.registry import ComponentRegistry

# Core infrastructure (available but not imported by default to avoid overhead)
# from ragicamp.core import get_logger, RAGiCampError, AgentType, etc.

__all__ = [
    "ComponentFactory",
    "ComponentRegistry",
    "__version__",
]
