"""RAGiCamp: A modular framework for experimenting with RAG approaches."""

__version__ = "0.2.0"

from ragicamp.factory import ComponentFactory
from ragicamp.registry import ComponentRegistry

# Core infrastructure (available but not imported by default to avoid overhead)
# from ragicamp.core import get_logger, RAGiCampError, AgentType, etc.

__all__ = [
    "ComponentFactory",
    "ComponentRegistry",
    "__version__",
]
