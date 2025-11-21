"""RAGiCamp: A modular framework for experimenting with RAG approaches."""

__version__ = "0.1.0"

from ragicamp.factory import ComponentFactory
from ragicamp.registry import ComponentRegistry

__all__ = ["ComponentFactory", "ComponentRegistry"]
