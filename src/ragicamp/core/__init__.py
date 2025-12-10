"""Core infrastructure for RAGiCamp.

This module provides foundational components:
- Exceptions: Custom exception hierarchy
- Logging: Structured logging configuration
- Protocols: Type-checkable interfaces
- Constants: Enums and constants
"""

from ragicamp.core.exceptions import (
    RAGiCampError,
    ConfigurationError,
    ComponentNotFoundError,
    EvaluationError,
    CheckpointError,
    RetrieverError,
    ModelError,
    MetricError,
    DatasetError,
)

from ragicamp.core.logging import get_logger, configure_logging

from ragicamp.core.constants import (
    AgentType,
    MetricType,
    ModelType,
    RetrieverType,
    EvaluationMode,
)

__all__ = [
    # Exceptions
    "RAGiCampError",
    "ConfigurationError",
    "ComponentNotFoundError",
    "EvaluationError",
    "CheckpointError",
    "RetrieverError",
    "ModelError",
    "MetricError",
    "DatasetError",
    # Logging
    "get_logger",
    "configure_logging",
    # Constants
    "AgentType",
    "MetricType",
    "ModelType",
    "RetrieverType",
    "EvaluationMode",
]
