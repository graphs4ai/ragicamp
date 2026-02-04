"""Core infrastructure for RAGiCamp.

This module provides foundational components:
- Schemas: Data contracts (PredictionRecord, RetrievedDoc, etc.)
- Exceptions: Custom exception hierarchy
- Logging: Structured logging configuration
- Protocols: Type-checkable interfaces
- Constants: Enums and constants
"""

from ragicamp.core.constants import (
    AgentType,
    EvaluationMode,
    MetricType,
    ModelType,
    RetrieverType,
)
from ragicamp.core.exceptions import (
    ConfigError,
    ConfigurationError,
    EvaluationError,
    ModelError,
    RAGiCampError,
)
from ragicamp.core.logging import configure_logging, get_logger
from ragicamp.core.schemas import (
    ExperimentSpec,
    PipelineLog,
    PredictionRecord,
    PromptStyle,
    QueryTransformStep,
    RAGResponseMeta,
    RerankStep,
    RetrievalStep,
    RetrievedDoc,
)

__all__ = [
    # Schemas (data contracts)
    "PredictionRecord",
    "RetrievedDoc",
    "RAGResponseMeta",
    "ExperimentSpec",
    "PromptStyle",
    # Pipeline logging (modular metadata)
    "PipelineLog",
    "QueryTransformStep",
    "RetrievalStep",
    "RerankStep",
    # Exceptions
    "RAGiCampError",
    "ConfigError",
    "ConfigurationError",  # Alias for ConfigError
    "ModelError",
    "EvaluationError",
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
