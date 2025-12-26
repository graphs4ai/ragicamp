"""Utility modules for RAGiCamp."""

from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.paths import ensure_dir, ensure_output_dirs, safe_write_json
from ragicamp.utils.prompts import PromptBuilder
from ragicamp.utils.resource_manager import ResourceManager, gpu_memory_scope, managed_model

# Import new utilities (with graceful fallback)
try:
    from ragicamp.utils.experiment_state import (
        ExperimentState,
        PhaseState,
        PhaseStatus,
    )

    _has_state = True
except ImportError:
    _has_state = False

try:
    from ragicamp.utils.mlflow_utils import MLflowTracker, create_mlflow_tracker

    _has_mlflow_utils = True
except ImportError:
    _has_mlflow_utils = False

__all__ = [
    "ContextFormatter",
    "PromptBuilder",
    "ensure_dir",
    "ensure_output_dirs",
    "safe_write_json",
    "ResourceManager",
    "gpu_memory_scope",
    "managed_model",
]

# Add optional exports
if _has_state:
    __all__.extend(["ExperimentState", "PhaseState", "PhaseStatus"])
if _has_mlflow_utils:
    __all__.extend(["MLflowTracker", "create_mlflow_tracker"])
