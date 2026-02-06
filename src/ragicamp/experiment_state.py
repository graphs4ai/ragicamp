"""Backward compatibility shim for experiment state.

DEPRECATED: Import from ``ragicamp.state`` instead.

This module re-exports all state-related types from the new location.
All internal code has been migrated.  This shim exists only for
external callers and will be removed in a future release.
"""

import warnings

warnings.warn(
    "ragicamp.experiment_state is deprecated. Use ragicamp.state instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from new location for backward compatibility
from ragicamp.state import (
    PHASE_ORDER,
    ExperimentHealth,
    ExperimentPhase,
    ExperimentState,
    check_health,
    detect_state,
)

# Backward compatibility alias for private variable
_PHASE_ORDER = PHASE_ORDER

__all__ = [
    "ExperimentPhase",
    "ExperimentState",
    "ExperimentHealth",
    "check_health",
    "detect_state",
    "_PHASE_ORDER",
]
