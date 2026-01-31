"""Backward compatibility shim for experiment state.

This module re-exports all state-related types from the new location.
Canonical imports should use: from ragicamp.state import ...
"""

# Re-export everything from new location for backward compatibility
from ragicamp.state import (
    ExperimentHealth,
    ExperimentPhase,
    ExperimentState,
    PHASE_ORDER,
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
