"""Execution module for resilient agent execution.

This module provides abstractions for safely executing agent calls with:
- Automatic batch size reduction on CUDA/OOM errors
- GPU memory management
- Retry logic with exponential backoff

Example:
    from ragicamp.execution import ResilientExecutor

    executor = ResilientExecutor(agent, batch_size=32, min_batch_size=1)
    results = executor.execute(queries)
"""

from ragicamp.execution.executor import (
    BatchResult,
    ExecutionConfig,
    ResilientExecutor,
)

__all__ = [
    "ResilientExecutor",
    "ExecutionConfig",
    "BatchResult",
]
