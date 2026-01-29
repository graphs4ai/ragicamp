"""Execution module for resilient agent execution and experiment running.

This module provides:
- Resilient execution with automatic batch size reduction
- GPU memory management
- Experiment specification and execution

Example:
    from ragicamp.execution import ResilientExecutor, ExpSpec, build_specs

    executor = ResilientExecutor(agent, batch_size=32, min_batch_size=1)
    results = executor.execute(queries)
"""

from ragicamp.execution.executor import (
    BatchResult,
    ExecutionConfig,
    ResilientExecutor,
)
from ragicamp.execution.runner import (
    ExpSpec,
    build_specs,
    run_spec,
    run_spec_subprocess,
    run_generation,
    run_metrics_only,
)

__all__ = [
    # Executor
    "ResilientExecutor",
    "ExecutionConfig",
    "BatchResult",
    # Runner
    "ExpSpec",
    "build_specs",
    "run_spec",
    "run_spec_subprocess",
    "run_generation",
    "run_metrics_only",
]
