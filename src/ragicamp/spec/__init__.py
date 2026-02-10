"""Experiment specification package.

This package provides immutable experiment configuration objects:
- ExperimentSpec: Immutable configuration for an experiment
- build_specs: Build specs from YAML config
- name_direct, name_rag: Naming conventions
"""

from ragicamp.spec.builder import build_specs
from ragicamp.spec.experiment import ExperimentSpec
from ragicamp.spec.naming import _model_short, _spec_hash, name_direct, name_rag

__all__ = [
    "ExperimentSpec",
    "build_specs",
    "name_direct",
    "name_rag",
    "_model_short",
    "_spec_hash",
]
