"""Tests for module imports.

These tests ensure all key modules can be imported without errors.
This catches issues like missing re-exports or incorrect import paths.
"""

import pytest


class TestCLIImports:
    """Test that CLI modules import correctly."""

    def test_import_cli_main(self):
        """Test importing cli.main module."""
        from ragicamp.cli.main import main
        assert callable(main)

    def test_import_cli_study(self):
        """Test importing cli.study module."""
        from ragicamp.cli import study
        assert hasattr(study, 'run_study')

    def test_cli_study_has_build_specs(self):
        """Test that cli.study exports build_specs."""
        from ragicamp.cli.study import build_specs
        assert callable(build_specs)

    def test_cli_study_has_run_spec(self):
        """Test that cli.study exports run_spec."""
        from ragicamp.cli.study import run_spec
        assert callable(run_spec)


class TestModelImports:
    """Test that model modules import correctly."""

    def test_import_huggingface_model(self):
        """Test importing HuggingFaceModel."""
        from ragicamp.models import HuggingFaceModel
        assert HuggingFaceModel is not None

    def test_import_vllm_model(self):
        """Test importing VLLMModel."""
        from ragicamp.models import VLLMModel
        assert VLLMModel is not None

    def test_import_openai_model(self):
        """Test importing OpenAIModel."""
        from ragicamp.models import OpenAIModel
        assert OpenAIModel is not None


class TestIndexImports:
    """Test that index modules import correctly."""

    def test_import_embedding_index(self):
        """Test importing EmbeddingIndex."""
        from ragicamp.indexes import EmbeddingIndex
        assert EmbeddingIndex is not None

    def test_import_hierarchical_index(self):
        """Test importing HierarchicalIndex."""
        from ragicamp.indexes import HierarchicalIndex
        assert HierarchicalIndex is not None


class TestFactoryImports:
    """Test that factory modules import correctly."""

    def test_import_model_factory(self):
        """Test importing ModelFactory."""
        from ragicamp.factory import ModelFactory
        assert ModelFactory is not None

    def test_import_agent_factory(self):
        """Test importing AgentFactory."""
        from ragicamp.factory import AgentFactory
        assert AgentFactory is not None


class TestSpecImports:
    """Test that spec modules import correctly."""

    def test_import_experiment_spec(self):
        """Test importing ExperimentSpec."""
        from ragicamp.spec import ExperimentSpec
        assert ExperimentSpec is not None

    def test_import_build_specs(self):
        """Test importing build_specs."""
        from ragicamp.spec import build_specs
        assert callable(build_specs)

    def test_import_naming_functions(self):
        """Test importing naming functions."""
        from ragicamp.spec import name_direct, name_rag
        assert callable(name_direct)
        assert callable(name_rag)


class TestExecutionImports:
    """Test that execution modules import correctly."""

    def test_import_run_spec(self):
        """Test importing run_spec."""
        from ragicamp.execution.runner import run_spec
        assert callable(run_spec)


class TestRerankerImports:
    """Test that reranker modules import correctly."""

    def test_import_cross_encoder_reranker(self):
        """Test importing CrossEncoderReranker."""
        from ragicamp.rag.rerankers import CrossEncoderReranker
        assert CrossEncoderReranker is not None
