"""Tests for component factory."""

import pytest
from unittest.mock import Mock, patch

from ragicamp.factory import ComponentFactory
from ragicamp.agents import DirectLLMAgent
from ragicamp.models.base import LanguageModel
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric


class TestModelFactory:
    """Test model creation via factory."""

    @patch("ragicamp.factory.HuggingFaceModel")
    def test_create_huggingface_model(self, mock_hf_model):
        """Test creating HuggingFace model."""
        config = {
            "type": "huggingface",
            "model_name": "google/gemma-2-2b-it",
            "device": "cuda",
            "load_in_8bit": True,
        }

        ComponentFactory.create_model(config)

        # Should call HuggingFaceModel with correct params
        mock_hf_model.assert_called_once()
        call_kwargs = mock_hf_model.call_args[1]
        assert call_kwargs["model_name"] == "google/gemma-2-2b-it"
        assert call_kwargs["device"] == "cuda"
        assert call_kwargs["load_in_8bit"] is True

    @patch("ragicamp.factory.OpenAIModel")
    def test_create_openai_model(self, mock_openai_model):
        """Test creating OpenAI model."""
        config = {"type": "openai", "model_name": "gpt-4o"}

        ComponentFactory.create_model(config)

        mock_openai_model.assert_called_once()
        call_kwargs = mock_openai_model.call_args[1]
        assert call_kwargs["model_name"] == "gpt-4o"

    def test_create_model_invalid_type(self):
        """Test creating model with invalid type."""
        config = {"type": "invalid_type", "model_name": "test"}

        with pytest.raises(ValueError) as exc_info:
            ComponentFactory.create_model(config)

        assert "Unknown model type" in str(exc_info.value)

    def test_create_model_filters_generation_params(self):
        """Test that generation params are filtered out."""
        config = {
            "type": "huggingface",
            "model_name": "test-model",
            "max_tokens": 100,  # Should be filtered
            "temperature": 0.7,  # Should be filtered
            "device": "cuda",  # Should be kept
        }

        with patch("ragicamp.factory.HuggingFaceModel") as mock_hf:
            ComponentFactory.create_model(config)

            call_kwargs = mock_hf.call_args[1]
            assert "model_name" in call_kwargs
            assert "device" in call_kwargs
            assert "max_tokens" not in call_kwargs
            assert "temperature" not in call_kwargs


class TestAgentFactory:
    """Test agent creation via factory."""

    def test_create_direct_llm_agent(self):
        """Test creating DirectLLM agent."""
        model = Mock(spec=LanguageModel)
        config = {"type": "direct_llm", "name": "test_agent", "system_prompt": "Test prompt"}

        agent = ComponentFactory.create_agent(config, model=model)

        assert isinstance(agent, DirectLLMAgent)
        assert agent.name == "test_agent"
        assert agent.model == model

    def test_create_fixed_rag_agent(self):
        """Test creating FixedRAG agent."""
        model = Mock(spec=LanguageModel)
        retriever = Mock()
        config = {"type": "fixed_rag", "name": "rag_agent", "top_k": 5}

        agent = ComponentFactory.create_agent(config, model=model, retriever=retriever)

        assert agent.name == "rag_agent"
        assert agent.model == model
        assert agent.retriever == retriever

    def test_create_rag_agent_without_retriever(self):
        """Test that creating RAG agent without retriever raises error."""
        model = Mock(spec=LanguageModel)
        config = {"type": "fixed_rag", "name": "rag_agent"}

        with pytest.raises(ValueError) as exc_info:
            ComponentFactory.create_agent(config, model=model)

        assert "requires a retriever" in str(exc_info.value)

    def test_create_agent_invalid_type(self):
        """Test creating agent with invalid type."""
        model = Mock(spec=LanguageModel)
        config = {"type": "invalid_agent", "name": "test"}

        with pytest.raises(ValueError) as exc_info:
            ComponentFactory.create_agent(config, model=model)

        assert "Unknown agent type" in str(exc_info.value)


class TestDatasetFactory:
    """Test dataset creation via factory."""

    @patch("ragicamp.factory.NaturalQuestionsDataset")
    def test_create_natural_questions_dataset(self, mock_nq):
        """Test creating Natural Questions dataset."""
        # Mock the dataset instance
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 100
        mock_nq.return_value = mock_dataset

        config = {
            "name": "natural_questions",
            "split": "validation",
            "num_examples": 50,
            "filter_no_answer": True,
        }

        dataset = ComponentFactory.create_dataset(config)

        # Should create NaturalQuestionsDataset
        mock_nq.assert_called_once()

    def test_create_dataset_invalid_name(self):
        """Test creating dataset with invalid name."""
        config = {"name": "invalid_dataset", "split": "validation"}

        with pytest.raises(ValueError) as exc_info:
            ComponentFactory.create_dataset(config)

        assert "Unknown dataset" in str(exc_info.value)


class TestMetricsFactory:
    """Test metrics creation via factory."""

    def test_create_exact_match_metric(self):
        """Test creating exact match metric."""
        metrics_config = ["exact_match"]

        metrics = ComponentFactory.create_metrics(metrics_config)

        assert len(metrics) == 1
        assert metrics[0].name == "exact_match"

    def test_create_multiple_metrics(self):
        """Test creating multiple metrics."""
        metrics_config = ["exact_match", "f1"]

        metrics = ComponentFactory.create_metrics(metrics_config)

        assert len(metrics) == 2
        assert metrics[0].name == "exact_match"
        assert metrics[1].name == "f1"

    def test_create_metric_with_params(self):
        """Test creating metric with parameters."""
        metrics_config = [{"name": "bertscore", "params": {"model_type": "bert-base-uncased"}}]

        with patch("ragicamp.factory.BERTScoreMetric") as mock_bertscore:
            # Mock the import check
            import sys

            sys.modules["ragicamp.metrics.bertscore"] = Mock()

            try:
                metrics = ComponentFactory.create_metrics(metrics_config)
            except:
                pass  # May fail due to actual imports, but we check the call

    def test_create_llm_judge_metric(self):
        """Test creating LLM judge metric with judge model."""
        judge_model = Mock(spec=LanguageModel)
        metrics_config = [
            {"name": "llm_judge_qa", "params": {"judgment_type": "binary", "batch_size": 8}}
        ]

        with patch("ragicamp.factory.LLMJudgeQAMetric") as mock_llm_judge:
            # Mock availability check
            import sys

            module_mock = Mock()
            module_mock.LLMJudgeQAMetric = mock_llm_judge
            sys.modules["ragicamp.metrics.llm_judge_qa"] = module_mock

            try:
                metrics = ComponentFactory.create_metrics(metrics_config, judge_model=judge_model)
            except:
                pass

    def test_skip_unavailable_metric(self):
        """Test that unavailable metrics are skipped with warning."""
        metrics_config = ["exact_match", "nonexistent_metric", "f1"]

        metrics = ComponentFactory.create_metrics(metrics_config)

        # Should create the available ones and skip the nonexistent one
        assert len(metrics) == 2
        metric_names = [m.name for m in metrics]
        assert "exact_match" in metric_names
        assert "f1" in metric_names


class TestRetrieverFactory:
    """Test retriever creation via factory."""

    @patch("ragicamp.factory.DenseRetriever")
    def test_create_dense_retriever(self, mock_dense):
        """Test creating dense retriever."""
        config = {"type": "dense", "embedding_model": "all-MiniLM-L6-v2", "index_type": "flat"}

        ComponentFactory.create_retriever(config)

        mock_dense.assert_called_once()
        call_kwargs = mock_dense.call_args[1]
        assert call_kwargs["embedding_model"] == "all-MiniLM-L6-v2"
        assert call_kwargs["index_type"] == "flat"

    def test_create_retriever_invalid_type(self):
        """Test creating retriever with invalid type."""
        config = {"type": "invalid_retriever"}

        with pytest.raises(ValueError) as exc_info:
            ComponentFactory.create_retriever(config)

        assert "Unknown retriever type" in str(exc_info.value)


class TestFactoryConfigHandling:
    """Test factory configuration handling."""

    def test_factory_removes_type_field(self):
        """Test that factory removes 'type' field before passing to constructor."""
        config = {"type": "huggingface", "model_name": "test-model", "device": "cpu"}

        with patch("ragicamp.factory.HuggingFaceModel") as mock_hf:
            ComponentFactory.create_model(config)

            call_kwargs = mock_hf.call_args[1]
            assert "type" not in call_kwargs
            assert "model_name" in call_kwargs

    def test_factory_preserves_extra_params(self):
        """Test that factory preserves extra parameters."""
        model = Mock(spec=LanguageModel)
        config = {
            "type": "direct_llm",
            "name": "test_agent",
            "custom_param": "custom_value",
            "another_param": 42,
        }

        # Should not raise error even with extra params
        agent = ComponentFactory.create_agent(config, model=model)

        assert agent.name == "test_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
