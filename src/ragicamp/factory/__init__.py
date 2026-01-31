"""Factory package for creating RAGiCamp components from configuration.

This package provides factories for creating:
- Models (HuggingFace, OpenAI, vLLM)
- Datasets (NQ, TriviaQA, HotpotQA, etc.)
- Metrics (Exact Match, F1, BERTScore, etc.)
- Retrievers (Dense, Sparse, Hybrid, Hierarchical)
- Agents (DirectLLM, FixedRAG)

Example:
    from ragicamp.factory import ModelFactory, DatasetFactory

    model = ModelFactory.create({"type": "huggingface", "model_name": "google/gemma-2-2b-it"})
    dataset = DatasetFactory.create({"name": "natural_questions"})
"""

from ragicamp.factory.agents import AgentFactory
from ragicamp.factory.datasets import DatasetFactory
from ragicamp.factory.metrics import MetricFactory
from ragicamp.factory.models import ModelFactory, validate_model_config
from ragicamp.factory.retrievers import (
    RetrieverFactory,
    create_query_transformer,
    create_reranker,
)

# Backward compatibility: ComponentFactory as unified facade
class ComponentFactory:
    """Unified factory facade for backward compatibility.

    This class delegates to the individual specialized factories.
    New code should use the specialized factories directly.
    """

    # Re-export registries from specialized factories
    _custom_models = ModelFactory._custom_models
    _custom_agents = AgentFactory._custom_agents
    _custom_metrics = MetricFactory._custom_metrics
    _custom_retrievers = RetrieverFactory._custom_retrievers

    @classmethod
    def register_model(cls, name: str):
        """Register a custom model type."""
        return ModelFactory.register(name)

    @classmethod
    def register_agent(cls, name: str):
        """Register a custom agent type."""
        return AgentFactory.register(name)

    @classmethod
    def register_metric(cls, name: str):
        """Register a custom metric type."""
        return MetricFactory.register(name)

    @classmethod
    def register_retriever(cls, name: str):
        """Register a custom retriever type."""
        return RetrieverFactory.register(name)

    @staticmethod
    def parse_model_spec(spec, quantization="none", **kwargs):
        """Parse model spec string."""
        return ModelFactory.parse_spec(spec, quantization, **kwargs)

    @staticmethod
    def parse_dataset_spec(name, split="validation", limit=None, **kwargs):
        """Parse dataset spec."""
        return DatasetFactory.parse_spec(name, split, limit, **kwargs)

    @classmethod
    def create_model(cls, config):
        """Create a language model."""
        return ModelFactory.create(config)

    @classmethod
    def create_agent(cls, config, model, retriever=None, **kwargs):
        """Create an agent."""
        return AgentFactory.create(config, model, retriever, **kwargs)

    @staticmethod
    def create_dataset(config):
        """Create a dataset."""
        return DatasetFactory.create(config)

    @classmethod
    def create_metrics(cls, config, judge_model=None):
        """Create metrics."""
        return MetricFactory.create(config, judge_model)

    @staticmethod
    def create_retriever(config):
        """Create a retriever."""
        return RetrieverFactory.create(config)


# Convenience function for loading retrievers
def load_retriever(retriever_name: str):
    """Load a retriever by name."""
    return RetrieverFactory.load(retriever_name)


__all__ = [
    # New individual factories
    "ModelFactory",
    "DatasetFactory",
    "MetricFactory",
    "RetrieverFactory",
    "AgentFactory",
    # Backward compatibility
    "ComponentFactory",
    # Utility functions
    "validate_model_config",
    "load_retriever",
    "create_query_transformer",
    "create_reranker",
]
