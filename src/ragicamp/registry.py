"""Component registry for RAGiCamp.

This module provides a registry system for dynamically registering and retrieving
custom components (models, agents, metrics, datasets, retrievers).

Example:
    >>> from ragicamp.registry import ComponentRegistry
    >>> from ragicamp.models.base import LanguageModel
    >>>
    >>> @ComponentRegistry.register_model("my_custom_model")
    >>> class MyCustomModel(LanguageModel):
    ...     def generate(self, prompt, **kwargs):
    ...         return "custom response"
    >>>
    >>> # Now it can be used in configs:
    >>> # model:
    >>> #   type: my_custom_model
    >>> #   param1: value1
"""

from typing import Any, Callable, Dict, List, Type


class ComponentRegistry:
    """Registry for dynamically registering and retrieving components.
    
    Allows users to register custom components that can then be used
    in YAML configurations without modifying core code.
    """

    _models: Dict[str, Type] = {}
    _agents: Dict[str, Type] = {}
    _metrics: Dict[str, Type] = {}
    _datasets: Dict[str, Type] = {}
    _retrievers: Dict[str, Type] = {}

    # ========== Models ==========

    @classmethod
    def register_model(cls, name: str) -> Callable:
        """Decorator to register a model class.

        Args:
            name: Unique identifier for the model

        Returns:
            Decorator function

        Example:
            >>> @ComponentRegistry.register_model("gpt4")
            >>> class GPT4Model(LanguageModel):
            ...     pass
        """

        def decorator(model_class: Type) -> Type:
            if name in cls._models:
                print(
                    f"Warning: Overwriting existing model registration: {name}"
                )
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get_model(cls, name: str) -> Type:
        """Get a registered model class.

        Args:
            name: Model identifier

        Returns:
            Model class

        Raises:
            ValueError: If model not found
        """
        if name not in cls._models:
            available = ", ".join(sorted(cls._models.keys()))
            raise ValueError(
                f"Unknown model: {name}. Available: {available or 'none'}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return sorted(cls._models.keys())

    # ========== Agents ==========

    @classmethod
    def register_agent(cls, name: str) -> Callable:
        """Decorator to register an agent class.

        Args:
            name: Unique identifier for the agent

        Returns:
            Decorator function

        Example:
            >>> @ComponentRegistry.register_agent("my_rag")
            >>> class MyRAGAgent(RAGAgent):
            ...     pass
        """

        def decorator(agent_class: Type) -> Type:
            if name in cls._agents:
                print(
                    f"Warning: Overwriting existing agent registration: {name}"
                )
            cls._agents[name] = agent_class
            return agent_class

        return decorator

    @classmethod
    def get_agent(cls, name: str) -> Type:
        """Get a registered agent class.

        Args:
            name: Agent identifier

        Returns:
            Agent class

        Raises:
            ValueError: If agent not found
        """
        if name not in cls._agents:
            available = ", ".join(sorted(cls._agents.keys()))
            raise ValueError(
                f"Unknown agent: {name}. Available: {available or 'none'}"
            )
        return cls._agents[name]

    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent names."""
        return sorted(cls._agents.keys())

    # ========== Metrics ==========

    @classmethod
    def register_metric(cls, name: str) -> Callable:
        """Decorator to register a metric class.

        Args:
            name: Unique identifier for the metric

        Returns:
            Decorator function

        Example:
            >>> @ComponentRegistry.register_metric("my_metric")
            >>> class MyMetric(Metric):
            ...     pass
        """

        def decorator(metric_class: Type) -> Type:
            if name in cls._metrics:
                print(
                    f"Warning: Overwriting existing metric registration: {name}"
                )
            cls._metrics[name] = metric_class
            return metric_class

        return decorator

    @classmethod
    def get_metric(cls, name: str) -> Type:
        """Get a registered metric class.

        Args:
            name: Metric identifier

        Returns:
            Metric class

        Raises:
            ValueError: If metric not found
        """
        if name not in cls._metrics:
            available = ", ".join(sorted(cls._metrics.keys()))
            raise ValueError(
                f"Unknown metric: {name}. Available: {available or 'none'}"
            )
        return cls._metrics[name]

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metric names."""
        return sorted(cls._metrics.keys())

    # ========== Datasets ==========

    @classmethod
    def register_dataset(cls, name: str) -> Callable:
        """Decorator to register a dataset class.

        Args:
            name: Unique identifier for the dataset

        Returns:
            Decorator function

        Example:
            >>> @ComponentRegistry.register_dataset("my_dataset")
            >>> class MyDataset(QADataset):
            ...     pass
        """

        def decorator(dataset_class: Type) -> Type:
            if name in cls._datasets:
                print(
                    f"Warning: Overwriting existing dataset registration: {name}"
                )
            cls._datasets[name] = dataset_class
            return dataset_class

        return decorator

    @classmethod
    def get_dataset(cls, name: str) -> Type:
        """Get a registered dataset class.

        Args:
            name: Dataset identifier

        Returns:
            Dataset class

        Raises:
            ValueError: If dataset not found
        """
        if name not in cls._datasets:
            available = ", ".join(sorted(cls._datasets.keys()))
            raise ValueError(
                f"Unknown dataset: {name}. Available: {available or 'none'}"
            )
        return cls._datasets[name]

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return sorted(cls._datasets.keys())

    # ========== Retrievers ==========

    @classmethod
    def register_retriever(cls, name: str) -> Callable:
        """Decorator to register a retriever class.

        Args:
            name: Unique identifier for the retriever

        Returns:
            Decorator function

        Example:
            >>> @ComponentRegistry.register_retriever("my_retriever")
            >>> class MyRetriever(Retriever):
            ...     pass
        """

        def decorator(retriever_class: Type) -> Type:
            if name in cls._retrievers:
                print(
                    f"Warning: Overwriting existing retriever registration: {name}"
                )
            cls._retrievers[name] = retriever_class
            return retriever_class

        return decorator

    @classmethod
    def get_retriever(cls, name: str) -> Type:
        """Get a registered retriever class.

        Args:
            name: Retriever identifier

        Returns:
            Retriever class

        Raises:
            ValueError: If retriever not found
        """
        if name not in cls._retrievers:
            available = ", ".join(sorted(cls._retrievers.keys()))
            raise ValueError(
                f"Unknown retriever: {name}. Available: {available or 'none'}"
            )
        return cls._retrievers[name]

    @classmethod
    def list_retrievers(cls) -> List[str]:
        """List all registered retriever names."""
        return sorted(cls._retrievers.keys())

    # ========== Utilities ==========

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._models.clear()
        cls._agents.clear()
        cls._metrics.clear()
        cls._datasets.clear()
        cls._retrievers.clear()

    @classmethod
    def summary(cls) -> Dict[str, List[str]]:
        """Get a summary of all registered components.

        Returns:
            Dict mapping component types to lists of registered names
        """
        return {
            "models": cls.list_models(),
            "agents": cls.list_agents(),
            "metrics": cls.list_metrics(),
            "datasets": cls.list_datasets(),
            "retrievers": cls.list_retrievers(),
        }


# Register built-in components on module import
def _register_builtin_components():
    """Register all built-in components with the registry."""
    from ragicamp.agents import (
        DirectLLMAgent,
        FixedRAGAgent,
        BanditRAGAgent,
        MDPRAGAgent,
    )
    from ragicamp.datasets import (
        NaturalQuestionsDataset,
        TriviaQADataset,
        HotpotQADataset,
    )
    from ragicamp.models import HuggingFaceModel, OpenAIModel
    from ragicamp.retrievers import DenseRetriever, SparseRetriever

    # Register models
    ComponentRegistry.register_model("huggingface")(HuggingFaceModel)
    ComponentRegistry.register_model("openai")(OpenAIModel)

    # Register agents
    ComponentRegistry.register_agent("direct_llm")(DirectLLMAgent)
    ComponentRegistry.register_agent("fixed_rag")(FixedRAGAgent)
    ComponentRegistry.register_agent("bandit_rag")(BanditRAGAgent)
    ComponentRegistry.register_agent("mdp_rag")(MDPRAGAgent)

    # Register datasets
    ComponentRegistry.register_dataset("natural_questions")(
        NaturalQuestionsDataset
    )
    ComponentRegistry.register_dataset("triviaqa")(TriviaQADataset)
    ComponentRegistry.register_dataset("hotpotqa")(HotpotQADataset)

    # Register retrievers
    ComponentRegistry.register_retriever("dense")(DenseRetriever)
    ComponentRegistry.register_retriever("sparse")(SparseRetriever)

    # Metrics are handled differently (conditional imports in factory)


# Auto-register built-in components when module is imported
_register_builtin_components()

