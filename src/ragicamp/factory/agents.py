"""Agent factory for creating RAG agents from configuration.

This module provides two ways to create agents:

1. AgentFactory.create(config, model, retriever) - Low-level, dict-based config
2. AgentFactory.from_spec(spec, model, retriever) - High-level, from ExperimentSpec

The from_spec() method is preferred as it handles all the wiring of query
transformers, rerankers, and prompt builders automatically.
"""

from typing import TYPE_CHECKING, Any, Optional

from ragicamp.agents import DirectLLMAgent, FixedRAGAgent, RAGAgent
from ragicamp.core.logging import get_logger
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Retriever

if TYPE_CHECKING:
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


class AgentFactory:
    """Factory for creating RAG agents from configuration.

    Supports two creation patterns:

    1. Dict-based (low-level):
        >>> config = {"type": "direct_llm", "name": "baseline"}
        >>> agent = AgentFactory.create(config, model)

    2. Spec-based (high-level, preferred):
        >>> agent = AgentFactory.from_spec(spec, model)

    Custom agents can be registered using the @register decorator:
        >>> @AgentFactory.register("my_agent")
        ... class MyAgent(RAGAgent):
        ...     pass
    """

    # Custom agent registry
    _custom_agents: dict[str, type] = {}

    # Agent type aliases for backwards compatibility and clarity
    _aliases: dict[str, str] = {
        "pipeline_rag": "fixed_rag",  # Clearer name for fixed_rag
        "direct": "direct_llm",  # Short form
    }

    @classmethod
    def register(cls, name: str):
        """Register a custom agent type.

        Example:
            >>> @AgentFactory.register("iterative_rag")
            ... class IterativeRAGAgent(RAGAgent):
            ...     def __init__(self, name, model, retriever, max_iterations=2, **kwargs):
            ...         ...
        """

        def decorator(agent_class: type) -> type:
            cls._custom_agents[name] = agent_class
            logger.debug("Registered agent type: %s -> %s", name, agent_class.__name__)
            return agent_class

        return decorator

    @classmethod
    def from_spec(
        cls,
        spec: "ExperimentSpec",
        model: LanguageModel,
        retriever: Optional[Retriever] = None,
    ) -> RAGAgent:
        """Create an agent from an ExperimentSpec.

        This is the preferred method for creating agents as it handles all
        the wiring of optional components (query transformers, rerankers,
        prompt builders) automatically based on the spec.

        Args:
            spec: Experiment specification containing all config
            model: Language model instance
            retriever: Optional retriever (will be loaded from spec.retriever if not provided)

        Returns:
            Fully configured RAGAgent

        Example:
            >>> from ragicamp.spec import ExperimentSpec
            >>> spec = ExperimentSpec(name="test", exp_type="rag", ...)
            >>> agent = AgentFactory.from_spec(spec, model)
        """
        from ragicamp.factory.retrievers import create_query_transformer, create_reranker
        from ragicamp.utils.prompts import PromptBuilder

        # Determine agent type: explicit > inferred from exp_type
        agent_type = spec.agent_type
        if not agent_type:
            agent_type = "direct_llm" if spec.exp_type == "direct" else "fixed_rag"

        # Resolve aliases
        agent_type = cls._aliases.get(agent_type, agent_type)

        # Load retriever if needed and not provided
        if retriever is None and spec.retriever:
            from ragicamp.factory.retrievers import RetrieverFactory

            retriever = RetrieverFactory.load(spec.retriever)

        # Build prompt builder
        prompt_builder = PromptBuilder.from_config(spec.prompt, dataset=spec.dataset)

        # Build base kwargs
        kwargs: dict[str, Any] = {
            "name": spec.name,
            "prompt_builder": prompt_builder,
        }

        # Add RAG-specific components
        if agent_type != "direct_llm":
            kwargs["top_k"] = spec.top_k

            # Add fetch_k for reranking pool
            if spec.fetch_k:
                kwargs["top_k_retrieve"] = spec.fetch_k

            # Create query transformer if specified
            if spec.query_transform:
                query_transformer = create_query_transformer(spec.query_transform, model)
                if query_transformer:
                    kwargs["query_transformer"] = query_transformer

            # Create reranker if specified
            if spec.reranker_model:
                reranker = create_reranker(spec.reranker_model)
                if reranker:
                    kwargs["reranker"] = reranker

        # Add agent-specific params from spec
        if spec.agent_params:
            kwargs.update(dict(spec.agent_params))

        # Create the agent
        return cls._create_instance(agent_type, model, retriever, kwargs)

    @classmethod
    def _create_instance(
        cls,
        agent_type: str,
        model: LanguageModel,
        retriever: Optional[Retriever],
        kwargs: dict[str, Any],
    ) -> RAGAgent:
        """Internal method to instantiate an agent by type."""
        # Check custom registry first
        if agent_type in cls._custom_agents:
            agent_cls = cls._custom_agents[agent_type]
            # For direct-style agents, don't pass retriever
            if agent_type in ("direct_llm", "direct"):
                return agent_cls(model=model, **kwargs)
            return agent_cls(model=model, retriever=retriever, **kwargs)

        # Built-in agents
        if agent_type == "direct_llm":
            return DirectLLMAgent(model=model, **kwargs)

        elif agent_type == "fixed_rag":
            if retriever is None:
                raise ValueError("fixed_rag agent requires a retriever")
            return FixedRAGAgent(model=model, retriever=retriever, **kwargs)

        else:
            available = cls.get_available_agents()
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {', '.join(available)}")

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        model: LanguageModel,
        retriever: Optional[Retriever] = None,
        **kwargs: Any,
    ) -> RAGAgent:
        """Create an agent from a configuration dict.

        This is the low-level API. For most use cases, prefer from_spec().

        Args:
            config: Agent configuration dict with 'type' and agent-specific params
            model: Language model instance
            retriever: Optional retriever for RAG agents
            **kwargs: Additional arguments merged into config

        Returns:
            Instantiated RAGAgent

        Example:
            >>> config = {"type": "direct_llm", "name": "baseline"}
            >>> agent = AgentFactory.create(config, model)
        """
        agent_type = config["type"]
        config_copy = dict(config)
        config_copy.pop("type", None)
        config_copy.update(kwargs)

        # Resolve aliases
        agent_type = cls._aliases.get(agent_type, agent_type)

        return cls._create_instance(agent_type, model, retriever, config_copy)

    @classmethod
    def get_available_agents(cls) -> list:
        """Get list of all available agent types."""
        built_in = ["direct_llm", "fixed_rag", "pipeline_rag", "direct"]
        custom = list(cls._custom_agents.keys())
        return built_in + custom
