"""Agent factory for creating RAG agents from configuration."""

from typing import Any, Dict, Optional

from ragicamp.agents import DirectLLMAgent, FixedRAGAgent, RAGAgent
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Retriever


class AgentFactory:
    """Factory for creating RAG agents from configuration."""

    # Custom agent registry
    _custom_agents: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Register a custom agent type."""
        def decorator(agent_class: type) -> type:
            cls._custom_agents[name] = agent_class
            return agent_class
        return decorator

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model: LanguageModel,
        retriever: Optional[Retriever] = None,
        **kwargs: Any,
    ) -> RAGAgent:
        """Create an agent from configuration.

        Args:
            config: Agent configuration dict with 'type' and agent-specific params
            model: Language model instance
            retriever: Optional retriever for RAG agents
            **kwargs: Additional arguments

        Returns:
            Instantiated RAGAgent

        Example:
            >>> config = {"type": "direct_llm", "name": "baseline"}
            >>> agent = AgentFactory.create(config, model)
        """
        agent_type = config["type"]
        config_copy = dict(config)
        config_copy.pop("type", None)

        # Check custom registry first
        if agent_type in cls._custom_agents:
            return cls._custom_agents[agent_type](model=model, retriever=retriever, **config_copy)

        # Built-in agents
        if agent_type == "direct_llm":
            return DirectLLMAgent(model=model, **config_copy)

        elif agent_type == "fixed_rag":
            if retriever is None:
                raise ValueError("fixed_rag agent requires a retriever")
            return FixedRAGAgent(model=model, retriever=retriever, **config_copy)

        else:
            available = ["direct_llm", "fixed_rag"] + list(cls._custom_agents.keys())
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {', '.join(available)}"
            )
