"""Bandit-based RAG agent - Adaptive parameter selection using bandit algorithms."""

from typing import Any, Dict, List

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.models.base import LanguageModel
from ragicamp.policies.base import Policy
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder


class BanditRAGAgent(RAGAgent):
    """RAG agent that uses bandit policies to adaptively select parameters.

    This agent uses bandit algorithms to dynamically choose:
    - Number of documents to retrieve (top_k)
    - Retrieval strategy
    - Context formatting approach
    - Other RAG hyperparameters

    The policy learns which parameter configurations work best over time.
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        policy: Policy,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
        **kwargs: Any,
    ):
        """Initialize the bandit-based RAG agent.

        Args:
            name: Agent identifier
            model: The language model to use
            retriever: The retriever for finding relevant documents
            policy: Bandit policy for parameter selection
            system_prompt: System prompt for the LLM
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.policy = policy
        self.prompt_builder = PromptBuilder(system_prompt=system_prompt)

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using bandit-selected parameters.

        Args:
            query: The input question
            **kwargs: Additional generation parameters

        Returns:
            RAGResponse with the answer and selected parameters
        """
        # Use policy to select RAG parameters
        rag_params = self.policy.select_action(query=query)

        # Retrieve documents with selected parameters
        top_k = rag_params.get("top_k", 5)
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k, **rag_params)

        # Create context
        context = RAGContext(
            query=query, retrieved_docs=retrieved_docs, metadata={"selected_params": rag_params}
        )

        # Format context using utility
        context_text = ContextFormatter.format_numbered(retrieved_docs)

        # Build prompt using utility
        prompt = self.prompt_builder.build_prompt(query=query, context=context_text)
        answer = self.model.generate(prompt, **kwargs)

        # Return response with selected parameters
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={
                "agent_type": "bandit_rag",
                "selected_params": rag_params,
                "num_docs_used": len(retrieved_docs),
            },
        )

    def update_policy(self, query: str, params: Dict[str, Any], reward: float) -> None:
        """Update the bandit policy based on observed reward.

        Args:
            query: The query that was answered
            params: The parameters that were selected
            reward: The reward/score achieved
        """
        self.policy.update(query=query, action=params, reward=reward)
