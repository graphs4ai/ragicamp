"""Fixed RAG agent - Baseline 2: Standard RAG with fixed parameters."""

from typing import Any, List, Optional

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.utils.artifacts import get_artifact_manager
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder


class FixedRAGAgent(RAGAgent):
    """Baseline RAG agent with fixed retrieval parameters.

    This agent implements the standard RAG pipeline:
    1. Retrieve top-k documents
    2. Format context with retrieved documents
    3. Generate answer using LLM with context
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        top_k: int = 5,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
        context_template: str = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        **kwargs: Any,
    ):
        """Initialize the fixed RAG agent.

        Args:
            name: Agent identifier
            model: The language model to use
            retriever: The retriever for finding relevant documents
            top_k: Number of documents to retrieve
            system_prompt: System prompt for the LLM
            context_template: Template for formatting retrieved context
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.prompt_builder = PromptBuilder(
            system_prompt=system_prompt, context_template=context_template
        )

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using fixed RAG pipeline.

        Args:
            query: The input question
            **kwargs: Additional generation parameters

        Returns:
            RAGResponse with the answer and retrieved context
        """
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)

        # Format context using utility
        context_text = ContextFormatter.format_numbered(retrieved_docs)

        # Build prompt using utility
        prompt = self.prompt_builder.build_prompt(query=query, context=context_text)

        # Create context with prompt included
        context = RAGContext(
            query=query,
            retrieved_docs=retrieved_docs,
            metadata={
                "top_k": self.top_k,
                "prompt": prompt,  # Store the actual prompt used
                "context_text": context_text,  # Store formatted context
            },
        )

        # Generate answer
        answer = self.model.generate(prompt, **kwargs)

        # Return response
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={"agent_type": "fixed_rag", "num_docs_used": len(retrieved_docs)},
        )

    def save(self, artifact_name: str, retriever_artifact_name: str) -> str:
        """Save agent configuration.

        Args:
            artifact_name: Name for this agent artifact (e.g., 'fixed_rag_wikipedia_v1')
            retriever_artifact_name: Name of the retriever artifact this agent uses

        Returns:
            Path where the artifact was saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_agent_path(artifact_name)

        # Save agent config
        config = {
            "agent_type": "fixed_rag",
            "name": self.name,
            "top_k": self.top_k,
            "retriever_artifact": retriever_artifact_name,
            "system_prompt": self.prompt_builder.system_prompt,
            "context_template": self.prompt_builder.context_template,
        }
        manager.save_json(config, artifact_path / "config.json")

        print(f"✓ Saved agent to: {artifact_path}")
        return str(artifact_path)

    @classmethod
    def load(
        cls, artifact_name: str, model: LanguageModel, load_retriever: bool = True
    ) -> "FixedRAGAgent":
        """Load a previously saved agent.

        Args:
            artifact_name: Name of the agent artifact to load
            model: Language model to use (not saved, must provide)
            load_retriever: Whether to load the associated retriever

        Returns:
            Loaded FixedRAGAgent instance
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_agent_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Load retriever if requested
        retriever = None
        if load_retriever:
            retriever_name = config["retriever_artifact"]
            retriever = DenseRetriever.load_index(retriever_name)

        # Create agent
        agent = cls(
            name=config["name"],
            model=model,
            retriever=retriever,
            top_k=config["top_k"],
            system_prompt=config["system_prompt"],
            context_template=config["context_template"],
        )

        print(f"✓ Loaded agent from: {artifact_path}")
        return agent
