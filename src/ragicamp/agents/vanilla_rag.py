"""Vanilla RAG agent - Simplest possible RAG baseline.

This agent implements the most basic RAG pipeline:
1. Retrieve top-k documents
2. Format context
3. Generate answer

No query transformation, no reranking - just retrieve and generate.
Use this for clean baseline comparisons.
"""

from typing import TYPE_CHECKING, Any, Optional

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.core.schemas import AgentType, RAGResponseMeta, RetrievedDoc
from ragicamp.factory.agents import AgentFactory
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Retriever
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder

if TYPE_CHECKING:
    pass


@AgentFactory.register("vanilla_rag")
class VanillaRAGAgent(RAGAgent):
    """Simplest RAG agent: retrieve → generate.

    This agent provides a clean baseline for RAG experiments:
    - No query transformation (uses raw query)
    - No reranking (uses retriever ordering)
    - Single retrieval step

    Use this when you want:
    - Clean baseline without any enhancements
    - Fast inference (single retrieval, no extra LLM calls)
    - Minimal complexity for debugging

    Config:
        agent_type: vanilla_rag
        top_k: 5
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        top_k: int = 5,
        prompt_builder: Optional[PromptBuilder] = None,
        **kwargs: Any,
    ):
        """Initialize the vanilla RAG agent.

        Args:
            name: Agent identifier
            model: The language model to use for generation
            retriever: The retriever for finding relevant documents
            top_k: Number of documents to retrieve
            prompt_builder: Optional PromptBuilder for customizing prompts
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.top_k = top_k

        # Use provided prompt_builder or create default
        if prompt_builder is not None:
            self.prompt_builder = prompt_builder
        else:
            self.prompt_builder = PromptBuilder()

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using simple retrieve → generate flow.

        Args:
            query: The input question
            **kwargs: Additional generation parameters

        Returns:
            RAGResponse with the answer and retrieved context
        """
        # Step 1: Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)

        # Step 2: Format context
        context_text = ContextFormatter.format_numbered(retrieved_docs)

        # Step 3: Build prompt
        prompt = self.prompt_builder.build_rag(query, context_text)

        # Step 4: Generate answer
        answer = self.model.generate(prompt, **kwargs)

        # Build context object
        context = RAGContext(
            query=query,
            retrieved_docs=retrieved_docs,
            metadata={
                "top_k": self.top_k,
                "context_text": context_text,
            },
        )

        # Build structured retrieved docs for metadata
        retrieved_structured = [
            RetrievedDoc(
                rank=i + 1,
                content=doc.text,
                score=getattr(doc, "score", None),
                doc_id=getattr(doc, "id", None),
            )
            for i, doc in enumerate(retrieved_docs)
        ]

        return RAGResponse(
            answer=answer,
            context=context,
            prompt=prompt,
            metadata=RAGResponseMeta(
                agent_type=AgentType.VANILLA_RAG,
                num_docs_used=len(retrieved_docs),
                retrieved_docs=retrieved_structured,
            ),
        )

    def batch_answer(self, queries: list[str], **kwargs: Any) -> list[RAGResponse]:
        """Generate answers for multiple queries using batch processing.

        Optimized for speed:
        - Batch retrieval (if supported by retriever)
        - Batch LLM generation

        Args:
            queries: List of input questions
            **kwargs: Additional generation parameters

        Returns:
            List of RAGResponse objects, one per query
        """
        # Step 1: Batch retrieve
        if hasattr(self.retriever, "batch_retrieve"):
            all_docs = self.retriever.batch_retrieve(queries, top_k=self.top_k)
        else:
            all_docs = [self.retriever.retrieve(q, top_k=self.top_k) for q in queries]

        # Step 2: Format contexts and build prompts
        prompts = []
        contexts = []
        context_texts = []

        for query, docs in zip(queries, all_docs):
            context_text = ContextFormatter.format_numbered(docs)
            context_texts.append(context_text)
            prompt = self.prompt_builder.build_rag(query, context_text)
            prompts.append(prompt)
            contexts.append(
                RAGContext(
                    query=query,
                    retrieved_docs=docs,
                    metadata={"top_k": self.top_k, "context_text": context_text},
                )
            )

        # Step 3: Batch generate
        answers = self.model.generate(prompts, **kwargs)

        # Step 4: Build responses
        responses = []
        for _query, prompt, answer, context, docs in zip(
            queries, prompts, answers, contexts, all_docs
        ):
            retrieved_structured = [
                RetrievedDoc(
                    rank=i + 1,
                    content=doc.text,
                    score=getattr(doc, "score", None),
                    doc_id=getattr(doc, "id", None),
                )
                for i, doc in enumerate(docs)
            ]

            response = RAGResponse(
                answer=answer,
                context=context,
                prompt=prompt,
                metadata=RAGResponseMeta(
                    agent_type=AgentType.VANILLA_RAG,
                    num_docs_used=len(docs),
                    batch_processing=True,
                    retrieved_docs=retrieved_structured,
                ),
            )
            responses.append(response)

        return responses
