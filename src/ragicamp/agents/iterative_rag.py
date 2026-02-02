"""Iterative RAG agent - Multi-turn query refinement.

This agent implements an iterative retrieval strategy:
1. Retrieve with original query
2. LLM evaluates: "Is context sufficient to answer?"
3. If not sufficient: LLM generates refined query → retrieve again
4. Merge documents (deduplicate)
5. Repeat until max_iterations or sufficient
6. Generate final answer with accumulated context
"""

from typing import TYPE_CHECKING, Any, Optional

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.core.logging import get_logger
from ragicamp.core.schemas import AgentType, RAGResponseMeta, RetrievedDoc
from ragicamp.factory import AgentFactory
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder

if TYPE_CHECKING:
    from ragicamp.rag.query_transform.base import QueryTransformer
    from ragicamp.rag.rerankers.base import Reranker

logger = get_logger(__name__)


# Prompts for the iterative refinement process
SUFFICIENCY_PROMPT = """Based on the following context, determine if it contains enough information to answer the question.

Context:
{context}

Question: {query}

Instructions:
- If the context provides sufficient information to answer the question, respond with: SUFFICIENT
- If the context is missing key information needed to answer, respond with: INSUFFICIENT
- Then briefly explain your reasoning in one sentence.

Response:"""

REFINEMENT_PROMPT = """The retrieved context was insufficient to answer the question. Generate a refined search query that might find the missing information.

Original Question: {query}
Previous Query: {previous_query}

Retrieved Context (insufficient):
{context}

What key information is missing? Generate a better search query to find it.
Be specific and focus on the missing information, not what was already found.

Refined Query:"""


@AgentFactory.register("iterative_rag")
class IterativeRAGAgent(RAGAgent):
    """Iterative RAG agent with multi-turn query refinement.

    This agent iteratively refines its search query based on retrieved context,
    accumulating relevant documents until it has enough information to answer
    or reaches the maximum number of iterations.

    Flow:
    1. Initial retrieval with original query
    2. Evaluate if context is sufficient
    3. If insufficient and iterations remain:
       - Generate refined query
       - Retrieve more documents
       - Merge with existing documents
       - Repeat evaluation
    4. Generate final answer with all accumulated context

    Example:
        >>> agent = IterativeRAGAgent(
        ...     name="iterative",
        ...     model=model,
        ...     retriever=retriever,
        ...     max_iterations=2,
        ... )
        >>> response = agent.answer("What year did Einstein win the Nobel Prize?")
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        top_k: int = 5,
        max_iterations: int = 2,
        stop_on_sufficient: bool = True,
        prompt_builder: Optional[PromptBuilder] = None,
        # Advanced options
        query_transformer: Optional["QueryTransformer"] = None,
        reranker: Optional["Reranker"] = None,
        top_k_retrieve: Optional[int] = None,
        # Legacy parameters
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
        **kwargs: Any,
    ):
        """Initialize the iterative RAG agent.

        Args:
            name: Agent identifier
            model: Language model for generation and evaluation
            retriever: Document retriever
            top_k: Documents to use per iteration
            max_iterations: Maximum refinement iterations (default: 2)
            stop_on_sufficient: Stop early if context is sufficient (default: True)
            prompt_builder: PromptBuilder for final answer generation
            query_transformer: Optional initial query transformer
            reranker: Optional reranker for retrieved documents
            top_k_retrieve: Documents to retrieve before reranking
            system_prompt: System prompt for answer generation
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.stop_on_sufficient = stop_on_sufficient

        # Optional components
        self.query_transformer = query_transformer
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve or (top_k * 4 if reranker else top_k)

        # Prompt builder for final answer
        if prompt_builder is not None:
            self.prompt_builder = prompt_builder
        else:
            from ragicamp.utils.prompts import PromptConfig

            self.prompt_builder = PromptBuilder(PromptConfig(style=system_prompt))

        self._system_prompt = system_prompt

    def _retrieve(self, query: str) -> list[Document]:
        """Retrieve documents for a query."""
        # Apply query transformer if present
        search_query = query
        if self.query_transformer:
            search_query = self.query_transformer.transform(query)

        # Retrieve
        docs = self.retriever.retrieve(search_query, top_k=self.top_k_retrieve)

        # Rerank if present
        if self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=self.top_k)
        else:
            docs = docs[: self.top_k]

        return docs

    def _evaluate_sufficiency(self, query: str, context: str) -> bool:
        """Evaluate if the context is sufficient to answer the question."""
        prompt = SUFFICIENCY_PROMPT.format(context=context, query=query)
        response = self.model.generate(prompt, max_tokens=100)

        # Parse response - look for SUFFICIENT/INSUFFICIENT
        response_upper = response.upper()
        is_sufficient = "SUFFICIENT" in response_upper and "INSUFFICIENT" not in response_upper

        logger.debug("Sufficiency check: %s", "sufficient" if is_sufficient else "insufficient")
        return is_sufficient

    def _generate_refined_query(self, query: str, previous_query: str, context: str) -> str:
        """Generate a refined search query based on insufficient context."""
        prompt = REFINEMENT_PROMPT.format(
            query=query,
            previous_query=previous_query,
            context=context[:2000],  # Limit context in prompt
        )
        refined = self.model.generate(prompt, max_tokens=100)

        # Clean up the response
        refined = refined.strip()
        if refined.startswith('"') and refined.endswith('"'):
            refined = refined[1:-1]

        logger.debug("Refined query: %s -> %s", previous_query, refined)
        return refined

    def _merge_documents(
        self,
        existing: list[Document],
        new_docs: list[Document],
    ) -> list[Document]:
        """Merge documents, removing duplicates based on content hash."""
        seen_content: set[str] = set()
        merged: list[Document] = []

        for doc in existing + new_docs:
            # Use first 200 chars as content key for deduplication
            content_key = doc.text[:200] if doc.text else ""
            if content_key not in seen_content:
                seen_content.add(content_key)
                merged.append(doc)

        return merged

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using iterative query refinement.

        Args:
            query: The input question
            **kwargs: Additional generation parameters

        Returns:
            RAGResponse with answer and iteration metadata
        """
        # Track iterations and accumulated documents
        iterations: list[dict[str, Any]] = []
        all_docs: list[Document] = []
        current_query = query

        for iteration in range(self.max_iterations + 1):  # +1 for initial retrieval
            # Retrieve with current query
            new_docs = self._retrieve(current_query)
            all_docs = self._merge_documents(all_docs, new_docs)

            # Format context
            context_text = ContextFormatter.format_numbered(all_docs[: self.top_k * 2])

            # Log iteration
            iteration_info = {
                "iteration": iteration,
                "query": current_query,
                "docs_retrieved": len(new_docs),
                "total_docs": len(all_docs),
            }

            # Check if we should stop
            if iteration >= self.max_iterations:
                iteration_info["stopped"] = "max_iterations"
                iterations.append(iteration_info)
                break

            # Evaluate sufficiency
            if self.stop_on_sufficient:
                is_sufficient = self._evaluate_sufficiency(query, context_text)
                iteration_info["sufficient"] = is_sufficient

                if is_sufficient:
                    iteration_info["stopped"] = "sufficient"
                    iterations.append(iteration_info)
                    break

            iterations.append(iteration_info)

            # Generate refined query for next iteration
            if iteration < self.max_iterations:
                current_query = self._generate_refined_query(
                    query=query,
                    previous_query=current_query,
                    context=context_text,
                )

        # Use top documents for final answer
        final_docs = all_docs[: self.top_k * 2]
        context_text = ContextFormatter.format_numbered(final_docs)

        # Log iteration summary
        stop_reason = iterations[-1].get("stopped", "continued") if iterations else "none"
        logger.debug(
            "Iterative RAG: %d iterations, %d docs, stopped=%s",
            len(iterations),
            len(final_docs),
            stop_reason,
        )

        # Build final prompt
        prompt = self.prompt_builder.build_rag(query, context_text)

        # Generate answer
        answer = self.model.generate(prompt, **kwargs)

        # Build context object
        context = RAGContext(
            query=query,
            retrieved_docs=final_docs,
            intermediate_steps=iterations,
            metadata={
                "iterations": len(iterations),
                "total_docs_retrieved": len(all_docs),
                "final_docs_used": len(final_docs),
            },
        )

        # Build structured retrieved docs
        retrieved_structured = [
            RetrievedDoc(
                rank=i + 1,
                content=doc.text,
                score=getattr(doc, "score", None),
            )
            for i, doc in enumerate(final_docs)
        ]

        return RAGResponse(
            answer=answer,
            context=context,
            prompt=prompt,
            metadata=RAGResponseMeta(
                agent_type=AgentType.ITERATIVE_RAG,
                num_docs_used=len(final_docs),
                retrieved_docs=retrieved_structured,
            ),
        )

    def batch_answer(self, queries: list[str], **kwargs: Any) -> list[RAGResponse]:
        """Generate answers for multiple queries.

        Note: Iterative refinement is inherently sequential per query,
        so this uses a simple loop. Batch optimization is applied within
        each iteration's retrieval and generation steps.

        Args:
            queries: List of input questions
            **kwargs: Additional generation parameters

        Returns:
            List of RAGResponse objects
        """
        return [self.answer(q, **kwargs) for q in queries]
