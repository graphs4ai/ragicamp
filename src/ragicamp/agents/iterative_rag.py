"""Iterative RAG Agent - Multi-turn query refinement.

Uses model providers with proper lifecycle management:
- Each iteration: load embedder → retrieve → unload → load generator → evaluate → unload
- Final: load generator → generate answer → unload

Interleaved pattern with clean resource management.
"""

from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ragicamp.agents.base import Agent, AgentResult, Query, Step, StepTimer
from ragicamp.core.logging import get_logger
from ragicamp.indexes.vector_index import VectorIndex, SearchResult, Document
from ragicamp.models.providers import EmbedderProvider, GeneratorProvider
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)


# Prompts for iterative refinement
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


class IterativeRAGAgent(Agent):
    """Iterative RAG agent with multi-turn query refinement.

    This agent iteratively refines its search query based on retrieved context,
    accumulating relevant documents until it has enough information.

    Flow per query:
    1. Load embedder → encode query → search → unload
    2. Load generator → evaluate sufficiency → unload
    3. If insufficient: load generator → refine query → unload
    4. Repeat 1-3 until sufficient or max_iterations
    5. Load generator → generate final answer → unload

    Note: This is inherently sequential per query. For batch processing,
    queries are processed one at a time.
    """

    def __init__(
        self,
        name: str,
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: VectorIndex,
        top_k: int = 5,
        max_iterations: int = 2,
        stop_on_sufficient: bool = True,
        prompt_builder: PromptBuilder | None = None,
        **config,
    ):
        """Initialize agent with providers.

        Args:
            name: Agent identifier
            embedder_provider: Provides embedder with lazy loading
            generator_provider: Provides generator with lazy loading
            index: Vector index (just data)
            top_k: Documents to retrieve per iteration
            max_iterations: Maximum refinement iterations
            stop_on_sufficient: Stop early if context is sufficient
            prompt_builder: For building prompts
        """
        super().__init__(name, **config)

        self.embedder_provider = embedder_provider
        self.generator_provider = generator_provider
        self.index = index
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.stop_on_sufficient = stop_on_sufficient
        self.prompt_builder = prompt_builder or PromptBuilder(PromptConfig())

    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process queries with iterative refinement.

        Each query is processed sequentially due to the iterative nature.
        """
        # Load checkpoint if resuming
        results, completed_idx = [], set()
        if checkpoint_path:
            results, completed_idx = self._load_checkpoint(checkpoint_path)

        pending = [q for q in queries if q.idx not in completed_idx]

        if not pending:
            logger.info("All queries already completed")
            return results

        logger.info("IterativeRAG: Processing %d queries (max %d iterations each)",
                   len(pending), self.max_iterations)

        iterator = pending
        if show_progress:
            iterator = tqdm(pending, desc="Processing queries")

        for query in iterator:
            result = self._process_single_query(query)
            results.append(result)

            if on_result:
                on_result(result)

            if checkpoint_path:
                self._save_checkpoint(results, checkpoint_path)

        logger.info("Completed %d queries", len(pending))
        return results

    def _process_single_query(self, query: Query) -> AgentResult:
        """Process a single query with iterative refinement."""
        steps: list[Step] = []
        all_docs: list[Document] = []
        current_query_text = query.text
        iterations_info: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations + 1):
            # Step 1: Retrieve with current query
            with self.embedder_provider.load() as embedder:
                with StepTimer("encode", model=self.embedder_provider.model_name) as step:
                    query_embedding = embedder.batch_encode([current_query_text])
                    step.input = {"query": current_query_text, "iteration": iteration}
                    step.output = {"embedding_shape": query_embedding.shape}
                steps.append(step)

            # Search index
            with StepTimer("search") as step:
                search_results = self.index.batch_search(query_embedding, top_k=self.top_k)[0]
                new_docs = [r.document for r in search_results]
                step.input = {"iteration": iteration, "top_k": self.top_k}
                step.output = {"n_docs": len(new_docs)}
            steps.append(step)

            # Merge documents
            all_docs = self._merge_documents(all_docs, new_docs)
            context_text = ContextFormatter.format_numbered_from_docs(all_docs[:self.top_k * 2])

            iteration_info = {
                "iteration": iteration,
                "query": current_query_text,
                "docs_retrieved": len(new_docs),
                "total_docs": len(all_docs),
            }

            # Check if we should stop
            if iteration >= self.max_iterations:
                iteration_info["stopped"] = "max_iterations"
                iterations_info.append(iteration_info)
                break

            # Step 2: Evaluate sufficiency (requires generator)
            if self.stop_on_sufficient:
                with self.generator_provider.load() as generator:
                    with StepTimer("evaluate_sufficiency", model=self.generator_provider.model_name) as step:
                        is_sufficient = self._evaluate_sufficiency(
                            generator, query.text, context_text
                        )
                        step.input = {"query": query.text}
                        step.output = {"sufficient": is_sufficient}
                    steps.append(step)

                iteration_info["sufficient"] = is_sufficient

                if is_sufficient:
                    iteration_info["stopped"] = "sufficient"
                    iterations_info.append(iteration_info)
                    break

            iterations_info.append(iteration_info)

            # Step 3: Generate refined query (requires generator)
            if iteration < self.max_iterations:
                with self.generator_provider.load() as generator:
                    with StepTimer("refine_query", model=self.generator_provider.model_name) as step:
                        current_query_text = self._generate_refined_query(
                            generator, query.text, current_query_text, context_text
                        )
                        step.input = {"original_query": query.text}
                        step.output = {"refined_query": current_query_text}
                    steps.append(step)

        # Final answer generation
        final_docs = all_docs[:self.top_k * 2]
        context_text = ContextFormatter.format_numbered_from_docs(final_docs)
        prompt = self.prompt_builder.build_rag(query.text, context_text)

        with self.generator_provider.load() as generator:
            with StepTimer("generate", model=self.generator_provider.model_name) as step:
                answer = generator.generate(prompt)
                step.input = {"query": query.text}
                step.output = answer
            steps.append(step)

        return AgentResult(
            query=query,
            answer=answer,
            steps=steps,
            prompt=prompt,
            metadata={
                "iterations": iterations_info,
                "total_docs": len(all_docs),
                "final_docs_used": len(final_docs),
            },
        )

    def _evaluate_sufficiency(self, generator, query: str, context: str) -> bool:
        """Evaluate if context is sufficient to answer the question."""
        prompt = SUFFICIENCY_PROMPT.format(context=context, query=query)
        response = generator.generate(prompt, max_tokens=100)

        response_upper = response.upper()
        is_sufficient = "SUFFICIENT" in response_upper and "INSUFFICIENT" not in response_upper

        logger.debug("Sufficiency: %s", "sufficient" if is_sufficient else "insufficient")
        return is_sufficient

    def _generate_refined_query(
        self, generator, original_query: str, previous_query: str, context: str
    ) -> str:
        """Generate a refined search query."""
        prompt = REFINEMENT_PROMPT.format(
            query=original_query,
            previous_query=previous_query,
            context=context[:2000],
        )
        refined = generator.generate(prompt, max_tokens=100)

        # Clean up
        refined = refined.strip()
        if refined.startswith('"') and refined.endswith('"'):
            refined = refined[1:-1]

        logger.debug("Refined query: %s -> %s", previous_query[:30], refined[:30])
        return refined

    def _merge_documents(
        self,
        existing: list[Document],
        new_docs: list[Document],
    ) -> list[Document]:
        """Merge documents, removing duplicates based on content."""
        seen_content: set[str] = set()
        merged: list[Document] = []

        for doc in existing + new_docs:
            # Use first 200 chars as content key
            content_key = doc.text[:200] if doc.text else ""
            if content_key not in seen_content:
                seen_content.add(content_key)
                merged.append(doc)

        return merged
