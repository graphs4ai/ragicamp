"""Vanilla RAG Agent - Simplest possible RAG baseline.

Uses model providers for explicit lifecycle control:
1. Load embedder → batch encode queries → batch search → unload embedder
2. Load generator → batch generate answers → unload generator

Each model gets full GPU when it runs.
Identical to FixedRAGAgent but without any query transformation.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from ragicamp.agents.base import Agent, AgentResult, Query, Step, StepTimer
from ragicamp.core.logging import get_logger
from ragicamp.indexes.vector_index import VectorIndex, SearchResult
from ragicamp.models.providers import EmbedderProvider, GeneratorProvider
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)


class VanillaRAGAgent(Agent):
    """Simplest RAG agent: retrieve → generate.

    This agent provides a clean baseline for RAG experiments:
    - No query transformation (uses raw query)
    - No reranking (uses retriever ordering)
    - Single retrieval step

    Execution phases:
    1. EMBED: Load embedder (full GPU) → encode all queries
    2. SEARCH: Batch search index (CPU/mmap)
    3. UNLOAD: Free embedder from GPU
    4. GENERATE: Load generator (full GPU) → batch generate answers
    5. UNLOAD: Free generator from GPU
    """

    def __init__(
        self,
        name: str,
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: VectorIndex,
        top_k: int = 5,
        prompt_builder: PromptBuilder | None = None,
        **config,
    ):
        """Initialize agent with providers (not loaded models).

        Args:
            name: Agent identifier
            embedder_provider: Provides embedder with lazy loading
            generator_provider: Provides generator with lazy loading
            index: Vector index (just data, no models)
            top_k: Number of documents to retrieve
            prompt_builder: For building prompts
        """
        super().__init__(name, **config)

        self.embedder_provider = embedder_provider
        self.generator_provider = generator_provider
        self.index = index
        self.top_k = top_k
        self.prompt_builder = prompt_builder or PromptBuilder(PromptConfig())

    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process all queries with phase-based resource management.

        Phase 1: Embed all queries (embedder gets full GPU)
        Phase 2: Search index (CPU)
        Phase 3: Generate all answers (generator gets full GPU)
        """
        # Load checkpoint if resuming
        results, completed_idx = [], set()
        if checkpoint_path:
            results, completed_idx = self._load_checkpoint(checkpoint_path)

        pending = [q for q in queries if q.idx not in completed_idx]

        if not pending:
            logger.info("All queries already completed")
            return results

        logger.info("VanillaRAG: Processing %d queries", len(pending))

        # Phase 1 & 2: Encode and search
        logger.info("=== Phase 1: Encoding & Searching ===")
        query_texts = [q.text for q in pending]
        retrievals, embed_steps = self._phase_retrieve(query_texts, show_progress)

        # Phase 3: Generate
        logger.info("=== Phase 2: Generating ===")
        new_results = self._phase_generate(
            pending, retrievals, embed_steps, show_progress
        )

        # Stream results and checkpoint
        for result in new_results:
            results.append(result)
            if on_result:
                on_result(result)

        if checkpoint_path:
            self._save_checkpoint(results, checkpoint_path)

        logger.info("Completed %d queries", len(new_results))
        return results

    def _phase_retrieve(
        self,
        query_texts: list[str],
        show_progress: bool,
    ) -> tuple[list[list[SearchResult]], list[Step]]:
        """Phase 1: Encode queries and search index."""
        steps: list[Step] = []

        # Load embedder, encode, then unload
        with self.embedder_provider.load() as embedder:
            with StepTimer("batch_encode", model=self.embedder_provider.model_name) as step:
                step.input = {"n_queries": len(query_texts)}
                query_embeddings = embedder.batch_encode(query_texts)
                step.output = {"embedding_shape": query_embeddings.shape}
            steps.append(step)

            logger.info("Encoded %d queries", len(query_texts))

        # Embedder is now unloaded

        # Batch search (CPU)
        with StepTimer("batch_search") as step:
            step.input = {"n_queries": len(query_texts), "top_k": self.top_k}
            retrievals = self.index.batch_search(query_embeddings, top_k=self.top_k)
            step.output = {"n_results": sum(len(r) for r in retrievals)}
        steps.append(step)

        logger.info("Retrieved documents for %d queries", len(query_texts))

        return retrievals, steps

    def _phase_generate(
        self,
        queries: list[Query],
        retrievals: list[list[SearchResult]],
        retrieve_steps: list[Step],
        show_progress: bool,
    ) -> list[AgentResult]:
        """Phase 2: Generate answers for all queries."""
        # Build all prompts
        prompts: list[str] = []
        for query, results in zip(queries, retrievals):
            docs = [r.document for r in results]
            context_text = ContextFormatter.format_numbered_from_docs(docs)
            prompt = self.prompt_builder.build_rag(query.text, context_text)
            prompts.append(prompt)

        # Load generator, generate, then unload
        with self.generator_provider.load() as generator:
            with StepTimer("batch_generate", model=self.generator_provider.model_name) as step:
                step.input = {"n_prompts": len(prompts)}
                answers = generator.batch_generate(prompts)
                step.output = {"n_answers": len(answers)}

        # Generator is now unloaded

        # Build results
        results: list[AgentResult] = []

        iterator = zip(queries, retrievals, prompts, answers)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Building results")

        for query, search_results, prompt, answer in iterator:
            query_steps = retrieve_steps.copy()
            query_steps.append(Step(
                type="generate",
                input={"query": query.text},
                output=answer,
                model=self.generator_provider.model_name,
            ))

            result = AgentResult(
                query=query,
                answer=answer,
                steps=query_steps,
                prompt=prompt,
                metadata={
                    "num_docs": len(search_results),
                    "top_k": self.top_k,
                    "doc_scores": [r.score for r in search_results],
                },
            )
            results.append(result)

        return results
