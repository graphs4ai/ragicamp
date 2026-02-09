"""Fixed RAG Agent with clean resource management.

Uses model providers for explicit lifecycle control:
1. Load embedder → batch encode queries → batch search → unload embedder
2. Load generator → batch generate answers → unload generator

Each model gets full GPU when it runs.

Supports multiple search backends:
- VectorIndex: Dense semantic search
- HybridSearcher: Dense + sparse fusion
- HierarchicalSearcher: Child search, parent return
"""

from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ragicamp.agents.base import (
    Agent,
    AgentResult,
    Query,
    RetrievedDocInfo,
    Step,
    StepTimer,
    batch_embed_and_search,
    batch_transform_embed_and_search,
    is_hybrid_searcher,
)
from ragicamp.core.logging import get_logger
from ragicamp.core.step_types import BATCH_GENERATE, GENERATE, RERANK
from ragicamp.core.types import SearchBackend, SearchResult
from ragicamp.indexes.vector_index import VectorIndex
from ragicamp.models.providers import EmbedderProvider, GeneratorProvider
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)


class FixedRAGAgent(Agent):
    """Standard RAG agent with clean GPU resource management.

    Execution phases:
    1. EMBED: Load embedder (full GPU) → encode all queries
    2. SEARCH: Batch search index (CPU/mmap)
    3. UNLOAD: Free embedder from GPU
    4. GENERATE: Load generator (full GPU) → batch generate answers
    5. UNLOAD: Free generator from GPU

    Each model gets exclusive GPU access for maximum throughput.

    Supports multiple search backends:
    - VectorIndex: Dense semantic search
    - HybridSearcher: Dense + sparse with RRF fusion
    - HierarchicalSearcher: Child search, parent return
    """

    def __init__(
        self,
        name: str,
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: SearchBackend,
        top_k: int = 5,
        prompt_builder: PromptBuilder | None = None,
        retrieval_store: Any | None = None,
        retriever_name: str | None = None,
        reranker_provider: Any | None = None,
        fetch_k: int | None = None,
        query_transformer: Any | None = None,
        **config,
    ):
        """Initialize agent with providers (not loaded models).

        Args:
            name: Agent identifier
            embedder_provider: Provides embedder with lazy loading
            generator_provider: Provides generator with lazy loading
            index: Search backend (VectorIndex, HybridSearcher, or HierarchicalSearcher)
            top_k: Number of documents to retrieve (final count after reranking)
            prompt_builder: For building prompts
            retrieval_store: Optional RetrievalStore for caching retrieval results
            retriever_name: Retriever identifier for cache keys
            reranker_provider: Optional RerankerProvider for cross-encoder reranking
            fetch_k: Documents to retrieve before reranking (None = same as top_k)
            query_transformer: Optional QueryTransformer for query expansion
        """
        super().__init__(name, **config)

        self.embedder_provider = embedder_provider
        self.generator_provider = generator_provider
        self.index = index
        self.top_k = top_k
        self.prompt_builder = prompt_builder or PromptBuilder(PromptConfig())
        self.retrieval_store = retrieval_store
        self.retriever_name = retriever_name
        self.reranker_provider = reranker_provider
        self.fetch_k = fetch_k or top_k
        self.query_transformer = query_transformer

        # Check if this is a hybrid searcher (needs query text too)
        self._is_hybrid = is_hybrid_searcher(index)

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

        from time import perf_counter as _pc

        logger.info("Processing %d queries with phase-based execution", len(pending))
        _run_t0 = _pc()

        # Phase 1 & 2: Encode and search (embedder loaded)
        logger.info("=== Phase 1: Encoding & Searching ===")
        _phase_t0 = _pc()
        query_texts = [q.text for q in pending]
        retrievals, embed_steps = self._phase_retrieve(query_texts, show_progress)
        logger.info("Phase 1 (retrieve) completed in %.1fs", _pc() - _phase_t0)

        # Phase 3: Generate (generator loaded)
        logger.info("=== Phase 2: Generating ===")
        _phase_t0 = _pc()
        new_results = self._phase_generate(pending, retrievals, embed_steps, show_progress)
        logger.info("Phase 2 (generate) completed in %.1fs", _pc() - _phase_t0)

        # Stream results and checkpoint
        for result in new_results:
            results.append(result)
            if on_result:
                on_result(result)

        if checkpoint_path:
            self._save_checkpoint(results, checkpoint_path)

        logger.info("Completed %d queries in %.1fs", len(new_results), _pc() - _run_t0)
        return results

    def _phase_retrieve(
        self,
        query_texts: list[str],
        show_progress: bool,
    ) -> tuple[list[list[SearchResult]], list[Step]]:
        """Phase 1: Encode queries and search index, optionally rerank.

        Uses the shared batch_embed_and_search utility for all backends.
        When a reranker_provider is set, retrieves ``fetch_k`` documents
        and reranks down to ``top_k``.
        """
        # When reranking, retrieve more docs (fetch_k) then trim after rerank
        retrieve_k = self.fetch_k if self.reranker_provider else self.top_k

        retrievals, steps = batch_transform_embed_and_search(
            query_transformer=self.query_transformer,
            embedder_provider=self.embedder_provider,
            index=self.index,
            query_texts=query_texts,
            top_k=retrieve_k,
            is_hybrid=self._is_hybrid,
            retrieval_store=self.retrieval_store,
            retriever_name=self.retriever_name,
        )

        # Apply cross-encoder reranking if configured
        if self.reranker_provider is not None:
            retrievals, rerank_step = self._apply_reranking(
                query_texts, retrievals,
            )
            steps.append(rerank_step)

        logger.info("Retrieved documents for %d queries", len(query_texts))
        return retrievals, steps

    def _apply_reranking(
        self,
        query_texts: list[str],
        retrievals: list[list[SearchResult]],
    ) -> tuple[list[list[SearchResult]], Step]:
        """Load reranker, rerank documents, and rebuild SearchResult lists."""
        from time import perf_counter as _pc

        from ragicamp.core.types import Document, SearchResult

        _t0 = _pc()
        with self.reranker_provider.load() as reranker:
            # Extract Document lists from SearchResult lists
            docs_lists: list[list[Document]] = [
                [sr.document for sr in srs] for srs in retrievals
            ]

            reranked_docs = reranker.batch_rerank(
                query_texts, docs_lists, top_k=self.top_k,
            )

        # Rebuild SearchResult wrappers with reranker scores
        reranked_retrievals: list[list[SearchResult]] = []
        for docs in reranked_docs:
            reranked_retrievals.append([
                SearchResult(
                    document=doc,
                    score=getattr(doc, "score", 0.0),
                    rank=rank,
                )
                for rank, doc in enumerate(docs)
            ])

        elapsed = _pc() - _t0
        logger.info(
            "Reranked %d queries (%d -> %d docs each) in %.1fs",
            len(query_texts), self.fetch_k, self.top_k, elapsed,
        )

        rerank_step = Step(
            type=RERANK,
            input={"n_queries": len(query_texts), "fetch_k": self.fetch_k},
            output={"top_k": self.top_k},
            model=self.reranker_provider.model_name,
            timing_ms=elapsed * 1000,
        )

        return reranked_retrievals, rerank_step

    def _phase_generate(
        self,
        queries: list[Query],
        retrievals: list[list[SearchResult]],
        retrieve_steps: list[Step],
        show_progress: bool,
    ) -> list[AgentResult]:
        """Phase 2: Generate answers for all queries.

        Generator is loaded, used, then unloaded.
        """
        # Build all prompts
        prompts: list[str] = []
        for query, results in zip(queries, retrievals):
            docs = [r.document for r in results]
            context_text = ContextFormatter.format_numbered_from_docs(docs)
            prompt = self.prompt_builder.build_rag(query.text, context_text)
            prompts.append(prompt)

        # Load generator, generate, then unload
        with self.generator_provider.load() as generator:
            with StepTimer(BATCH_GENERATE, model=self.generator_provider.model_name) as step:
                step.input = {"n_prompts": len(prompts)}
                answers = generator.batch_generate(prompts)
                step.output = {"n_answers": len(answers)}

        # Generator is now unloaded - GPU is free

        # Build results
        results: list[AgentResult] = []

        iterator = zip(queries, retrievals, prompts, answers)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Building results")

        for query, search_results, prompt, answer in iterator:
            # Per-query steps
            query_steps = retrieve_steps.copy()
            query_steps.append(
                Step(
                    type=GENERATE,
                    input={"query": query.text},
                    output=answer,
                    model=self.generator_provider.model_name,
                )
            )

            # Build structured retrieved docs info
            retrieved_docs = RetrievedDocInfo.from_search_results(search_results)

            result = AgentResult(
                query=query,
                answer=answer,
                steps=query_steps,
                prompt=prompt,
                retrieved_docs=retrieved_docs,
                metadata={
                    "num_docs": len(search_results),
                    "top_k": self.top_k,
                },
            )
            results.append(result)

        return results


# =============================================================================
# Factory function for easy creation
# =============================================================================


def create_fixed_rag_agent(
    name: str,
    embedder_model: str,
    generator_model: str,
    index_path: str | Path,
    top_k: int = 5,
    embedder_backend: str = "vllm",
    generator_backend: str = "vllm",
    search_type: str = "dense",
    sparse_method: str = "tfidf",
    hybrid_alpha: float = 0.5,
) -> FixedRAGAgent:
    """Create a FixedRAGAgent with providers.

    Args:
        name: Agent name
        embedder_model: Embedding model name
        generator_model: Generator model name
        index_path: Path to vector index
        top_k: Number of documents to retrieve
        embedder_backend: "vllm" or "sentence_transformers"
        generator_backend: "vllm" or "hf"
        search_type: "dense", "hybrid", or "hierarchical"
        sparse_method: For hybrid: "tfidf" or "bm25"
        hybrid_alpha: Weight for dense vs sparse (0=sparse, 1=dense)

    Returns:
        Configured FixedRAGAgent
    """
    from ragicamp.models.providers import (
        EmbedderConfig,
        EmbedderProvider,
        GeneratorConfig,
        GeneratorProvider,
    )

    embedder_provider = EmbedderProvider(
        EmbedderConfig(
            model_name=embedder_model,
            backend=embedder_backend,
        )
    )

    generator_provider = GeneratorProvider(
        GeneratorConfig(
            model_name=generator_model,
            backend=generator_backend,
        )
    )

    # Load appropriate search backend
    if search_type == "dense":
        index = VectorIndex.load(index_path)
    elif search_type == "hybrid":
        from ragicamp.indexes.sparse import SparseIndex
        from ragicamp.retrievers.hybrid import HybridSearcher

        vector_index = VectorIndex.load(index_path)
        sparse_index = SparseIndex.load(
            f"{index_path}_sparse_{sparse_method}",
            documents=vector_index.documents,
        )
        index = HybridSearcher(
            vector_index=vector_index,
            sparse_index=sparse_index,
            alpha=hybrid_alpha,
        )
    elif search_type == "hierarchical":
        from ragicamp.indexes.hierarchical import HierarchicalIndex
        from ragicamp.retrievers.hierarchical import HierarchicalSearcher

        hier_index = HierarchicalIndex.load(str(index_path))
        index = HierarchicalSearcher(hier_index)
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    return FixedRAGAgent(
        name=name,
        embedder_provider=embedder_provider,
        generator_provider=generator_provider,
        index=index,
        top_k=top_k,
    )
