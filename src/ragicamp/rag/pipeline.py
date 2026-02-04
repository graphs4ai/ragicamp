"""Modular RAG pipeline orchestrator.

The RAGPipeline combines:
- Query transformation (HyDE, multi-query, or passthrough)
- Retrieval (using VectorIndex + EmbedderProvider)
- Reranking (cross-encoder)

Clean architecture with provider-based GPU lifecycle management.

Example usage:
    from ragicamp.models.providers import EmbedderProvider, GeneratorProvider
    from ragicamp.indexes import VectorIndex

    # Create providers
    embedder = EmbedderProvider(EmbedderConfig("BAAI/bge-large-en"))
    generator = GeneratorProvider(GeneratorConfig("meta-llama/Llama-3.2-3B"))
    index = VectorIndex.load("my_index")

    # Create pipeline with optional components
    pipeline = RAGPipeline(
        embedder_provider=embedder,
        index=index,
        query_transformer=HyDETransformer(generator),  # Optional
        reranker=CrossEncoderReranker("bge"),  # Optional
        top_k_retrieve=20,
        top_k_final=5,
    )

    # Retrieve documents
    docs = pipeline.retrieve("What is the capital of France?")
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ragicamp.core.logging import get_logger
from ragicamp.core.schemas import (
    PipelineLog,
    QueryTransformStep,
    RerankStep,
    RetrievalStep,
)
from ragicamp.core.types import Document
from ragicamp.rag.query_transform.base import PassthroughTransformer, QueryTransformer
from ragicamp.rag.rerankers.base import Reranker

if TYPE_CHECKING:
    from ragicamp.indexes.vector_index import VectorIndex
    from ragicamp.models.providers import EmbedderProvider

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from pipeline retrieval with full metadata.

    Attributes:
        documents: Final retrieved documents (after all pipeline stages)
        pipeline_log: Log of all pipeline stages with scores and latencies
    """

    documents: list[Document] = field(default_factory=list)
    pipeline_log: PipelineLog | None = None


class RAGPipeline:
    """Modular RAG pipeline with pluggable components.

    The pipeline executes in stages:
    1. Query Transformation: Expand/modify the query (HyDE, multi-query)
    2. Embedding: Encode queries using EmbedderProvider
    3. Retrieval: Search VectorIndex for candidates
    4. Deduplication: Merge results from multiple queries
    5. Reranking: Reorder candidates using a cross-encoder
    6. Top-K Selection: Return final documents

    Each component is optional and can be swapped out.
    Uses provider pattern for clean GPU lifecycle management.
    """

    def __init__(
        self,
        embedder_provider: "EmbedderProvider",
        index: "VectorIndex",
        query_transformer: QueryTransformer | None = None,
        reranker: Reranker | None = None,
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
    ):
        """Initialize the RAG pipeline.

        Args:
            embedder_provider: Provider for embedding model (lazy loading)
            index: Vector index with documents
            query_transformer: Optional query transformer (HyDE, multi-query)
            reranker: Optional reranker (cross-encoder)
            top_k_retrieve: Number of docs to retrieve per query (before reranking)
            top_k_final: Final number of docs to return (after reranking)
        """
        self.embedder_provider = embedder_provider
        self.index = index
        self.query_transformer = query_transformer or PassthroughTransformer()
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final
        self.name = f"pipeline_{index.config.embedding_model.split('/')[-1]}"

    def retrieve(self, query: str) -> list[Document]:
        """Execute the full retrieval pipeline.

        Args:
            query: The user's query

        Returns:
            List of top-k documents after all pipeline stages
        """
        result = self.retrieve_with_log(query)
        return result.documents

    def retrieve_with_log(self, query: str) -> RetrievalResult:
        """Execute the full retrieval pipeline with detailed logging.

        Returns both documents and a complete log of all pipeline stages,
        including scores at each stage for analysis.

        Args:
            query: The user's query

        Returns:
            RetrievalResult with documents and pipeline_log
        """
        pipeline_log = PipelineLog()
        total_start = time.perf_counter()

        # Stage 1: Query transformation
        transform_start = time.perf_counter()
        transformed_queries = self.query_transformer.transform(query)
        transform_ms = (time.perf_counter() - transform_start) * 1000

        # Get transformer type name
        transformer_type = self.query_transformer.__class__.__name__.lower()
        if "hyde" in transformer_type:
            transformer_type = "hyde"
        elif "multi" in transformer_type:
            transformer_type = "multi_query"
        elif "passthrough" in transformer_type:
            transformer_type = "passthrough"

        # Get hypothetical doc if HyDE
        hypothetical_doc = None
        if hasattr(self.query_transformer, "last_hypothetical"):
            hypothetical_doc = self.query_transformer.last_hypothetical

        pipeline_log.add_step(
            QueryTransformStep(
                transformer_type=transformer_type,
                original_query=query,
                transformed_queries=transformed_queries,
                hypothetical_doc=hypothetical_doc,
                latency_ms=round(transform_ms, 2),
            )
        )

        logger.debug(
            "Query transformation: %d queries generated",
            len(transformed_queries),
        )

        # Stage 2: Embed queries and search
        retrieval_start = time.perf_counter()
        all_docs: list[Document] = []
        seen_ids: set[str] = set()

        with self.embedder_provider.load() as embedder:
            # Batch encode all transformed queries
            query_embeddings = embedder.batch_encode(transformed_queries)
            if not isinstance(query_embeddings, np.ndarray):
                query_embeddings = np.array(query_embeddings)

            # Search index
            all_results = self.index.batch_search(query_embeddings, top_k=self.top_k_retrieve)

            # Flatten and deduplicate
            for query_results in all_results:
                for result in query_results:
                    doc = result.document
                    doc.score = result.score
                    if doc.id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(doc.id)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Capture retrieval scores before any reranking
        for rank, doc in enumerate(
            sorted(all_docs, key=lambda d: getattr(d, "score", 0), reverse=True), 1
        ):
            doc._retrieval_score = getattr(doc, "score", None)
            doc._retrieval_rank = rank

        # Build candidates list for logging
        candidates = [
            {
                "doc_id": doc.id,
                "score": round(getattr(doc, "score", 0), 4),
                "rank": getattr(doc, "_retrieval_rank", i + 1),
            }
            for i, doc in enumerate(
                sorted(all_docs, key=lambda d: getattr(d, "score", 0), reverse=True)[
                    : self.top_k_final * 2
                ]
            )
        ]

        pipeline_log.add_step(
            RetrievalStep(
                retriever_type="dense",
                retriever_name=self.name,
                num_queries=len(transformed_queries),
                num_candidates=len(all_docs),
                top_k_requested=self.top_k_retrieve,
                latency_ms=round(retrieval_ms, 2),
                candidates=candidates,
            )
        )

        logger.debug(
            "Retrieved %d unique documents from %d queries",
            len(all_docs),
            len(transformed_queries),
        )

        # If no reranker, sort by score and return top-k
        if self.reranker is None:
            sorted_docs = sorted(
                all_docs,
                key=lambda d: getattr(d, "score", 0),
                reverse=True,
            )
            final_docs = sorted_docs[: self.top_k_final]
            pipeline_log.total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
            return RetrievalResult(documents=final_docs, pipeline_log=pipeline_log)

        # Stage 3: Reranking
        rerank_start = time.perf_counter()

        # Use original query for reranking (not transformed versions)
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=all_docs,
            top_k=self.top_k_final,
        )

        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        # Build score changes for logging
        score_changes = []
        for new_rank, doc in enumerate(reranked_docs, 1):
            retrieval_score = getattr(doc, "_retrieval_score", None)
            retrieval_rank = getattr(doc, "_retrieval_rank", None)
            rerank_score = getattr(doc, "score", None)

            doc._rerank_score = rerank_score

            score_changes.append(
                {
                    "doc_id": doc.id,
                    "retrieval_score": (round(retrieval_score, 4) if retrieval_score else None),
                    "rerank_score": round(rerank_score, 4) if rerank_score else None,
                    "retrieval_rank": retrieval_rank,
                    "final_rank": new_rank,
                    "rank_change": ((retrieval_rank - new_rank) if retrieval_rank else None),
                }
            )

        reranker_model = getattr(self.reranker, "model_name", "unknown")

        pipeline_log.add_step(
            RerankStep(
                reranker_type="cross_encoder",
                reranker_model=reranker_model,
                num_input_docs=len(all_docs),
                num_output_docs=len(reranked_docs),
                latency_ms=round(rerank_ms, 2),
                score_changes=score_changes,
            )
        )

        logger.debug("Reranked to %d documents", len(reranked_docs))

        pipeline_log.total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
        return RetrievalResult(documents=reranked_docs, pipeline_log=pipeline_log)

    def batch_retrieve(self, queries: list[str]) -> list[list[Document]]:
        """Retrieve documents for multiple queries with batched operations.

        Optimizations:
        - Batch query transformation
        - Batch embedding + search
        - Sequential reranking (cross-encoders can't easily batch multiple queries)

        Args:
            queries: List of user queries

        Returns:
            List of document lists, one per query
        """
        if not queries:
            return []

        # Stage 1: Batch query transformation
        transformed_query_lists = self.query_transformer.batch_transform(queries)

        # Stage 2: Collect all transformed queries for batch embedding
        all_transformed = []
        query_indices = []

        for orig_idx, transformed in enumerate(transformed_query_lists):
            for t_query in transformed:
                all_transformed.append(t_query)
                query_indices.append(orig_idx)

        # Stage 3: Batch embed and search
        with self.embedder_provider.load() as embedder:
            query_embeddings = embedder.batch_encode(all_transformed)
            if not isinstance(query_embeddings, np.ndarray):
                query_embeddings = np.array(query_embeddings)

            all_results = self.index.batch_search(query_embeddings, top_k=self.top_k_retrieve)

        # Stage 4: Aggregate per original query
        results_per_query: list[list[Document]] = [[] for _ in queries]
        seen_per_query: list[set[str]] = [set() for _ in queries]

        for search_results, orig_idx in zip(all_results, query_indices):
            for result in search_results:
                doc = result.document
                doc.score = result.score
                if doc.id not in seen_per_query[orig_idx]:
                    results_per_query[orig_idx].append(doc)
                    seen_per_query[orig_idx].add(doc.id)

        # Stage 5: Rerank
        if self.reranker is not None:
            if hasattr(self.reranker, "batch_rerank"):
                return self.reranker.batch_rerank(
                    queries=queries,
                    documents_list=results_per_query,
                    top_k=self.top_k_final,
                )
            else:
                reranked = []
                for orig_query, docs in zip(queries, results_per_query):
                    reranked_docs = self.reranker.rerank(
                        query=orig_query,
                        documents=docs,
                        top_k=self.top_k_final,
                    )
                    reranked.append(reranked_docs)
                return reranked
        else:
            # Sort by score and trim
            final_results = []
            for docs in results_per_query:
                sorted_docs = sorted(
                    docs,
                    key=lambda d: getattr(d, "score", 0),
                    reverse=True,
                )
                final_results.append(sorted_docs[: self.top_k_final])
            return final_results

    def get_config(self) -> dict:
        """Get pipeline configuration for logging/debugging."""
        return {
            "embedder": self.embedder_provider.model_name,
            "index": self.index.config.embedding_model,
            "query_transformer": repr(self.query_transformer),
            "reranker": repr(self.reranker) if self.reranker else None,
            "top_k_retrieve": self.top_k_retrieve,
            "top_k_final": self.top_k_final,
        }

    def __repr__(self) -> str:
        components = [f"index={self.index.config.embedding_model}"]
        if not isinstance(self.query_transformer, PassthroughTransformer):
            components.append(f"transformer={self.query_transformer.__class__.__name__}")
        if self.reranker:
            components.append(f"reranker={self.reranker.__class__.__name__}")
        return f"RAGPipeline({', '.join(components)})"
