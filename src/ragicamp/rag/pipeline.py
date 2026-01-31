"""Modular RAG pipeline orchestrator.

The RAGPipeline combines:
- Query transformation (HyDE, multi-query, or passthrough)
- Retrieval (dense, sparse, hybrid, or hierarchical)
- Reranking (cross-encoder)

This modular design allows mixing and matching components to find
the best configuration for your use case.

Example usage:
    pipeline = RAGPipeline(
        retriever=HybridRetriever("my_index"),
        query_transformer=HyDETransformer(llm),
        reranker=CrossEncoderReranker("bge"),
        top_k_retrieve=20,
        top_k_final=5,
    )

    docs = pipeline.retrieve("What is the capital of France?")
"""

from typing import TYPE_CHECKING, Optional

from ragicamp.core.logging import get_logger
from ragicamp.rag.query_transform.base import PassthroughTransformer, QueryTransformer
from ragicamp.rag.rerankers.base import Reranker

if TYPE_CHECKING:
    from ragicamp.retrievers.base import Document, Retriever

logger = get_logger(__name__)


class RAGPipeline:
    """Modular RAG pipeline with pluggable components.

    The pipeline executes in stages:
    1. Query Transformation: Expand/modify the query (HyDE, multi-query)
    2. Retrieval: Get candidate documents from all query variations
    3. Deduplication: Merge results from multiple queries
    4. Reranking: Reorder candidates using a cross-encoder
    5. Top-K Selection: Return final documents

    Each component is optional and can be swapped out.
    """

    def __init__(
        self,
        retriever: "Retriever",
        query_transformer: Optional[QueryTransformer] = None,
        reranker: Optional[Reranker] = None,
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
    ):
        """Initialize the RAG pipeline.

        Args:
            retriever: The retriever to use (dense, hybrid, hierarchical)
            query_transformer: Optional query transformer (HyDE, multi-query)
            reranker: Optional reranker (cross-encoder)
            top_k_retrieve: Number of docs to retrieve per query (before reranking)
            top_k_final: Final number of docs to return (after reranking)
        """
        self.retriever = retriever
        self.query_transformer = query_transformer or PassthroughTransformer()
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final

    def retrieve(self, query: str) -> list["Document"]:
        """Execute the full retrieval pipeline.

        Args:
            query: The user's query

        Returns:
            List of top-k documents after all pipeline stages
        """
        # Stage 1: Query transformation
        transformed_queries = self.query_transformer.transform(query)
        logger.debug(
            "Query transformation: %d queries generated",
            len(transformed_queries),
        )

        # Stage 2: Retrieval from all query variations
        all_docs: list[Document] = []
        seen_ids: set[str] = set()

        for q in transformed_queries:
            docs = self.retriever.retrieve(q, top_k=self.top_k_retrieve)
            for doc in docs:
                # Deduplicate by document ID
                if doc.id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc.id)

        logger.debug(
            "Retrieved %d unique documents from %d queries",
            len(all_docs),
            len(transformed_queries),
        )

        # If no reranker, sort by score and return top-k
        if self.reranker is None:
            # Sort by existing scores (if available)
            sorted_docs = sorted(
                all_docs,
                key=lambda d: getattr(d, "score", 0),
                reverse=True,
            )
            return sorted_docs[: self.top_k_final]

        # Stage 3: Reranking
        # Use original query for reranking (not transformed versions)
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=all_docs,
            top_k=self.top_k_final,
        )

        logger.debug(
            "Reranked to %d documents",
            len(reranked_docs),
        )

        return reranked_docs

    def batch_retrieve(self, queries: list[str]) -> list[list["Document"]]:
        """Retrieve documents for multiple queries with batched operations.

        Optimizations:
        - Batch query transformation (e.g., HyDE generates all hypotheticals in one LLM call)
        - Batch retrieval for all transformed queries
        - Sequential reranking (cross-encoders can't easily batch multiple queries)

        Args:
            queries: List of user queries

        Returns:
            List of document lists, one per query
        """
        if not queries:
            return []

        # Stage 1: Batch query transformation
        # This uses batch_transform() which is much faster for HyDE
        transformed_query_lists = self.query_transformer.batch_transform(queries)

        # Stage 2: Collect all transformed queries for batch retrieval
        # We need to track which original query each transformed query belongs to
        all_transformed = []
        query_indices = []  # Maps each transformed query to its original index

        for orig_idx, transformed in enumerate(transformed_query_lists):
            for t_query in transformed:
                all_transformed.append(t_query)
                query_indices.append(orig_idx)

        # Stage 3: Batch retrieve for all transformed queries at once
        if hasattr(self.retriever, "batch_retrieve") and all_transformed:
            all_retrieved = self.retriever.batch_retrieve(
                all_transformed, top_k=self.top_k_retrieve
            )
        else:
            # Fallback to sequential
            all_retrieved = [
                self.retriever.retrieve(q, top_k=self.top_k_retrieve) for q in all_transformed
            ]

        # Stage 4: Aggregate and deduplicate per original query
        results_per_query: list[list[Document]] = [[] for _ in queries]
        seen_per_query: list[set[str]] = [set() for _ in queries]

        for docs, orig_idx in zip(all_retrieved, query_indices):
            for doc in docs:
                if doc.id not in seen_per_query[orig_idx]:
                    results_per_query[orig_idx].append(doc)
                    seen_per_query[orig_idx].add(doc.id)

        # Stage 5: Rerank (batch if supported, else sequential)
        if self.reranker is not None:
            if hasattr(self.reranker, "batch_rerank"):
                # Use batch reranking for efficiency
                return self.reranker.batch_rerank(
                    queries=queries,
                    documents_list=results_per_query,
                    top_k=self.top_k_final,
                )
            else:
                # Fallback to sequential reranking
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
            "retriever": repr(self.retriever),
            "query_transformer": repr(self.query_transformer),
            "reranker": repr(self.reranker) if self.reranker else None,
            "top_k_retrieve": self.top_k_retrieve,
            "top_k_final": self.top_k_final,
        }

    def __repr__(self) -> str:
        components = [f"retriever={self.retriever.name}"]
        if not isinstance(self.query_transformer, PassthroughTransformer):
            components.append(f"transformer={self.query_transformer.__class__.__name__}")
        if self.reranker:
            components.append(f"reranker={self.reranker.__class__.__name__}")
        return f"RAGPipeline({', '.join(components)})"
