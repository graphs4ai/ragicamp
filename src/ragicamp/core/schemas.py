"""Data contracts for RAGiCamp.

This module defines the canonical data structures used throughout the framework.
All components should use these schemas to ensure consistency.

Key Principles:
1. Use dataclasses for structured data (not Dict[str, Any])
2. All fields are explicitly typed
3. Optional fields have sensible defaults
4. Schemas are the single source of truth

Usage:
    from ragicamp.core.schemas import PredictionRecord, RetrievedDoc

    # Create a prediction
    pred = PredictionRecord(
        idx=0,
        question="What is the capital?",
        prediction="Paris",
        expected=["Paris"],
        prompt="Answer: What is the capital?",
    )
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from ragicamp.core.constants import AgentType

# =============================================================================
# Retrieved Documents
# =============================================================================


@dataclass
class RetrievedDoc:
    """A single retrieved document.

    This is the canonical format for retrieved documents throughout the system.
    Saved in predictions.json as part of RAG experiments.

    Attributes:
        rank: Final position in results (1-indexed, after all pipeline stages)
        content: Document text content
        score: Final score (after reranking if applicable)
        retrieval_score: Original retrieval score (before reranking)
        rerank_score: Cross-encoder rerank score (if reranking was applied)
        retrieval_rank: Original rank before reranking
        source: Source identifier (optional, e.g., "wikipedia")
        doc_id: Document ID in corpus (optional)
    """

    rank: int
    content: str
    score: Optional[float] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_rank: Optional[int] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {"rank": self.rank, "content": self.content}
        if self.score is not None:
            d["score"] = self.score
        if self.retrieval_score is not None:
            d["retrieval_score"] = self.retrieval_score
        if self.rerank_score is not None:
            d["rerank_score"] = self.rerank_score
        if self.retrieval_rank is not None:
            d["retrieval_rank"] = self.retrieval_rank
        if self.source is not None:
            d["source"] = self.source
        if self.doc_id is not None:
            d["doc_id"] = self.doc_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RetrievedDoc":
        """Create from dict (for loading from JSON)."""
        return cls(
            rank=d["rank"],
            content=d["content"],
            score=d.get("score"),
            retrieval_score=d.get("retrieval_score"),
            rerank_score=d.get("rerank_score"),
            retrieval_rank=d.get("retrieval_rank"),
            source=d.get("source"),
            doc_id=d.get("doc_id"),
        )


# =============================================================================
# Pipeline Step Logging (modular metadata for each RAG component)
# =============================================================================


@dataclass
class QueryTransformStep:
    """Log entry for query transformation stage.

    Captures what the query transformer did (HyDE, multi-query, etc.)
    """

    transformer_type: str  # "hyde", "multi_query", "passthrough"
    original_query: str
    transformed_queries: list[str]
    hypothetical_doc: Optional[str] = None  # For HyDE
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "stage": "query_transform",
            "transformer_type": self.transformer_type,
            "original_query": self.original_query,
            "transformed_queries": self.transformed_queries,
        }
        if self.hypothetical_doc is not None:
            d["hypothetical_doc"] = self.hypothetical_doc
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        return d


@dataclass
class RetrievalStep:
    """Log entry for retrieval stage.

    Captures retrieval results before any reranking.
    """

    retriever_type: str  # "dense", "sparse", "hybrid", "hierarchical"
    retriever_name: str  # e.g., "en_bge_large_c512_o50"
    num_queries: int  # How many queries were used (after transform)
    num_candidates: int  # Total unique docs retrieved
    top_k_requested: int
    latency_ms: Optional[float] = None
    # Candidates with their retrieval scores (before rerank)
    candidates: Optional[list[dict[str, Any]]] = None  # [{doc_id, score, rank}]

    def to_dict(self) -> dict[str, Any]:
        d = {
            "stage": "retrieval",
            "retriever_type": self.retriever_type,
            "retriever_name": self.retriever_name,
            "num_queries": self.num_queries,
            "num_candidates": self.num_candidates,
            "top_k_requested": self.top_k_requested,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.candidates is not None:
            d["candidates"] = self.candidates
        return d


@dataclass
class RerankStep:
    """Log entry for reranking stage.

    Captures score changes from reranking.
    """

    reranker_type: str  # "cross_encoder", "cohere", etc.
    reranker_model: str  # e.g., "bge-reranker-large"
    num_input_docs: int
    num_output_docs: int
    latency_ms: Optional[float] = None
    # Score changes: [{doc_id, retrieval_score, rerank_score, rank_change}]
    score_changes: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "stage": "rerank",
            "reranker_type": self.reranker_type,
            "reranker_model": self.reranker_model,
            "num_input_docs": self.num_input_docs,
            "num_output_docs": self.num_output_docs,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.score_changes is not None:
            d["score_changes"] = self.score_changes
        return d


@dataclass
class PipelineLog:
    """Complete log of all pipeline stages.

    Modular design: each component adds its own step.
    Agents can extend with their own steps (e.g., self-rag decision, iteration).
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: Optional[float] = None

    def add_step(
        self, step: "QueryTransformStep | RetrievalStep | RerankStep | dict"
    ) -> None:
        """Add a step to the pipeline log."""
        if hasattr(step, "to_dict"):
            self.steps.append(step.to_dict())
        else:
            self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        d = {"pipeline_steps": self.steps}
        if self.total_latency_ms is not None:
            d["total_latency_ms"] = self.total_latency_ms
        return d


# =============================================================================
# Predictions
# =============================================================================


@dataclass
class PredictionRecord:
    """A single prediction record.

    This is the canonical format for predictions saved in predictions.json.
    All experiments MUST save predictions in this format.

    Attributes:
        idx: Index in the dataset (0-indexed)
        question: The input question
        prediction: The model's prediction
        expected: List of acceptable answers
        prompt: The full prompt sent to the model
        retrieved_docs: Retrieved documents (RAG only)
        metrics: Per-item metric scores (computed after generation)
        error: Error message if prediction failed
    """

    idx: int
    question: str
    prediction: str
    expected: list[str]
    prompt: str
    retrieved_docs: Optional[list[RetrievedDoc]] = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {
            "idx": self.idx,
            "question": self.question,
            "prediction": self.prediction,
            "expected": self.expected,
            "prompt": self.prompt,
            "metrics": self.metrics,
        }
        if self.retrieved_docs:
            d["retrieved_docs"] = [doc.to_dict() for doc in self.retrieved_docs]
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PredictionRecord":
        """Create from dict (for loading from JSON)."""
        retrieved_docs = None
        if "retrieved_docs" in d and d["retrieved_docs"]:
            retrieved_docs = [RetrievedDoc.from_dict(doc) for doc in d["retrieved_docs"]]

        return cls(
            idx=d["idx"],
            question=d["question"],
            prediction=d["prediction"],
            expected=d["expected"],
            prompt=d["prompt"],
            retrieved_docs=retrieved_docs,
            metrics=d.get("metrics", {}),
            error=d.get("error"),
        )


# =============================================================================
# Agent Response Metadata
# =============================================================================


@dataclass
class RAGResponseMeta:
    """Typed metadata for RAG responses.

    Instead of Dict[str, Any], use this structured class.
    Ensures all agents return consistent metadata.

    Modular design: Each pipeline component can add its own step to pipeline_log.
    This allows analysis of each stage (retrieval scores, rerank scores, etc.)

    Attributes:
        agent_type: Type of agent that generated the response
        batch_processing: Whether this was part of a batch
        num_docs_used: Number of documents used (RAG only)
        retrieved_docs: Structured retrieved documents with all scores (RAG only)
        pipeline_log: Log of all pipeline stages (query transform, retrieval, rerank)
    """

    agent_type: AgentType
    batch_processing: bool = False
    num_docs_used: Optional[int] = None
    retrieved_docs: Optional[list[RetrievedDoc]] = None
    pipeline_log: Optional[PipelineLog] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backwards compatibility."""
        d = {
            "agent_type": self.agent_type.value,
            "batch_processing": self.batch_processing,
        }
        if self.num_docs_used is not None:
            d["num_docs_used"] = self.num_docs_used
        if self.retrieved_docs:
            d["retrieved_docs"] = [doc.to_dict() for doc in self.retrieved_docs]
        if self.pipeline_log:
            d.update(self.pipeline_log.to_dict())
        return d
