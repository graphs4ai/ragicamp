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

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum


# =============================================================================
# Enums
# =============================================================================


class AgentType(str, Enum):
    """Types of RAG agents."""
    DIRECT_LLM = "direct_llm"
    FIXED_RAG = "fixed_rag"
    # Future: ADAPTIVE_RAG, MDP_RAG, etc.


class PromptStyle(str, Enum):
    """Prompt styles available."""
    DEFAULT = "default"
    CONCISE = "concise"
    FEWSHOT = "fewshot"
    FEWSHOT_3 = "fewshot_3"
    FEWSHOT_1 = "fewshot_1"


# =============================================================================
# Retrieved Documents
# =============================================================================


@dataclass
class RetrievedDoc:
    """A single retrieved document.
    
    This is the canonical format for retrieved documents throughout the system.
    Saved in predictions.json as part of RAG experiments.
    
    Attributes:
        rank: Position in retrieval results (1-indexed)
        content: Document text content
        score: Retrieval similarity score (optional)
        source: Source identifier (optional, e.g., "wikipedia")
        doc_id: Document ID in corpus (optional)
    """
    rank: int
    content: str
    score: Optional[float] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {"rank": self.rank, "content": self.content}
        if self.score is not None:
            d["score"] = self.score
        if self.source is not None:
            d["source"] = self.source
        if self.doc_id is not None:
            d["doc_id"] = self.doc_id
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievedDoc":
        """Create from dict (for loading from JSON)."""
        return cls(
            rank=d["rank"],
            content=d["content"],
            score=d.get("score"),
            source=d.get("source"),
            doc_id=d.get("doc_id"),
        )


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
    expected: List[str]
    prompt: str
    retrieved_docs: Optional[List[RetrievedDoc]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRecord":
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
    
    Attributes:
        agent_type: Type of agent that generated the response
        batch_processing: Whether this was part of a batch
        num_docs_used: Number of documents used (RAG only)
        retrieved_docs: Structured retrieved documents (RAG only)
    """
    agent_type: AgentType
    batch_processing: bool = False
    num_docs_used: Optional[int] = None
    retrieved_docs: Optional[List[RetrievedDoc]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backwards compatibility."""
        d = {
            "agent_type": self.agent_type.value,
            "batch_processing": self.batch_processing,
        }
        if self.num_docs_used is not None:
            d["num_docs_used"] = self.num_docs_used
        if self.retrieved_docs:
            d["retrieved_docs"] = [doc.to_dict() for doc in self.retrieved_docs]
        return d


# =============================================================================
# Experiment Specification
# =============================================================================


@dataclass
class ExperimentSpec:
    """Full specification for an experiment.
    
    This replaces the ExpSpec namedtuple in study.py with a proper dataclass.
    
    Attributes:
        name: Unique experiment identifier
        exp_type: "direct" or "rag"
        model: Model spec (e.g., "hf:google/gemma-2b-it")
        dataset: Dataset name (e.g., "nq", "hotpotqa")
        prompt_style: Prompt style to use
        quantization: Model quantization ("4bit", "8bit", "none")
        retriever: Retriever name (RAG only)
        top_k: Number of docs to retrieve (RAG only)
        batch_size: Batch size for generation
        min_batch_size: Minimum batch size (for auto-reduction)
    """
    name: str
    exp_type: str  # "direct" or "rag"
    model: str
    dataset: str
    prompt_style: PromptStyle = PromptStyle.DEFAULT
    quantization: str = "4bit"
    retriever: Optional[str] = None
    top_k: int = 5
    batch_size: int = 32
    min_batch_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "name": self.name,
            "exp_type": self.exp_type,
            "model": self.model,
            "dataset": self.dataset,
            "prompt_style": self.prompt_style.value,
            "quantization": self.quantization,
            "retriever": self.retriever,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "min_batch_size": self.min_batch_size,
        }
