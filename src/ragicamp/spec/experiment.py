"""Experiment specification dataclass.

ExperimentSpec is an immutable configuration object that defines what to run.
It should be fully serializable and contain all the information needed to
reproduce an experiment.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable experiment configuration.

    This dataclass captures all parameters needed to run an experiment.
    It's frozen (immutable) to ensure configuration doesn't change during execution.

    Attributes:
        name: Unique experiment identifier
        exp_type: Type of experiment ('direct' or 'rag')
        model: Model specification string (e.g., 'hf:google/gemma-2-2b-it')
        dataset: Dataset name (e.g., 'nq', 'triviaqa')
        prompt: Prompt style name (e.g., 'default', 'cot')
        quant: Quantization setting ('4bit', '8bit', 'none')
        retriever: Retriever name (for RAG experiments)
        top_k: Number of documents to retrieve
        query_transform: Query transformation type ('hyde', 'multiquery', None)
        reranker: Reranker type (e.g., 'bge', 'ms-marco')
        reranker_model: Full reranker model name
        batch_size: Batch size for generation
        min_batch_size: Minimum batch size to reduce to on OOM
        metrics: List of metric names to compute
    """

    name: str
    exp_type: Literal["direct", "rag"]
    model: str
    dataset: str
    prompt: str
    quant: str = "4bit"
    retriever: Optional[str] = None
    top_k: int = 5
    query_transform: Optional[str] = None
    reranker: Optional[str] = None
    reranker_model: Optional[str] = None
    batch_size: int = 8
    min_batch_size: int = 1
    metrics: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate spec after initialization."""
        if self.exp_type == "rag" and not self.retriever:
            raise ValueError("RAG experiments require a retriever")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "exp_type": self.exp_type,
            "model": self.model,
            "dataset": self.dataset,
            "prompt": self.prompt,
            "quant": self.quant,
            "retriever": self.retriever,
            "top_k": self.top_k,
            "query_transform": self.query_transform,
            "reranker": self.reranker,
            "reranker_model": self.reranker_model,
            "batch_size": self.batch_size,
            "min_batch_size": self.min_batch_size,
            "metrics": list(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            exp_type=data["exp_type"],
            model=data["model"],
            dataset=data["dataset"],
            prompt=data["prompt"],
            quant=data.get("quant", "4bit"),
            retriever=data.get("retriever"),
            top_k=data.get("top_k", 5),
            query_transform=data.get("query_transform"),
            reranker=data.get("reranker"),
            reranker_model=data.get("reranker_model"),
            batch_size=data.get("batch_size", 8),
            min_batch_size=data.get("min_batch_size", 1),
            metrics=data.get("metrics", []),
        )
