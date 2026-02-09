"""Experiment specification dataclass.

ExperimentSpec is an immutable configuration object that defines what to run.
It should be fully serializable and contain all the information needed to
reproduce an experiment.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


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
        retriever: Retriever name (for RAG experiments)
        embedding_index: Actual index directory name (if different from retriever name)
        sparse_index: Sparse index name for hybrid retrieval
        top_k: Number of documents to use for generation (after reranking if applicable)
        fetch_k: Number of documents to retrieve before reranking (None = auto)
        query_transform: Query transformation type ('hyde', 'multiquery', None)
        reranker: Reranker type (e.g., 'bge', 'ms-marco')
        reranker_model: Full reranker model name
        batch_size: Batch size for generation
        metrics: List of metric names to compute
        agent_type: Explicit agent type (e.g., 'vanilla_rag', 'pipeline_rag', 'iterative_rag')
        hypothesis: Documentation of what this experiment tests (for singleton experiments)
        agent_params: Agent-specific parameters as tuple of (key, value) pairs
    """

    name: str
    exp_type: Literal["direct", "rag"]
    model: str
    dataset: str
    prompt: str
    retriever: Optional[str] = None
    embedding_index: Optional[str] = None  # Actual index path (if different from retriever)
    sparse_index: Optional[str] = None  # For hybrid retrieval
    top_k: int = 5
    fetch_k: Optional[int] = None
    query_transform: Optional[str] = None
    reranker: Optional[str] = None
    reranker_model: Optional[str] = None
    batch_size: int = 8
    metrics: list[str] = field(default_factory=list)
    # Singleton experiment fields
    agent_type: Optional[str] = None
    hypothesis: Optional[str] = None
    agent_params: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    _VALID_QUERY_TRANSFORMS = {None, "none", "hyde", "multiquery"}

    def __post_init__(self) -> None:
        """Validate spec after initialization."""
        if self.exp_type == "rag" and not self.retriever:
            raise ValueError("RAG experiments require a retriever")
        if self.query_transform not in self._VALID_QUERY_TRANSFORMS:
            raise ValueError(
                f"query_transform must be one of {self._VALID_QUERY_TRANSFORMS}, "
                f"got '{self.query_transform}'"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "name": self.name,
            "exp_type": self.exp_type,
            "model": self.model,
            "dataset": self.dataset,
            "prompt": self.prompt,
            "retriever": self.retriever,
            "embedding_index": self.embedding_index,
            "sparse_index": self.sparse_index,
            "top_k": self.top_k,
            "fetch_k": self.fetch_k,
            "query_transform": self.query_transform,
            "reranker": self.reranker,
            "reranker_model": self.reranker_model,
            "batch_size": self.batch_size,
            "metrics": list(self.metrics),
        }
        # Only include optional singleton fields if set
        if self.agent_type:
            d["agent_type"] = self.agent_type
        if self.hypothesis:
            d["hypothesis"] = self.hypothesis
        if self.agent_params:
            d["agent_params"] = dict(self.agent_params)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentSpec":
        """Create from dictionary."""
        # Convert agent_params dict to tuple of tuples
        agent_params_dict = data.get("agent_params", {})
        agent_params = tuple(agent_params_dict.items()) if agent_params_dict else ()

        return cls(
            name=data["name"],
            exp_type=data["exp_type"],
            model=data["model"],
            dataset=data["dataset"],
            prompt=data["prompt"],
            retriever=data.get("retriever"),
            embedding_index=data.get("embedding_index"),
            sparse_index=data.get("sparse_index"),
            top_k=data.get("top_k", 5),
            fetch_k=data.get("fetch_k"),
            query_transform=data.get("query_transform"),
            reranker=data.get("reranker"),
            reranker_model=data.get("reranker_model"),
            batch_size=data.get("batch_size", 8),
            metrics=data.get("metrics", []),
            agent_type=data.get("agent_type"),
            hypothesis=data.get("hypothesis"),
            agent_params=agent_params,
        )

    def get_agent_params_dict(self) -> dict[str, Any]:
        """Get agent_params as a dictionary for convenience."""
        return dict(self.agent_params)
