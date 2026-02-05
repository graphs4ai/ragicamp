"""Base classes for RAG agents.

Clean Architecture Design:
- Agents receive providers (not loaded models)
- Agents manage their own GPU lifecycle via context managers
- Each agent optimizes its own loading/batching strategy
- All intermediate steps are captured for analysis
- Simple interface: agent.run(queries) → results
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Core Data Types
# =============================================================================

@dataclass
class Query:
    """Input query for agent processing.
    
    Attributes:
        idx: Unique index for ordering and checkpointing
        text: The query text
        expected: Expected answers for evaluation (optional)
    """
    idx: int
    text: str
    expected: list[str] | None = None


@dataclass 
class Step:
    """Intermediate step in agent processing.
    
    Every operation (retrieval, generation, reranking, etc.) is logged
    for later analysis.
    """
    type: str  # "retrieve", "generate", "rerank", "hyde", "encode", etc.
    input: Any = None
    output: Any = None
    timing_ms: float = 0.0
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StepTimer:
    """Context manager for timing steps.
    
    Usage:
        with StepTimer("retrieve", model="bge-large") as step:
            docs = retriever.retrieve(query)
            step.output = docs
    """
    
    def __init__(self, step_type: str, model: str | None = None, **metadata):
        self.step = Step(type=step_type, model=model, metadata=metadata)
        self._start: float = 0.0
    
    def __enter__(self) -> Step:
        self._start = perf_counter()
        return self.step
    
    def __exit__(self, *args):
        self.step.timing_ms = (perf_counter() - self._start) * 1000


@dataclass
class RetrievedDocInfo:
    """Retrieved document info for logging.
    
    Captures all stages of retrieval pipeline for analysis.
    """
    rank: int  # Final rank (1-indexed)
    doc_id: str | None = None
    content: str | None = None  # Can be truncated for disk space
    score: float | None = None  # Final score
    retrieval_score: float | None = None  # Before reranking
    retrieval_rank: int | None = None  # Before reranking
    rerank_score: float | None = None  # Cross-encoder score
    
    def to_dict(self) -> dict[str, Any]:
        d = {"rank": self.rank}
        if self.doc_id is not None:
            d["doc_id"] = self.doc_id
        if self.content is not None:
            d["content"] = self.content
        if self.score is not None:
            d["score"] = self.score
        if self.retrieval_score is not None:
            d["retrieval_score"] = self.retrieval_score
        if self.retrieval_rank is not None:
            d["retrieval_rank"] = self.retrieval_rank
        if self.rerank_score is not None:
            d["rerank_score"] = self.rerank_score
        return d


@dataclass
class AgentResult:
    """Result from processing a single query.
    
    Contains the answer and all intermediate steps for analysis.
    
    Attributes:
        query: The input query
        answer: Generated answer
        steps: Pipeline steps with timing
        prompt: Full prompt sent to LLM
        retrieved_docs: Structured retrieved doc info (RAG only)
        metadata: Additional metadata (flexible)
    """
    query: Query
    answer: str
    steps: list[Step] = field(default_factory=list)
    prompt: str | None = None
    retrieved_docs: list[RetrievedDocInfo] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_content: bool = True, max_content_len: int = 500) -> dict[str, Any]:
        """Serialize for checkpointing/storage.
        
        Args:
            include_content: Whether to include document content (disk space tradeoff)
            max_content_len: Max chars per doc content (0 = full)
        """
        d = {
            "idx": self.query.idx,
            "question": self.query.text,
            "answer": self.answer,
            "expected": self.query.expected,
            "prompt": self.prompt,
            "steps": [
                {
                    "type": s.type,
                    "timing_ms": s.timing_ms,
                    "model": s.model,
                    **({"metadata": s.metadata} if s.metadata else {}),
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }
        
        # Add retrieved docs if present
        if self.retrieved_docs:
            docs_list = []
            for doc in self.retrieved_docs:
                doc_dict = doc.to_dict()
                # Optionally truncate content
                if "content" in doc_dict and include_content:
                    if max_content_len > 0 and len(doc_dict["content"]) > max_content_len:
                        doc_dict["content"] = doc_dict["content"][:max_content_len] + "..."
                elif not include_content:
                    doc_dict.pop("content", None)
                docs_list.append(doc_dict)
            d["retrieved_docs"] = docs_list
        
        return d


# =============================================================================
# Agent Base Class
# =============================================================================

class Agent(ABC):
    """Base class for all agents.
    
    Agents receive all queries and manage their own resources.
    Each agent type implements its optimal strategy:
    
    - FixedRAGAgent: batch_retrieve → unload_embedder → batch_generate
    - IterativeRAGAgent: per-query with multiple rounds
    - SelfRAGAgent: per-query with conditional retrieval
    - DirectLLMAgent: batch_generate only (no retrieval)
    """

    def __init__(self, name: str, **config):
        self.name = name
        self.config = config

    @abstractmethod
    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process all queries with agent-specific optimization.
        
        This is THE interface for running agents. Each agent type
        implements its own optimal strategy for:
        - Model loading/unloading (GPU optimization)
        - Batching strategy (throughput)
        - Checkpointing (resume capability)
        
        Args:
            queries: All queries to process
            on_result: Called after each result (for streaming/incremental save)
            checkpoint_path: Enables resume from crashes
            show_progress: Show progress bar
            
        Returns:
            Results with answers and all intermediate steps
        """
        ...

    def _load_checkpoint(self, path: Path) -> tuple[list[AgentResult], set[int]]:
        """Load checkpoint and return completed results."""
        import json
        
        if not path.exists():
            return [], set()
        
        with open(path) as f:
            data = json.load(f)
        
        results = []
        for r in data.get("results", []):
            query = Query(idx=r["idx"], text=r["query"], expected=r.get("expected"))
            result = AgentResult(
                query=query,
                answer=r["answer"],
                prompt=r.get("prompt"),
                metadata=r.get("metadata", {}),
            )
            results.append(result)
        
        completed_idx = {r.query.idx for r in results}
        logger.info("Loaded checkpoint: %d completed", len(completed_idx))
        return results, completed_idx

    def _save_checkpoint(self, results: list[AgentResult], path: Path) -> None:
        """Save checkpoint atomically."""
        import json
        
        data = {"results": [r.to_dict() for r in results]}
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
