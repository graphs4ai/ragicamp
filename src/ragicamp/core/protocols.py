"""Protocol definitions for RAGiCamp.

Protocols define interfaces that components must implement.
Using @runtime_checkable allows isinstance() checks.

Usage:
    from ragicamp.core.protocols import HasGenerate
    
    def evaluate_model(model: HasGenerate):
        if not isinstance(model, HasGenerate):
            raise TypeError("Model must implement generate()")
        return model.generate(prompt)
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)


# === Model Protocols ===

@runtime_checkable
class HasGenerate(Protocol):
    """Protocol for objects that can generate text.
    
    Implemented by: LanguageModel, HuggingFaceModel, OpenAIModel
    """
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate text from a prompt."""
        ...


@runtime_checkable
class HasEmbeddings(Protocol):
    """Protocol for objects that can produce embeddings.
    
    Implemented by: LanguageModel, EmbeddingModel
    """
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        ...


@runtime_checkable
class HasTokenCount(Protocol):
    """Protocol for objects that can count tokens.
    
    Implemented by: LanguageModel
    """
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...


# === Retriever Protocols ===

@runtime_checkable
class HasRetrieve(Protocol):
    """Protocol for objects that can retrieve documents.
    
    Implemented by: Retriever, DenseRetriever, SparseRetriever
    """
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Any]:  # List[Document]
        """Retrieve relevant documents for a query."""
        ...


@runtime_checkable
class HasIndex(Protocol):
    """Protocol for objects that can index documents.
    
    Implemented by: Retriever
    """
    
    def index_documents(self, documents: List[Any]) -> None:
        """Index a collection of documents."""
        ...


# === Agent Protocols ===

@runtime_checkable
class HasAnswer(Protocol):
    """Protocol for objects that can answer questions.
    
    Implemented by: RAGAgent, DirectLLMAgent, FixedRAGAgent
    """
    
    def answer(self, query: str, **kwargs: Any) -> Any:  # RAGResponse
        """Generate an answer for a query."""
        ...


@runtime_checkable
class HasBatchAnswer(Protocol):
    """Protocol for objects that support batch answering.
    
    Implemented by: RAGAgent (with batch support)
    """
    
    def batch_answer(
        self,
        queries: List[str],
        **kwargs: Any,
    ) -> List[Any]:  # List[RAGResponse]
        """Generate answers for multiple queries."""
        ...


# === Metric Protocols ===

@runtime_checkable
class HasCompute(Protocol):
    """Protocol for objects that can compute metrics.
    
    Implemented by: Metric, ExactMatchMetric, BERTScoreMetric
    """
    
    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute metric scores."""
        ...


@runtime_checkable  
class HasComputeSingle(Protocol):
    """Protocol for objects that can compute single metrics.
    
    Implemented by: Metric
    """
    
    def compute_single(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute metric for a single prediction."""
        ...


# === Dataset Protocols ===

@runtime_checkable
class HasIterate(Protocol):
    """Protocol for objects that can be iterated.
    
    Implemented by: QADataset
    """
    
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int): ...


# === Persistence Protocols ===

@runtime_checkable
class HasSave(Protocol):
    """Protocol for objects that can be saved.
    
    Implemented by: Retriever, Agent
    """
    
    def save(self, path: str, **kwargs: Any) -> str:
        """Save to disk."""
        ...


@runtime_checkable
class HasLoad(Protocol):
    """Protocol for objects that can be loaded.
    
    Implemented by: Retriever, Agent
    """
    
    @classmethod
    def load(cls, path: str, **kwargs: Any) -> Any:
        """Load from disk."""
        ...


# === State Protocols ===

@runtime_checkable
class HasReset(Protocol):
    """Protocol for objects with resettable state.
    
    Implemented by: RAGAgent
    """
    
    def reset(self) -> None:
        """Reset internal state."""
        ...


# === Utility Functions ===

def check_implements(obj: Any, protocol: type) -> bool:
    """Check if an object implements a protocol.
    
    Args:
        obj: Object to check
        protocol: Protocol class to check against
        
    Returns:
        True if object implements protocol
        
    Example:
        >>> if not check_implements(model, HasGenerate):
        ...     raise TypeError("Model must implement generate()")
    """
    return isinstance(obj, protocol)


def require_implements(obj: Any, protocol: type, name: str = "Object") -> None:
    """Require that an object implements a protocol.
    
    Args:
        obj: Object to check
        protocol: Protocol class to check against
        name: Name to use in error message
        
    Raises:
        TypeError: If object doesn't implement protocol
        
    Example:
        >>> require_implements(model, HasGenerate, "model")
    """
    if not isinstance(obj, protocol):
        protocol_name = protocol.__name__
        obj_type = type(obj).__name__
        raise TypeError(
            f"{name} must implement {protocol_name}, got {obj_type}"
        )
