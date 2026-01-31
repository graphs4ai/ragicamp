"""Document retrieval systems.

Architecture:
- Indexes (ragicamp.indexes): Store documents + embeddings (expensive, reusable)
- Retrievers (this module): Search strategies (cheap, configurable)

Retrievers are thin wrappers around Indexes. Multiple retrievers can share
the same index with different search parameters.
"""

from ragicamp.retrievers.base import Document, Retriever
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.hierarchical import HierarchicalRetriever
from ragicamp.retrievers.hybrid import HybridRetriever
from ragicamp.retrievers.sparse import SparseRetriever

__all__ = [
    "Retriever",
    "Document",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "HierarchicalRetriever",
]
