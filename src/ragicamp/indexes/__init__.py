"""Index classes for storing and searching document embeddings.

Indexes are the reusable, expensive-to-build artifacts that store:
- Document chunks
- Embeddings
- FAISS indices

Retrievers are cheap strategy wrappers that use indexes.
"""

from ragicamp.indexes.base import Index
from ragicamp.indexes.embedding import EmbeddingIndex
from ragicamp.indexes.hierarchical import HierarchicalIndex

__all__ = [
    "Index",
    "EmbeddingIndex",
    "HierarchicalIndex",
]
