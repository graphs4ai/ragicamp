"""Index classes for storing and searching document embeddings.

Indexes are the reusable, expensive-to-build artifacts that store:
- Document chunks
- Embeddings
- FAISS indices

Retrievers are cheap strategy wrappers that use indexes.

Index building utilities are in ragicamp.indexes.builder.
"""

from ragicamp.indexes.base import Index
from ragicamp.indexes.builder import (
    build_embedding_index,
    build_hierarchical_index,
    build_retriever_from_index,
    ensure_indexes_exist,
    get_embedding_index_name,
)
from ragicamp.indexes.embedding import EmbeddingIndex
from ragicamp.indexes.hierarchical import HierarchicalIndex

__all__ = [
    "Index",
    "EmbeddingIndex",
    "HierarchicalIndex",
    # Builder utilities
    "build_embedding_index",
    "build_hierarchical_index",
    "build_retriever_from_index",
    "ensure_indexes_exist",
    "get_embedding_index_name",
]
