"""Index classes for storing and searching document embeddings.

New architecture:
- VectorIndex: just data (FAISS + documents), no model ownership
- Legacy: EmbeddingIndex, HierarchicalIndex (still supported)

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
from ragicamp.indexes.sparse import SparseIndex, SparseMethod, get_sparse_index_name
from ragicamp.indexes.vector_index import Document, SearchResult, VectorIndex

__all__ = [
    # New architecture (clean separation)
    "VectorIndex",
    "Document",
    "SearchResult",
    # Legacy indexes
    "Index",
    "EmbeddingIndex",
    "HierarchicalIndex",
    "SparseIndex",
    "SparseMethod",
    # Builder utilities
    "build_embedding_index",
    "build_hierarchical_index",
    "build_retriever_from_index",
    "ensure_indexes_exist",
    "get_embedding_index_name",
    "get_sparse_index_name",
]
