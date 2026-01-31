"""Index builders for creating various types of indexes.

This package provides specialized builders for:
- Embedding indexes: Dense vector indexes for semantic search
- Hierarchical indexes: Parent-child chunk relationships
"""

from ragicamp.indexes.builders.embedding_builder import build_embedding_index
from ragicamp.indexes.builders.hierarchical_builder import build_hierarchical_index

__all__ = [
    "build_embedding_index",
    "build_hierarchical_index",
]
