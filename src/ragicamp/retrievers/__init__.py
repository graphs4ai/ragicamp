"""Document retrieval systems."""

from ragicamp.retrievers.base import Document, Retriever
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.sparse import SparseRetriever

__all__ = [
    "Retriever",
    "Document",
    "DenseRetriever",
    "SparseRetriever",
]
