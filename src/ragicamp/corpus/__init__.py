"""Document corpus module for RAG systems.

This module provides abstractions for document sources used in retrieval.
Important: Corpora provide documents WITHOUT answer information.
"""

from ragicamp.corpus.base import CorpusConfig, DocumentCorpus
from ragicamp.corpus.wikipedia import WikipediaCorpus

__all__ = [
    "DocumentCorpus",
    "CorpusConfig",
    "WikipediaCorpus",
]
