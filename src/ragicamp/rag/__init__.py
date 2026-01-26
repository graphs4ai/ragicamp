"""Modular RAG pipeline components.

This module provides a composable RAG pipeline with:
- Query transformers (HyDE, Multi-query)
- Rerankers (Cross-encoder)
- Advanced chunking (Semantic, Hierarchical)
"""

from ragicamp.rag.pipeline import RAGPipeline

__all__ = [
    "RAGPipeline",
]
