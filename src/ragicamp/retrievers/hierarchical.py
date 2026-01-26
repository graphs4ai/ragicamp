"""Hierarchical retriever with parent-child chunk structure.

Hierarchical retrieval addresses a fundamental tradeoff in RAG:
- Small chunks: Better for precise semantic matching
- Large chunks: Better context for the LLM to generate answers

Solution: Search with small "child" chunks, return larger "parent" chunks.

This gives you:
- Precise retrieval (small chunks match better)
- Rich context (large chunks have more information)
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ragicamp.core.logging import get_logger
from ragicamp.rag.chunking.hierarchical import HierarchicalChunker
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class HierarchicalRetriever(Retriever):
    """Retriever that searches child chunks but returns parent chunks.

    The retrieval flow:
    1. Documents are split into parent chunks (large, contextual)
    2. Parents are split into child chunks (small, precise)
    3. Child chunks are indexed for search
    4. Query matches against child embeddings
    5. Matching children are mapped back to parents
    6. Deduplicated parent chunks are returned

    This gives precise matching with rich context.
    """

    def __init__(
        self,
        name: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
        parent_overlap: int = 100,
        child_overlap: int = 50,
        **kwargs: Any,
    ):
        """Initialize hierarchical retriever.

        Args:
            name: Retriever identifier
            embedding_model: Model for embeddings
            parent_chunk_size: Size of parent chunks (returned to LLM)
            child_chunk_size: Size of child chunks (used for matching)
            parent_overlap: Overlap between parent chunks
            child_overlap: Overlap between child chunks
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.embedding_model_name = embedding_model
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

        # Initialize encoder
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Initialize chunker
        self.chunker = HierarchicalChunker(
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_overlap,
        )

        # Storage
        self.parent_docs: List[Document] = []
        self.child_docs: List[Document] = []
        self.child_to_parent: Dict[str, str] = {}
        self.parent_id_to_idx: Dict[str, int] = {}

        # FAISS index for child chunks
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents using hierarchical chunking.

        Args:
            documents: List of documents to index
        """
        logger.info(
            "Indexing %d documents with hierarchical chunking (parent=%d, child=%d)",
            len(documents),
            self.parent_chunk_size,
            self.child_chunk_size,
        )

        # Create hierarchical chunks
        self.parent_docs, self.child_docs, self.child_to_parent = self.chunker.chunk_documents(
            iter(documents)
        )

        # Build parent ID lookup
        self.parent_id_to_idx = {doc.id: idx for idx, doc in enumerate(self.parent_docs)}

        logger.info(
            "Created %d parent chunks and %d child chunks",
            len(self.parent_docs),
            len(self.child_docs),
        )

        if not self.child_docs:
            logger.warning("No child chunks created!")
            return

        # Embed child chunks
        child_texts = [doc.text for doc in self.child_docs]
        embeddings = self.encoder.encode(child_texts, show_progress_bar=True)

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to FAISS index
        self.index.add(embeddings.astype("float32"))

        logger.info("Hierarchical indexing complete")

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """Retrieve parent chunks by searching child chunks.

        Args:
            query: Search query
            top_k: Number of parent documents to return
            **kwargs: Additional parameters

        Returns:
            List of parent documents (with richer context)
        """
        if not self.child_docs:
            return []

        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search more children than needed (multiple children may map to same parent)
        num_children_to_search = top_k * 5

        # Search child index
        scores, indices = self.index.search(query_embedding, num_children_to_search)

        # Map children to parents and aggregate scores
        parent_scores: Dict[str, float] = {}
        parent_best_child: Dict[str, Document] = {}

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.child_docs):
                continue

            child_doc = self.child_docs[idx]
            parent_id = self.child_to_parent.get(child_doc.id)

            if parent_id is None:
                continue

            # Keep best score for each parent
            if parent_id not in parent_scores or score > parent_scores[parent_id]:
                parent_scores[parent_id] = float(score)
                parent_best_child[parent_id] = child_doc

        # Sort parents by score and return top-k
        sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for parent_id, score in sorted_parents[:top_k]:
            parent_idx = self.parent_id_to_idx.get(parent_id)
            if parent_idx is not None:
                parent_doc = self.parent_docs[parent_idx]
                # Create a copy with the score
                result_doc = Document(
                    id=parent_doc.id,
                    text=parent_doc.text,
                    metadata={
                        **parent_doc.metadata,
                        "matched_child": parent_best_child[parent_id].id,
                        "matched_child_text": parent_best_child[parent_id].text[:200],
                    },
                )
                result_doc.score = score
                results.append(result_doc)

        return results

    def save(self, artifact_name: str) -> str:
        """Save the hierarchical retriever.

        Args:
            artifact_name: Name for this retriever artifact

        Returns:
            Path where the artifact was saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Save FAISS index
        faiss.write_index(self.index, str(artifact_path / "child_index.faiss"))

        # Save documents and mappings
        with open(artifact_path / "parent_docs.pkl", "wb") as f:
            pickle.dump(self.parent_docs, f)

        with open(artifact_path / "child_docs.pkl", "wb") as f:
            pickle.dump(self.child_docs, f)

        with open(artifact_path / "child_to_parent.pkl", "wb") as f:
            pickle.dump(self.child_to_parent, f)

        # Save config
        config = {
            "name": self.name,
            "type": "hierarchical",
            "embedding_model": self.embedding_model_name,
            "parent_chunk_size": self.parent_chunk_size,
            "child_chunk_size": self.child_chunk_size,
            "num_parents": len(self.parent_docs),
            "num_children": len(self.child_docs),
            "embedding_dim": self.embedding_dim,
        }
        manager.save_json(config, artifact_path / "config.json")

        logger.info("Saved hierarchical retriever to: %s", artifact_path)
        return str(artifact_path)

    @classmethod
    def load(cls, artifact_name: str) -> "HierarchicalRetriever":
        """Load a previously saved hierarchical retriever.

        Args:
            artifact_name: Name of the retriever artifact to load

        Returns:
            Loaded HierarchicalRetriever instance
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Create retriever
        retriever = cls(
            name=config["name"],
            embedding_model=config["embedding_model"],
            parent_chunk_size=config["parent_chunk_size"],
            child_chunk_size=config["child_chunk_size"],
        )

        # Load FAISS index
        retriever.index = faiss.read_index(str(artifact_path / "child_index.faiss"))

        # Load documents and mappings
        with open(artifact_path / "parent_docs.pkl", "rb") as f:
            retriever.parent_docs = pickle.load(f)

        with open(artifact_path / "child_docs.pkl", "rb") as f:
            retriever.child_docs = pickle.load(f)

        with open(artifact_path / "child_to_parent.pkl", "rb") as f:
            retriever.child_to_parent = pickle.load(f)

        # Rebuild lookups
        retriever.parent_id_to_idx = {
            doc.id: idx for idx, doc in enumerate(retriever.parent_docs)
        }

        logger.info(
            "Loaded hierarchical retriever: %s (%d parents, %d children)",
            artifact_name,
            len(retriever.parent_docs),
            len(retriever.child_docs),
        )
        return retriever
