"""Embedding index for dense retrieval."""

import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ragicamp.core.logging import get_logger
from ragicamp.indexes.base import Index
from ragicamp.retrievers.base import Document
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


class EmbeddingIndex(Index):
    """Index storing document embeddings in a FAISS index.

    This is the core reusable index for dense retrieval. It stores:
    - Document chunks
    - Their embeddings
    - A FAISS index for similarity search

    Multiple retrievers can share the same EmbeddingIndex:
    - DenseRetriever: direct similarity search
    - HybridRetriever: combines with BM25
    """

    def __init__(
        self,
        name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        **kwargs: Any,
    ):
        """Initialize embedding index.

        Args:
            name: Index identifier
            embedding_model: Sentence transformer model name
            index_type: FAISS index type (flat, ivf, hnsw)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.embedding_model_name = embedding_model
        self.index_type = index_type

        # Lazy load encoder (only when needed)
        self._encoder: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

        # Storage
        self.documents: list[Document] = []
        self.index: Optional[faiss.Index] = None

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy load the encoder with Flash Attention (if available) and torch.compile()."""
        if self._encoder is None:
            # Try Flash Attention 2 if available, otherwise fall back to default
            model_kwargs = {}
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 available, enabling for embeddings")
            except ImportError:
                logger.debug("flash-attn not installed, using default attention")

            self._encoder = SentenceTransformer(
                self.embedding_model_name,
                trust_remote_code=True,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()

            # Apply torch.compile() for additional speedup (PyTorch 2.0+)
            try:
                import torch

                if hasattr(torch, "compile") and torch.cuda.is_available():
                    self._encoder = torch.compile(self._encoder, mode="reduce-overhead")
                    logger.info("Applied torch.compile() to embedding model")
            except Exception as e:
                logger.debug("torch.compile() not applied: %s", e)

        return self._encoder

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            _ = self.encoder  # Force load
        return self._embedding_dim

    def _create_faiss_index(self) -> faiss.Index:
        """Create a new FAISS index based on index_type."""
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def build(self, documents: list[Document]) -> None:
        """Build index from documents.

        Args:
            documents: List of documents to index
        """
        self.documents = documents
        logger.info("Building embedding index for %d documents", len(documents))

        # Compute embeddings
        texts = [doc.text for doc in documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create and populate FAISS index
        self.index = self._create_faiss_index()

        if self.index_type == "ivf":
            self.index.train(embeddings.astype("float32"))

        self.index.add(embeddings.astype("float32"))
        logger.info("Index built with %d vectors", self.index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Search the index.

        Args:
            query_embedding: Query vector (already normalized)
            top_k: Number of results

        Returns:
            List of (document_idx, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.documents):
                results.append((int(idx), float(score)))

        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string to embedding.

        Args:
            query: Query text

        Returns:
            Normalized query embedding
        """
        embedding = self.encoder.encode([query]).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding[0]

    def batch_encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode multiple queries at once (much faster than sequential).

        Args:
            queries: List of query texts

        Returns:
            Normalized query embeddings, shape (n_queries, embedding_dim)
        """
        embeddings = self.encoder.encode(queries, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def batch_search(
        self, query_embeddings: np.ndarray, top_k: int = 10
    ) -> list[list[tuple[int, float]]]:
        """Search the index with multiple queries at once.

        Args:
            query_embeddings: Query vectors, shape (n_queries, embedding_dim)
            top_k: Number of results per query

        Returns:
            List of (document_idx, score) tuples for each query
        """
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        scores, indices = self.index.search(query_embeddings.astype("float32"), top_k)

        all_results = []
        for i in range(len(query_embeddings)):
            results = []
            for idx, score in zip(indices[i], scores[i]):
                if 0 <= idx < len(self.documents):
                    results.append((int(idx), float(score)))
            all_results.append(results)

        return all_results

    def get_document(self, idx: int) -> Optional[Document]:
        """Get document by index."""
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None

    def __len__(self) -> int:
        return len(self.documents)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save index to disk.

        Args:
            path: Optional custom path

        Returns:
            Path where saved
        """
        manager = get_artifact_manager()

        if path is None:
            path = manager.get_embedding_index_path(self.name)
        else:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save documents
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Save config
        config = {
            "name": self.name,
            "type": "embedding",
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents),
        }
        manager.save_json(config, path / "config.json")

        logger.info("Saved embedding index to: %s", path)
        return path

    @classmethod
    def load(cls, name: str, path: Optional[Path] = None) -> "EmbeddingIndex":
        """Load index from disk.

        Args:
            name: Index name
            path: Optional custom path

        Returns:
            Loaded EmbeddingIndex
        """
        manager = get_artifact_manager()

        if path is None:
            path = manager.get_embedding_index_path(name)
        else:
            path = Path(path)

        # Load config
        config = manager.load_json(path / "config.json")

        # Create index without loading encoder (lazy)
        index = cls(
            name=config["name"],
            embedding_model=config["embedding_model"],
            index_type=config.get("index_type", "flat"),
        )
        index._embedding_dim = config.get("embedding_dim")

        # Load FAISS index
        index.index = faiss.read_index(str(path / "index.faiss"))

        # Load documents
        with open(path / "documents.pkl", "rb") as f:
            index.documents = pickle.load(f)

        logger.info("Loaded embedding index: %s (%d docs)", name, len(index.documents))
        return index
