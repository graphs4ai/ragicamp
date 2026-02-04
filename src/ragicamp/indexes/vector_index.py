"""Vector index - just data (FAISS index + documents).

The index stores:
- Document chunks
- FAISS index for similarity search

It does NOT own the embedder. Embeddings are provided externally.
This allows clean lifecycle management of embedder models.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """A document chunk with metadata."""
    id: str
    text: str
    metadata: dict[str, Any] | None = None


@dataclass
class SearchResult:
    """Result from index search."""
    doc_idx: int
    score: float
    document: Document


class VectorIndex:
    """FAISS-based vector index - just data, no models.
    
    This is a clean separation of concerns:
    - VectorIndex: stores vectors and documents, handles search
    - Embedder: converts text to vectors (managed separately)
    
    Usage:
        # Build index with external embedder
        with embedder_provider.load() as embedder:
            embeddings = embedder.batch_encode(texts)
            index = VectorIndex.build(documents, embeddings)
            index.save("my_index")
        
        # Search with external embedder
        with embedder_provider.load() as embedder:
            query_emb = embedder.batch_encode([query])[0]
            results = index.search(query_emb, top_k=5)
    """
    
    def __init__(
        self,
        faiss_index: faiss.Index,
        documents: list[Document],
        embedding_dim: int,
        index_type: str = "flat",
    ):
        self.faiss_index = faiss_index
        self.documents = documents
        self.embedding_dim = embedding_dim
        self.index_type = index_type
    
    @classmethod
    def build(
        cls,
        documents: list[Document],
        embeddings: np.ndarray,
        index_type: str = "flat",
    ) -> "VectorIndex":
        """Build index from documents and their embeddings.
        
        Args:
            documents: List of documents
            embeddings: Pre-computed embeddings, shape (n_docs, dim)
            index_type: "flat", "ivf", or "hnsw"
        
        Returns:
            VectorIndex ready for search
        """
        n_docs, dim = embeddings.shape
        
        if len(documents) != n_docs:
            raise ValueError(f"Documents ({len(documents)}) != embeddings ({n_docs})")
        
        # Normalize embeddings for inner product search
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        if index_type == "hnsw":
            faiss_index = faiss.IndexHNSWFlat(dim, 32)
            faiss_index.hnsw.efConstruction = 200
            faiss_index.hnsw.efSearch = 128
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(4096, n_docs // 10)
            faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            faiss_index.train(embeddings)
            faiss_index.nprobe = 128
        else:  # flat
            faiss_index = faiss.IndexFlatIP(dim)
        
        faiss_index.add(embeddings)
        
        logger.info("Built %s index: %d vectors, dim=%d", index_type, n_docs, dim)
        return cls(faiss_index, documents, dim, index_type)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector (will be normalized)
            top_k: Number of results
        
        Returns:
            List of SearchResult with document and score
        """
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        scores, indices = self.faiss_index.search(query, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.documents):
                results.append(SearchResult(
                    doc_idx=int(idx),
                    score=float(score),
                    document=self.documents[idx],
                ))
        
        return results
    
    def batch_search(
        self, 
        query_embeddings: np.ndarray, 
        top_k: int = 10,
    ) -> list[list[SearchResult]]:
        """Search for multiple queries at once.
        
        Args:
            query_embeddings: Query vectors, shape (n_queries, dim)
            top_k: Number of results per query
        
        Returns:
            List of results for each query
        """
        queries = query_embeddings.astype(np.float32)
        faiss.normalize_L2(queries)
        
        scores, indices = self.faiss_index.search(queries, top_k)
        
        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for idx, score in zip(indices[q_idx], scores[q_idx]):
                if 0 <= idx < len(self.documents):
                    results.append(SearchResult(
                        doc_idx=int(idx),
                        score=float(score),
                        document=self.documents[idx],
                    ))
            all_results.append(results)
        
        return all_results
    
    def save(self, path: Path | str) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(path / "index.faiss"))
        
        # Save documents
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        import json
        metadata = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "n_documents": len(self.documents),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Saved index to %s", path)
    
    @classmethod
    def load(cls, path: Path | str, use_mmap: bool = True) -> "VectorIndex":
        """Load index from disk.
        
        Args:
            path: Index directory
            use_mmap: Memory-map the index (reduces RAM for large indexes)
        """
        path = Path(path)
        
        # Load metadata
        import json
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        
        # Load FAISS index
        io_flags = faiss.IO_FLAG_MMAP if use_mmap else 0
        faiss_index = faiss.read_index(str(path / "index.faiss"), io_flags)
        
        # Load documents
        with open(path / "documents.pkl", "rb") as f:
            documents = pickle.load(f)
        
        logger.info("Loaded index from %s (%d docs, mmap=%s)", 
                   path, len(documents), use_mmap)
        
        return cls(
            faiss_index=faiss_index,
            documents=documents,
            embedding_dim=metadata["embedding_dim"],
            index_type=metadata["index_type"],
        )
    
    def __len__(self) -> int:
        return len(self.documents)
