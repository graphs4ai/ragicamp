"""Embedding index for dense retrieval.

Supports both CPU and GPU FAISS indexes for high-performance similarity search.
GPU FAISS can provide 10-100x speedup for large indexes.
"""

import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.indexes.base import Index
from ragicamp.retrievers.base import Document
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)

# Track FAISS GPU resources globally to avoid re-initialization
_faiss_gpu_resources: Optional["faiss.StandardGpuResources"] = None
_faiss_gpu_available: Optional[bool] = None


def _check_faiss_gpu_available() -> bool:
    """Check if FAISS GPU is available."""
    global _faiss_gpu_available
    if _faiss_gpu_available is None:
        try:
            import faiss.contrib.torch_utils  # noqa: F401

            # Check if GPU resources can be created
            res = faiss.StandardGpuResources()
            del res
            _faiss_gpu_available = True
            logger.info("FAISS GPU support detected")
        except Exception as e:
            _faiss_gpu_available = False
            logger.debug("FAISS GPU not available: %s", e)
    return _faiss_gpu_available


def get_faiss_gpu_resources(temp_memory_mb: int = None) -> Optional["faiss.StandardGpuResources"]:
    """Get or create shared FAISS GPU resources.

    Uses a singleton pattern to avoid multiple GPU resource allocations.

    Args:
        temp_memory_mb: Temporary memory in MB for FAISS operations.
                       Defaults to Defaults.FAISS_GPU_TEMP_MEMORY_MB.

    Returns:
        FAISS GPU resources or None if GPU not available.
    """
    global _faiss_gpu_resources

    if not _check_faiss_gpu_available():
        return None

    if _faiss_gpu_resources is None:
        try:
            _faiss_gpu_resources = faiss.StandardGpuResources()
            temp_mem = (temp_memory_mb or Defaults.FAISS_GPU_TEMP_MEMORY_MB) * 1024 * 1024
            _faiss_gpu_resources.setTempMemory(temp_mem)
            logger.info("Initialized FAISS GPU resources (temp_memory=%dMB)", temp_memory_mb or Defaults.FAISS_GPU_TEMP_MEMORY_MB)
        except Exception as e:
            logger.warning("Failed to initialize FAISS GPU resources: %s", e)
            return None

    return _faiss_gpu_resources


def release_faiss_gpu_resources() -> None:
    """Release FAISS GPU resources to free memory."""
    global _faiss_gpu_resources
    if _faiss_gpu_resources is not None:
        del _faiss_gpu_resources
        _faiss_gpu_resources = None
        logger.info("Released FAISS GPU resources")


class EmbeddingIndex(Index):
    """Index storing document embeddings in a FAISS index.

    This is the core reusable index for dense retrieval. It stores:
    - Document chunks
    - Their embeddings
    - A FAISS index for similarity search

    Multiple retrievers can share the same EmbeddingIndex:
    - DenseRetriever: direct similarity search
    - HybridRetriever: combines with BM25

    Supports GPU acceleration for 10-100x faster similarity search.
    """

    def __init__(
        self,
        name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        use_gpu: Optional[bool] = None,
        nlist: int = None,
        nprobe: int = None,
        **kwargs: Any,
    ):
        """Initialize embedding index.

        Args:
            name: Index identifier
            embedding_model: Sentence transformer model name
            index_type: FAISS index type (flat, ivf, ivfpq, hnsw)
            use_gpu: Whether to use GPU FAISS. If None, uses Defaults.FAISS_USE_GPU.
            nlist: Number of clusters for IVF indexes. Defaults to FAISS_IVF_NLIST.
            nprobe: Number of clusters to search for IVF. Defaults to FAISS_IVF_NPROBE.
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.use_gpu = use_gpu if use_gpu is not None else Defaults.FAISS_USE_GPU
        self.nlist = nlist or Defaults.FAISS_IVF_NLIST
        self.nprobe = nprobe or Defaults.FAISS_IVF_NPROBE
        self._is_gpu_index = False

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

    def _create_faiss_index(self, num_vectors: int = 0) -> faiss.Index:
        """Create a new FAISS index based on index_type.

        Args:
            num_vectors: Expected number of vectors (used for IVF nlist tuning).

        Returns:
            FAISS index (CPU or GPU depending on configuration).
        """
        dim = self.embedding_dim

        # Determine nlist for IVF indexes (rule of thumb: sqrt(n) to 4*sqrt(n))
        if num_vectors > 0:
            nlist = min(self.nlist, max(64, int(np.sqrt(num_vectors) * 2)))
        else:
            nlist = self.nlist

        # Create CPU index first
        if self.index_type == "flat":
            cpu_index = faiss.IndexFlatIP(dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        elif self.index_type == "ivfpq":
            # Product Quantization for memory efficiency + speed
            # 32 subquantizers, 8 bits per subquantizer
            quantizer = faiss.IndexFlatIP(dim)
            m = min(32, dim // 4)  # subquantizers (must divide dim)
            cpu_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
        elif self.index_type == "hnsw":
            # HNSW is very fast for CPU, doesn't benefit as much from GPU
            cpu_index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
            cpu_index.hnsw.efConstruction = 200  # Higher = better quality, slower build
            cpu_index.hnsw.efSearch = 128  # Higher = better recall, slower search
        else:
            raise ValueError(f"Unknown index type: {self.index_type}. "
                           f"Valid types: flat, ivf, ivfpq, hnsw")

        # Move to GPU if requested and available
        if self.use_gpu and self.index_type != "hnsw":  # HNSW doesn't support GPU
            gpu_res = get_faiss_gpu_resources()
            if gpu_res is not None:
                try:
                    # Configure GPU index options
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = True  # Use FP16 for 2x memory efficiency
                    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index, co)
                    self._is_gpu_index = True
                    logger.info("Created GPU FAISS index (type=%s, nlist=%d)", self.index_type, nlist)
                    return gpu_index
                except Exception as e:
                    logger.warning("Failed to create GPU index, falling back to CPU: %s", e)

        logger.info("Created CPU FAISS index (type=%s, nlist=%d)", self.index_type, nlist)
        return cpu_index

    def _set_search_params(self) -> None:
        """Set optimal search parameters for the index type."""
        if self.index is None:
            return

        # Get the underlying CPU index if on GPU
        index = self.index
        if self._is_gpu_index:
            # For GPU index, we need to access parameters differently
            pass  # nprobe is set via search params for GPU

        # Set nprobe for IVF indexes
        if hasattr(index, 'nprobe'):
            index.nprobe = self.nprobe
            logger.debug("Set nprobe=%d for IVF index", self.nprobe)

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
        embeddings = embeddings.astype("float32")

        # Create FAISS index with knowledge of dataset size
        self.index = self._create_faiss_index(num_vectors=len(documents))

        # Train index if required (IVF variants need training)
        if self.index_type in ("ivf", "ivfpq"):
            logger.info("Training IVF index with %d vectors...", len(embeddings))
            # For GPU index, we may need to train on CPU first
            if self._is_gpu_index:
                # Create temporary CPU index for training
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                cpu_index.train(embeddings)
                # Copy trained parameters back to GPU
                gpu_res = get_faiss_gpu_resources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index, co)
            else:
                self.index.train(embeddings)

        # Add vectors to index
        self.index.add(embeddings)

        # Set search parameters (nprobe for IVF)
        self._set_search_params()

        logger.info("Index built with %d vectors (GPU=%s)", self.index.ntotal, self._is_gpu_index)

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

        Note: GPU indexes are converted to CPU for saving.

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

        # Convert GPU index to CPU for saving (GPU indexes can't be serialized directly)
        index_to_save = self.index
        if self._is_gpu_index:
            logger.info("Converting GPU index to CPU for saving...")
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        # Save FAISS index
        faiss.write_index(index_to_save, str(path / "index.faiss"))

        # Save documents
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Save config (include GPU settings for reload)
        config = {
            "name": self.name,
            "type": "embedding",
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents),
            "use_gpu": self.use_gpu,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }
        manager.save_json(config, path / "config.json")

        logger.info("Saved embedding index to: %s", path)
        return path

    @classmethod
    def load(
        cls,
        name: str,
        path: Optional[Path] = None,
        use_gpu: Optional[bool] = None,
    ) -> "EmbeddingIndex":
        """Load index from disk.

        Args:
            name: Index name
            path: Optional custom path
            use_gpu: Whether to load on GPU. If None, uses saved config or default.

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

        # Determine GPU usage: explicit param > saved config > default
        if use_gpu is None:
            use_gpu = config.get("use_gpu", Defaults.FAISS_USE_GPU)

        # Create index without loading encoder (lazy)
        index = cls(
            name=config["name"],
            embedding_model=config["embedding_model"],
            index_type=config.get("index_type", "flat"),
            use_gpu=use_gpu,
            nlist=config.get("nlist", Defaults.FAISS_IVF_NLIST),
            nprobe=config.get("nprobe", Defaults.FAISS_IVF_NPROBE),
        )
        index._embedding_dim = config.get("embedding_dim")

        # Load FAISS index (always saved as CPU)
        cpu_index = faiss.read_index(str(path / "index.faiss"))

        # Move to GPU if requested and supported
        index_type = config.get("index_type", "flat")
        if use_gpu and index_type != "hnsw":  # HNSW doesn't support GPU
            gpu_res = get_faiss_gpu_resources()
            if gpu_res is not None:
                try:
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = True
                    index.index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index, co)
                    index._is_gpu_index = True
                    logger.info("Loaded index to GPU: %s (%d docs)", name, cpu_index.ntotal)
                except Exception as e:
                    logger.warning("Failed to load index to GPU, using CPU: %s", e)
                    index.index = cpu_index
                    index._is_gpu_index = False
            else:
                index.index = cpu_index
                index._is_gpu_index = False
        else:
            index.index = cpu_index
            index._is_gpu_index = False

        # Set search parameters
        index._set_search_params()

        # Load documents
        with open(path / "documents.pkl", "rb") as f:
            index.documents = pickle.load(f)

        logger.info("Loaded embedding index: %s (%d docs, GPU=%s)", name, len(index.documents), index._is_gpu_index)
        return index
