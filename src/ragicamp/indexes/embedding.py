"""Embedding index for dense retrieval.

Supports both CPU and GPU FAISS indexes for high-performance similarity search.
GPU FAISS can provide 10-100x speedup for large indexes.
"""

import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Enable TensorFloat32 for faster matrix multiplication on Ampere+ GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.indexes.base import Index
from ragicamp.retrievers.base import Document
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


# Configure FAISS CPU threads for better performance
def _configure_faiss_threads():
    """Configure FAISS to use multiple CPU threads."""
    import os

    num_threads = Defaults.FAISS_CPU_THREADS

    # 0 means auto-detect (use all available cores)
    if num_threads == 0:
        num_threads = os.cpu_count() or 1

    # Set environment variable (affects OpenMP globally)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))

    # Also try the FAISS API
    try:
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(num_threads)

        # Log actual thread count
        if hasattr(faiss, "omp_get_max_threads"):
            actual = faiss.omp_get_max_threads()
            logger.info("FAISS CPU threads: %d", actual)
    except Exception as e:
        logger.debug("FAISS thread config: %s", e)


_configure_faiss_threads()

# Track FAISS GPU resources globally to avoid re-initialization
_faiss_gpu_resources: Optional["faiss.StandardGpuResources"] = None
_faiss_gpu_available: Optional[bool] = None


def _check_faiss_gpu_available() -> bool:
    """Check if FAISS GPU is available.

    Requires faiss-gpu package (not faiss-cpu).
    Install with: pip install faiss-gpu-cu12 (or faiss-gpu-cu11 for CUDA 11)
    """
    global _faiss_gpu_available
    if _faiss_gpu_available is None:
        try:
            # Check if faiss has GPU support (faiss-gpu vs faiss-cpu)
            if not hasattr(faiss, "StandardGpuResources"):
                logger.warning(
                    "FAISS GPU not available: faiss-cpu is installed. "
                    "Install faiss-gpu-cu12 for GPU support: "
                    "pip uninstall faiss-cpu -y && pip install faiss-gpu-cu12"
                )
                _faiss_gpu_available = False
                return False

            # Try to create GPU resources
            res = faiss.StandardGpuResources()
            del res
            _faiss_gpu_available = True
            logger.info("FAISS GPU support detected")
        except Exception as e:
            _faiss_gpu_available = False
            logger.warning("FAISS GPU not available: %s", e)
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
            logger.info(
                "Initialized FAISS GPU resources (temp_memory=%dMB)",
                temp_memory_mb or Defaults.FAISS_GPU_TEMP_MEMORY_MB,
            )
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
        """Lazy load encoder with optimizations similar to vLLM.

        Optimizations applied:
        1. Flash Attention 2 (if available) - faster attention computation
        2. FP16/BF16 precision - 2x memory savings, faster compute
        3. torch.compile() - kernel fusion and optimization
        4. Optimal batch handling - configured via Defaults.EMBEDDING_BATCH_SIZE
        """
        if self._encoder is None:
            model_kwargs = {}

            # Optimization 1: Flash Attention 2
            if Defaults.EMBEDDING_USE_FLASH_ATTENTION:
                try:
                    import flash_attn  # noqa: F401

                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled for embeddings")
                except ImportError:
                    logger.debug("flash-attn not installed, using default attention")

            # Optimization 2: FP16/BF16 precision
            if Defaults.EMBEDDING_USE_FP16 and torch.cuda.is_available():
                # Prefer BF16 on Ampere+ for better stability, else FP16
                if torch.cuda.is_bf16_supported():
                    model_kwargs["torch_dtype"] = torch.bfloat16
                    logger.info("Using BF16 precision for embeddings")
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                    logger.info("Using FP16 precision for embeddings")

            self._encoder = SentenceTransformer(
                self.embedding_model_name,
                trust_remote_code=True,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()

            # Optimization 3: torch.compile() for kernel fusion
            if Defaults.EMBEDDING_USE_TORCH_COMPILE:
                try:
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
            raise ValueError(
                f"Unknown index type: {self.index_type}. Valid types: flat, ivf, ivfpq, hnsw"
            )

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
                    if self.index_type in ("ivf", "ivfpq"):
                        logger.info(
                            "Created GPU FAISS index (type=%s, nlist=%d)", self.index_type, nlist
                        )
                    else:
                        logger.info("Created GPU FAISS index (type=%s)", self.index_type)
                    return gpu_index
                except Exception as e:
                    logger.warning("Failed to create GPU index, falling back to CPU: %s", e)

        if self.index_type in ("ivf", "ivfpq"):
            logger.info("Created CPU FAISS index (type=%s, nlist=%d)", self.index_type, nlist)
        elif self.index_type == "hnsw":
            logger.info(
                "Created CPU FAISS index (type=%s, M=32, efConstruction=200)", self.index_type
            )
        else:
            logger.info("Created CPU FAISS index (type=%s)", self.index_type)
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
        if hasattr(index, "nprobe"):
            index.nprobe = self.nprobe
            logger.debug("Set nprobe=%d for IVF index", self.nprobe)

    def build(
        self, documents: list[Document], batch_size: int | None = None
    ) -> None:
        """Build index from documents with optimized encoding.

        Args:
            documents: List of documents to index
            batch_size: Encoding batch size (None = use default from Defaults)
        """
        self.documents = documents
        logger.info("Building embedding index for %d documents", len(documents))

        # Use optimized batch size
        if batch_size is None:
            batch_size = Defaults.EMBEDDING_BATCH_SIZE

        # Compute embeddings with optimized settings
        texts = [doc.text for doc in documents]
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=Defaults.EMBEDDING_SHOW_PROGRESS,
            normalize_embeddings=False,  # We normalize after for consistency
        )

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

        logger.info(
            "Loaded embedding index: %s (%d docs, GPU=%s)",
            name,
            len(index.documents),
            index._is_gpu_index,
        )
        return index

    def convert_to(self, new_index_type: str, save: bool = True) -> "EmbeddingIndex":
        """Convert index to a different type without re-embedding.

        This extracts vectors from the current index and builds a new index
        with the specified type. Much faster than rebuilding from scratch.

        Args:
            new_index_type: Target index type ('flat', 'ivf', 'hnsw')
            save: Whether to save the converted index (overwrites existing)

        Returns:
            Self with converted index
        """
        if self.index is None:
            raise ValueError("No index to convert")

        old_type = self.index_type
        n_vectors = self.index.ntotal
        dim = self._embedding_dim

        logger.info(
            "Converting index from %s to %s (%d vectors)...", old_type, new_index_type, n_vectors
        )

        # Extract all vectors from current index
        # For flat index, we can reconstruct directly
        if hasattr(self.index, "reconstruct_n"):
            vectors = np.zeros((n_vectors, dim), dtype="float32")
            self.index.reconstruct_n(0, n_vectors, vectors)
        else:
            # Fallback: search for each vector by ID (slower)
            logger.warning("Index type doesn't support reconstruct_n, using slower method")
            vectors = np.zeros((n_vectors, dim), dtype="float32")
            for i in range(n_vectors):
                vectors[i] = self.index.reconstruct(i)

        # Create new index
        self.index_type = new_index_type
        self.index = self._create_faiss_index(num_vectors=n_vectors)

        # Train if needed (IVF)
        if new_index_type in ("ivf", "ivfpq"):
            logger.info("Training %s index...", new_index_type)
            self.index.train(vectors)

        # Add vectors to new index
        logger.info("Adding %d vectors to new %s index...", n_vectors, new_index_type)
        self.index.add(vectors)

        # Set search parameters
        self._set_search_params()

        logger.info("Conversion complete: %s -> %s", old_type, new_index_type)

        if save:
            self.save()
            logger.info("Saved converted index")

        return self

    @classmethod
    def convert_existing(
        cls,
        name: str,
        new_index_type: str,
        path: Optional[Path] = None,
    ) -> "EmbeddingIndex":
        """Load an existing index, convert it to a new type, and save.

        Args:
            name: Index name
            new_index_type: Target index type ('flat', 'ivf', 'hnsw')
            path: Optional custom path

        Returns:
            Converted EmbeddingIndex

        Example:
            >>> EmbeddingIndex.convert_existing("en_bge_m3_c512_o50", "hnsw")
        """
        # Load with GPU disabled (conversion happens on CPU)
        index = cls.load(name, path=path, use_gpu=False)
        index.convert_to(new_index_type, save=True)
        return index
