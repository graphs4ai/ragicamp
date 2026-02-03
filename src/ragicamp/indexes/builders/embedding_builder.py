"""Embedding index builder.

Builds dense vector indexes for semantic search.
Supports both sentence-transformers and vLLM embedding backends.

Uses a two-phase approach for optimal GPU utilization:
- Phase 1: Chunk + embed all batches, save raw embeddings to disk
- Phase 2: Load all, normalize, add to FAISS index

Supports checkpointing for resume after crashes.
"""

import gc
import os
import time
from typing import Any

# Disable tokenizers parallelism to avoid fork warnings
# Must be set before importing transformers/tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

from .checkpoint import CheckpointManager, EmbeddingCheckpoint
from .storage import EmbeddingStorage

logger = get_logger(__name__)

# Maximum document size in characters (larger docs are truncated)
MAX_DOC_CHARS = 100_000


def build_embedding_index(
    index_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_config: dict[str, Any],
    chunking_strategy: str = "recursive",
    doc_batch_size: int = 5000,
    embedding_batch_size: int = 64,
    chunk_workers: int = None,  # Kept for API compatibility, not used
    index_type: str = None,
    use_gpu: bool = None,
    nlist: int = None,
    nprobe: int = None,
    embedding_backend: str = "vllm",
    vllm_gpu_memory_fraction: float = 0.9,
) -> str:
    """Build a shared embedding index with batched processing.

    This is the expensive part - chunking + embedding. Once built, multiple
    retrievers can reuse this index.

    Uses a two-phase approach:
    - Phase 1: Chunk + embed all batches, save raw embeddings to disk
    - Phase 2: Load all, normalize, add to FAISS index

    Supports resume from checkpoint if process is interrupted.

    Args:
        index_name: Canonical name for this index
        embedding_model: Embedding model to use
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        corpus_config: Corpus configuration
        chunking_strategy: Chunking strategy ('recursive', 'fixed', 'sentence', 'paragraph')
        doc_batch_size: Documents to process per batch (default 5000)
        embedding_batch_size: Texts to embed per GPU batch (default 64, increase for more VRAM)
        chunk_workers: Deprecated, kept for API compatibility
        index_type: FAISS index type ('flat', 'ivf', 'ivfpq', 'hnsw'). Default: Defaults.FAISS_INDEX_TYPE
        use_gpu: Whether to use GPU FAISS. Default: Defaults.FAISS_USE_GPU
        nlist: Number of clusters for IVF indexes. Default: Defaults.FAISS_IVF_NLIST
        nprobe: Number of clusters to search. Default: Defaults.FAISS_IVF_NPROBE
        embedding_backend: 'vllm' or 'sentence_transformers'
        vllm_gpu_memory_fraction: GPU memory fraction for vLLM (default 0.9)

    Returns:
        Path to the saved index
    """
    # Apply defaults
    index_type = index_type or Defaults.FAISS_INDEX_TYPE
    use_gpu = use_gpu if use_gpu is not None else Defaults.FAISS_USE_GPU
    nlist = nlist or Defaults.FAISS_IVF_NLIST
    nprobe = nprobe or Defaults.FAISS_IVF_NPROBE

    import faiss
    import numpy as np
    from tqdm import tqdm

    from ragicamp.corpus import ChunkConfig, CorpusConfig, DocumentChunker, WikipediaCorpus
    from ragicamp.models.embedder import create_embedder
    from ragicamp.retrievers.base import Document

    manager = get_artifact_manager()

    print(f"\n{'=' * 60}")
    print(f"Building shared embedding index: {index_name}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Embedding backend: {embedding_backend}")
    print(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    print(f"  Chunking strategy: {chunking_strategy}")
    print(f"  Doc batch size: {doc_batch_size}, embedding batch size: {embedding_batch_size}")
    print(f"  FAISS index type: {index_type}, GPU: {use_gpu}")
    if index_type in ("ivf", "ivfpq"):
        print(f"  IVF params: nlist={nlist}, nprobe={nprobe}")
    print(f"{'=' * 60}")

    # Initialize corpus
    corpus_metadata = {}
    if corpus_config.get("wikirank_top_k"):
        corpus_metadata["wikirank_top_k"] = corpus_config["wikirank_top_k"]
    if corpus_config.get("min_chars"):
        corpus_metadata["min_chars"] = corpus_config["min_chars"]
    if "streaming" in corpus_config:
        corpus_metadata["streaming"] = corpus_config["streaming"]
    if corpus_config.get("num_proc"):
        corpus_metadata["num_proc"] = corpus_config["num_proc"]

    corpus_cfg = CorpusConfig(
        name=f"wikipedia_{corpus_config.get('version', 'simple')}",
        source=corpus_config.get("source", "wikimedia/wikipedia"),
        version=corpus_config.get("version", "20231101.simple"),
        max_docs=corpus_config.get("max_docs"),
        metadata=corpus_metadata,
    )
    corpus = WikipediaCorpus(corpus_cfg)

    # Initialize encoder using factory (backend-agnostic)
    print(f"Loading embedding model ({embedding_backend})...")
    encoder = create_embedder(
        model_name=embedding_model,
        backend=embedding_backend,
        gpu_memory_fraction=vllm_gpu_memory_fraction,
        enforce_eager=False,
    )
    embedding_dim = encoder.get_sentence_embedding_dimension()
    print(f"  Embedder loaded (dim={embedding_dim})")

    # Initialize FAISS index
    if index_type == "flat":
        cpu_index = faiss.IndexFlatIP(embedding_dim)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(embedding_dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
    elif index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(embedding_dim)
        m = min(32, embedding_dim // 4)
        cpu_index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, 8)
    elif index_type == "hnsw":
        cpu_index = faiss.IndexHNSWFlat(embedding_dim, 32)
        cpu_index.hnsw.efConstruction = 200
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index = cpu_index
    is_trained = index_type == "flat" or index_type == "hnsw"

    # Chunking config
    chunk_config = ChunkConfig(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunker = DocumentChunker(chunk_config)

    # Setup work directory for checkpointing and temp storage
    work_dir = manager.indexes_dir / ".work" / index_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Initialize storage and checkpoint manager
    storage = EmbeddingStorage(work_dir)
    checkpoint_mgr = CheckpointManager(work_dir)

    # Check for existing checkpoint (resume support)
    checkpoint = checkpoint_mgr.load(EmbeddingCheckpoint)
    if checkpoint:
        print(
            f"  Resuming from checkpoint: batch {checkpoint.batch_num}, {checkpoint.total_docs} docs"
        )
        start_batch = checkpoint.batch_num
        total_docs = checkpoint.total_docs
        total_chunks = checkpoint.total_chunks
        # Restore embedding sizes for loading later
        storage._embedding_sizes["default"] = checkpoint.embedding_sizes.copy()
    else:
        start_batch = 0
        total_docs = 0
        total_chunks = 0

    doc_batch = []
    batch_num = 0
    docs_to_skip = start_batch * doc_batch_size  # Skip already processed docs

    # ==========================================================================
    # PHASE 1: Chunk and embed all batches (GPU runs continuously)
    # ==========================================================================
    print("Phase 1: Chunking and embedding...")

    try:
        for doc in corpus.load():
            # Skip already processed docs when resuming
            if docs_to_skip > 0:
                docs_to_skip -= 1
                continue

            doc_batch.append(doc)

            if len(doc_batch) >= doc_batch_size:
                batch_num += 1

                # Skip if already processed (resume case)
                if batch_num <= start_batch:
                    doc_batch = []
                    continue

                batch_size = len(doc_batch)
                print(f"\n  [Batch {batch_num}] Processing {batch_size} documents...")

                # Chunking phase
                t_chunk = time.time()
                batch_chunks = []
                truncated = 0

                for doc in tqdm(doc_batch, desc="    Chunking", leave=False, ncols=80):
                    text = doc.text
                    if len(text) > MAX_DOC_CHARS:
                        text = text[:MAX_DOC_CHARS]
                        truncated += 1
                        doc = Document(id=doc.id, text=text, metadata=doc.metadata)
                    doc_chunks = list(chunker.strategy.chunk_document(doc))
                    batch_chunks.extend(doc_chunks)

                chunk_elapsed = time.time() - t_chunk
                msg = f"    ✓ {len(batch_chunks)} chunks in {chunk_elapsed:.1f}s"
                if truncated > 0:
                    msg += f" (truncated {truncated})"
                print(msg)

                total_docs += batch_size
                total_chunks += len(batch_chunks)

                if batch_chunks:
                    texts = [c.text for c in batch_chunks]
                    print(f"    Embedding {len(texts)} chunks...")

                    # Embed (no normalization yet - save raw)
                    embeddings = encoder.encode(
                        texts, show_progress_bar=True, batch_size=embedding_batch_size
                    )

                    # Save to storage
                    storage.append_embeddings(embeddings)
                    storage.append_chunks(batch_chunks)

                    # Save checkpoint
                    checkpoint_mgr.save(
                        EmbeddingCheckpoint(
                            batch_num=batch_num,
                            total_docs=total_docs,
                            total_chunks=total_chunks,
                            embedding_sizes=storage.get_embedding_sizes(),
                        )
                    )

                print(f"  ✓ Total: {total_docs} docs → {total_chunks} chunks")
                doc_batch = []
                del batch_chunks, texts, embeddings
                gc.collect()

        # Process remaining docs
        if doc_batch:
            batch_num += 1
            batch_size = len(doc_batch)
            print(f"\n  [Batch {batch_num}] Processing {batch_size} documents (final)...")

            t_chunk = time.time()
            batch_chunks = []
            truncated = 0

            for doc in tqdm(doc_batch, desc="    Chunking", leave=False, ncols=80):
                text = doc.text
                if len(text) > MAX_DOC_CHARS:
                    text = text[:MAX_DOC_CHARS]
                    truncated += 1
                    doc = Document(id=doc.id, text=text, metadata=doc.metadata)
                doc_chunks = list(chunker.strategy.chunk_document(doc))
                batch_chunks.extend(doc_chunks)

            chunk_elapsed = time.time() - t_chunk
            msg = f"    ✓ {len(batch_chunks)} chunks in {chunk_elapsed:.1f}s"
            if truncated > 0:
                msg += f" (truncated {truncated})"
            print(msg)

            total_docs += batch_size
            total_chunks += len(batch_chunks)

            if batch_chunks:
                texts = [c.text for c in batch_chunks]
                print(f"    Embedding {len(texts)} chunks...")
                embeddings = encoder.encode(
                    texts, show_progress_bar=True, batch_size=embedding_batch_size
                )
                storage.append_embeddings(embeddings)
                storage.append_chunks(batch_chunks)
                checkpoint_mgr.save(
                    EmbeddingCheckpoint(
                        batch_num=batch_num,
                        total_docs=total_docs,
                        total_chunks=total_chunks,
                        embedding_sizes=storage.get_embedding_sizes(),
                    )
                )

            print(f"  ✓ Total: {total_docs} docs → {total_chunks} chunks")
            del doc_batch, batch_chunks, texts, embeddings
            gc.collect()

    except Exception as e:
        storage.close()
        raise RuntimeError(f"Phase 1 failed at batch {batch_num}: {e}") from e

    print(f"✓ Phase 1 complete: {total_docs} docs → {total_chunks} chunks in {batch_num} batches")

    # ==========================================================================
    # PHASE 2: Load embeddings, normalize, add to FAISS
    # ==========================================================================
    print("\nPhase 2: Post-processing (normalize + index)...")

    # Load all embeddings from disk
    print("  Loading embeddings from disk...")
    t_load = time.time()
    embeddings = storage.load_all_embeddings()
    print(f"  ✓ Loaded {len(embeddings)} embeddings in {time.time() - t_load:.1f}s")

    # Normalize all at once (in-place for efficiency)
    print("  Normalizing embeddings...")
    t_norm = time.time()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings)
    print(f"  ✓ Normalized in {time.time() - t_norm:.1f}s")

    # Train IVF if needed
    if not is_trained and index_type in ("ivf", "ivfpq"):
        print(f"  Training IVF index on {len(embeddings)} vectors...")
        index.train(embeddings)
        is_trained = True

    # Add all to index at once
    print("  Adding to FAISS index...")
    t_add = time.time()
    index.add(embeddings)
    print(f"  ✓ Added {index.ntotal} vectors in {time.time() - t_add:.1f}s")

    del embeddings
    gc.collect()

    print(f"✓ Phase 2 complete: {index.ntotal} vectors indexed")

    # Set nprobe for IVF indexes
    if index_type in ("ivf", "ivfpq") and hasattr(index, "nprobe"):
        index.nprobe = nprobe
        print(f"  Set nprobe={nprobe} for IVF index")

    # Save to shared indexes directory
    artifact_path = manager.get_embedding_index_path(index_name)

    # Save FAISS index (always save CPU version for portability)
    faiss.write_index(index, str(artifact_path / "index.faiss"))

    # Load and save all chunks
    print("Saving documents...")
    all_chunks = storage.load_all_chunks()
    with open(artifact_path / "documents.pkl", "wb") as f:
        import pickle

        pickle.dump(all_chunks, f)

    # Cleanup work directory
    storage.cleanup()
    checkpoint_mgr.clear()

    # Try to remove work dir if empty
    try:
        work_dir.rmdir()
    except OSError:
        pass  # Not empty, leave it

    del all_chunks
    gc.collect()

    config = {
        "name": index_name,
        "type": "embedding",
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "index_type": index_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": chunking_strategy,
        "corpus_version": corpus_config.get("version"),
        "corpus_max_docs": corpus_config.get("max_docs"),
        "num_documents": total_chunks,
        "embedding_dim": embedding_dim,
        "use_gpu": use_gpu,
        "nlist": nlist,
        "nprobe": nprobe,
    }
    manager.save_json(config, artifact_path / "config.json")

    del index
    encoder.unload()
    gc.collect()

    print(f"✓ Saved to: {artifact_path}")
    if use_gpu:
        print(f"  Index will be loaded to GPU on next use (use_gpu={use_gpu})")
    return str(artifact_path)
