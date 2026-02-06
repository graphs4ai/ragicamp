"""Hierarchical index builder.

Builds hierarchical indexes with parent-child chunk relationships.

Uses incremental FAISS building for memory efficiency:
- Each batch: chunk → embed children → normalize → add to FAISS
- Checkpoints index every N batches for crash recovery
- Never loads all embeddings into RAM at once

Supports checkpointing for resume after crashes.
"""

import gc
import os
import pickle
import time
from typing import Any

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

from .checkpoint import CheckpointManager, HierarchicalCheckpoint
from .storage import EmbeddingStorage

logger = get_logger(__name__)

# Checkpoint every batch - hierarchical batches can take hours
CHECKPOINT_INTERVAL = 1


def build_hierarchical_index(
    retriever_config: dict[str, Any],
    corpus_config: dict[str, Any],
    doc_batch_size: int = 1000,
    embedding_batch_size: int = 64,
    embedding_backend: str = "vllm",
    vllm_gpu_memory_fraction: float = Defaults.VLLM_GPU_MEMORY_FRACTION,
) -> str:
    """Build a hierarchical index with batched processing.

    Hierarchical indexes have unique parent/child chunking, so they
    can't share indexes with other retrievers.

    Uses incremental FAISS building:
    - Each batch: chunk → embed children → normalize → add to FAISS
    - Checkpoints every 5 batches for crash recovery
    - Memory efficient: only one batch in RAM at a time

    Supports resume from checkpoint if process is interrupted.

    Args:
        retriever_config: Retriever configuration with chunk sizes
        corpus_config: Corpus configuration
        doc_batch_size: Documents to process per batch (default 1000)
        embedding_batch_size: Texts to embed per GPU batch (default 64)
        embedding_backend: 'vllm' or 'sentence_transformers'
        vllm_gpu_memory_fraction: GPU memory fraction for vLLM (default 0.9)

    Returns:
        Retriever name
    """
    import faiss
    import numpy as np

    from ragicamp.core.types import Document
    from ragicamp.corpus import CorpusConfig, WikipediaCorpus
    from ragicamp.models.embedder import create_embedder
    from ragicamp.rag.chunking.hierarchical import HierarchicalChunker

    manager = get_artifact_manager()
    # Use embedding_index for the index name (consistent with dense indexes)
    # Fall back to name if embedding_index not specified
    index_name = retriever_config.get("embedding_index", retriever_config["name"])
    retriever_name = retriever_config["name"]
    embedding_model = retriever_config.get("embedding_model", Defaults.EMBEDDING_MODEL)

    # Get embedding backend from retriever config, fall back to parameter
    backend = retriever_config.get("embedding_backend", embedding_backend)

    parent_chunk_size = retriever_config.get("parent_chunk_size", 1024)
    child_chunk_size = retriever_config.get("child_chunk_size", 256)
    parent_overlap = retriever_config.get("parent_overlap", 100)
    child_overlap = retriever_config.get("child_overlap", 50)

    # Use all CPU cores for FAISS
    num_threads = os.cpu_count() or 8
    faiss.omp_set_num_threads(num_threads)

    print(f"\n{'=' * 60}")
    print(f"Building hierarchical index: {index_name}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Embedding backend: {backend}")
    print(f"  Parent chunks: {parent_chunk_size}")
    print(f"  Child chunks: {child_chunk_size}")
    print(f"  Doc batch size: {doc_batch_size}, embedding batch size: {embedding_batch_size}")
    print(f"  FAISS threads: {num_threads}")
    print(f"{'=' * 60}")

    # Build metadata dict for WikiRank filter and other options
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

    # Initialize chunker
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_overlap,
    )

    # Initialize encoder using factory (backend-agnostic)
    print(f"Loading embedding model ({backend})...")
    encoder = create_embedder(
        model_name=embedding_model,
        backend=backend,
        gpu_memory_fraction=vllm_gpu_memory_fraction,
        enforce_eager=False,
    )
    embedding_dim = encoder.get_sentence_embedding_dimension()
    print(f"  Embedder loaded (dim={embedding_dim})")

    # Setup work directory for checkpointing and temp storage
    work_dir = manager.indexes_dir / ".work" / index_name
    work_dir.mkdir(parents=True, exist_ok=True)
    index_checkpoint_path = work_dir / "child_index.faiss"

    # Initialize storage and checkpoint manager
    storage = EmbeddingStorage(work_dir)
    checkpoint_mgr = CheckpointManager(work_dir)

    # Mapping file for child->parent relationships
    mapping_path = work_dir / "child_to_parent.pkl"

    # Check for existing checkpoint (resume support)
    checkpoint = checkpoint_mgr.load(HierarchicalCheckpoint)
    if checkpoint and index_checkpoint_path.exists():
        print(
            f"  Resuming from checkpoint: batch {checkpoint.batch_num}, {checkpoint.total_docs} docs"
        )
        print("  Loading checkpointed index...")
        index = faiss.read_index(str(index_checkpoint_path))
        print(f"  Loaded index with {index.ntotal} vectors")
        start_batch = checkpoint.batch_num
        total_docs = checkpoint.total_docs
        parent_count = checkpoint.total_parent_chunks
        child_count = checkpoint.total_child_chunks
        mapping_file = open(mapping_path, "ab")
    else:
        # Create fresh index
        index = faiss.IndexFlatIP(embedding_dim)
        start_batch = 0
        total_docs = 0
        parent_count = 0
        child_count = 0
        mapping_file = open(mapping_path, "wb")

    doc_batch = []
    batch_num = 0
    docs_to_skip = start_batch * doc_batch_size

    # ==========================================================================
    # INCREMENTAL BUILD: Chunk → Embed → Normalize → Add to FAISS
    # ==========================================================================
    print("Building index incrementally (chunk → embed → normalize → index)...")

    try:
        for doc in corpus.load():
            # Skip already processed docs when resuming
            if docs_to_skip > 0:
                docs_to_skip -= 1
                continue

            doc_with_id = Document(
                id=f"wiki_{total_docs}",
                text=doc.text,
                metadata=doc.metadata,
            )
            doc_batch.append(doc_with_id)
            total_docs += 1

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
                print("    Chunking (hierarchical)...", end=" ", flush=True)
                parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                    iter(doc_batch)
                )
                chunk_elapsed = time.time() - t_chunk
                print(
                    f"→ {len(parent_chunks)} parents, {len(child_chunks)} children ({chunk_elapsed:.1f}s)"
                )

                # Save mapping
                pickle.dump(batch_child_to_parent, mapping_file)
                mapping_file.flush()

                # Save parent chunks (no embedding needed for parents)
                storage.append_chunks(parent_chunks, key="parent")
                parent_count += len(parent_chunks)

                if child_chunks:
                    child_texts = [c.text for c in child_chunks]
                    print(f"    Embedding {len(child_texts)} child chunks...")

                    # Embed
                    embeddings = encoder.encode(
                        child_texts, show_progress_bar=True, batch_size=embedding_batch_size
                    )

                    # Normalize in-place
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    np.divide(embeddings, norms, out=embeddings)

                    # Add to index
                    t_index = time.time()
                    index.add(embeddings.astype(np.float32))
                    index_elapsed = time.time() - t_index
                    print(
                        f"    ✓ Added to index in {index_elapsed:.1f}s (total: {index.ntotal} vectors)"
                    )

                    # Save child chunks
                    storage.append_chunks(child_chunks, key="child")
                    child_count += len(child_chunks)

                    del child_texts, embeddings
                    gc.collect()

                print(
                    f"  ✓ Total: {total_docs} docs → {parent_count} parents, {child_count} children"
                )
                doc_batch = []
                del parent_chunks, child_chunks, batch_child_to_parent
                gc.collect()

                # Checkpoint every N batches
                if batch_num % CHECKPOINT_INTERVAL == 0:
                    print("    Saving checkpoint...")
                    faiss.write_index(index, str(index_checkpoint_path))
                    checkpoint_mgr.save(
                        HierarchicalCheckpoint(
                            batch_num=batch_num,
                            total_docs=total_docs,
                            total_parent_chunks=parent_count,
                            total_child_chunks=child_count,
                            parent_embedding_sizes=[],
                            child_embedding_sizes=[],
                        )
                    )

        # Process remaining docs
        if doc_batch:
            batch_num += 1
            batch_size = len(doc_batch)
            print(f"\n  [Batch {batch_num}] Processing {batch_size} documents (final)...")

            t_chunk = time.time()
            print("    Chunking (hierarchical)...", end=" ", flush=True)
            parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                iter(doc_batch)
            )
            chunk_elapsed = time.time() - t_chunk
            print(
                f"→ {len(parent_chunks)} parents, {len(child_chunks)} children ({chunk_elapsed:.1f}s)"
            )

            pickle.dump(batch_child_to_parent, mapping_file)
            mapping_file.flush()

            storage.append_chunks(parent_chunks, key="parent")
            parent_count += len(parent_chunks)

            if child_chunks:
                child_texts = [c.text for c in child_chunks]
                print(f"    Embedding {len(child_texts)} child chunks...")
                embeddings = encoder.encode(
                    child_texts, show_progress_bar=True, batch_size=embedding_batch_size
                )

                # Normalize in-place
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                np.divide(embeddings, norms, out=embeddings)

                # Add to index
                t_index = time.time()
                index.add(embeddings.astype(np.float32))
                print(f"    ✓ Added to index in {time.time() - t_index:.1f}s")

                storage.append_chunks(child_chunks, key="child")
                child_count += len(child_chunks)

                del child_texts, embeddings

            print(f"  ✓ Total: {total_docs} docs → {parent_count} parents, {child_count} children")
            del doc_batch, parent_chunks, child_chunks, batch_child_to_parent
            gc.collect()

    except Exception as e:
        # Save checkpoint on error
        print(f"\n*** Error at batch {batch_num}: {e}")
        print("    Saving checkpoint before exit...")
        faiss.write_index(index, str(index_checkpoint_path))
        checkpoint_mgr.save(
            HierarchicalCheckpoint(
                batch_num=batch_num - 1,
                total_docs=total_docs,
                total_parent_chunks=parent_count,
                total_child_chunks=child_count,
                parent_embedding_sizes=[],
                child_embedding_sizes=[],
            )
        )
        mapping_file.close()
        storage.close()
        raise RuntimeError(f"Build failed at batch {batch_num}: {e}") from e

    mapping_file.close()
    print(
        f"\n✓ Build complete: {total_docs} docs → {parent_count} parents, {child_count} children → {index.ntotal} vectors"
    )

    # ==========================================================================
    # Save to final location
    # ==========================================================================
    print("\nSaving to final location...")
    index_path = manager.get_embedding_index_path(index_name)
    index_path.mkdir(parents=True, exist_ok=True)

    # Load and save all documents
    print("  Saving parent documents...")
    all_parent_docs = storage.load_all_chunks(key="parent")
    with open(index_path / "parent_docs.pkl", "wb") as f:
        pickle.dump(all_parent_docs, f)
    print(f"    ✓ {len(all_parent_docs)} parent docs")

    print("  Saving child documents...")
    all_child_docs = storage.load_all_chunks(key="child")
    with open(index_path / "child_docs.pkl", "wb") as f:
        pickle.dump(all_child_docs, f)
    print(f"    ✓ {len(all_child_docs)} child docs")

    # Build parent ID -> index mapping
    parent_id_to_idx = {}
    for idx, doc in enumerate(all_parent_docs):
        parent_id_to_idx[doc.id] = idx

    # Load and merge child_to_parent mapping
    print("  Saving child-to-parent mapping...")
    child_to_parent = {}
    with open(mapping_path, "rb") as f:
        while True:
            try:
                batch_mapping = pickle.load(f)
                child_to_parent.update(batch_mapping)
            except EOFError:
                break

    with open(index_path / "child_to_parent.pkl", "wb") as f:
        pickle.dump(child_to_parent, f)
    print(f"    ✓ {len(child_to_parent)} mappings")

    # Save FAISS index
    faiss.write_index(index, str(index_path / "child_index.faiss"))

    # Save index config
    index_config = {
        "name": index_name,
        "type": "hierarchical",
        "embedding_model": embedding_model,
        "embedding_backend": backend,
        "parent_chunk_size": parent_chunk_size,
        "child_chunk_size": child_chunk_size,
        "parent_overlap": parent_overlap,
        "child_overlap": child_overlap,
        "num_parents": parent_count,
        "num_children": child_count,
        "embedding_dim": embedding_dim,
    }
    manager.save_json(index_config, index_path / "config.json")

    # Create retriever config that references the index
    retriever_path = manager.get_retriever_path(retriever_name)
    retriever_config_data = {
        "name": retriever_name,
        "type": "hierarchical",
        "embedding_index": index_name,  # Reference the index by its name
        "embedding_model": embedding_model,
        "embedding_backend": backend,
        "parent_chunk_size": parent_chunk_size,
        "child_chunk_size": child_chunk_size,
        "num_parents": parent_count,
        "num_children": child_count,
    }
    manager.save_json(retriever_config_data, retriever_path / "config.json")

    # Cleanup work directory
    storage.cleanup()
    checkpoint_mgr.clear()
    if mapping_path.exists():
        mapping_path.unlink()
    if index_checkpoint_path.exists():
        index_checkpoint_path.unlink()

    # Try to remove work dir if empty
    try:
        work_dir.rmdir()
    except OSError:
        pass

    # Clean up
    del all_parent_docs, all_child_docs, child_to_parent, parent_id_to_idx
    del index
    encoder.unload()
    del chunker
    gc.collect()

    print(f"✓ Saved hierarchical index: {index_name}")
    return index_name
