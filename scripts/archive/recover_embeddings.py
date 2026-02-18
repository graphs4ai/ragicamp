#!/usr/bin/env python3
"""Recovery script for interrupted embedding builds.

Completes Phase 2 (normalize + index) from Phase 1 temp files.

Usage:
    python scripts/recover_embeddings.py \
        --emb-file artifacts/indexes/tmppzyovtz_.npy \
        --chunks-file artifacts/indexes/tmpali7f93w.pkl \
        --index-name en_e5_mistral_7b_instruct_c512_o50 \
        --embedding-dim 4096 \
        --num-batches 36

The script will:
1. Load raw embeddings from the .npy file
2. Load chunks from the .pkl file
3. Normalize embeddings
4. Create FAISS HNSW index
5. Save to the proper index directory
"""

import argparse
import gc
import pickle
import time
import traceback
from pathlib import Path

import faiss
import numpy as np


def count_embeddings_in_file(emb_path: Path) -> tuple[int, list[int]]:
    """First pass: count total embeddings and batch sizes without keeping data."""
    print(f"  First pass: counting embeddings in {emb_path}...")
    batch_sizes = []
    total = 0

    with open(emb_path, "rb") as f:
        batch_num = 0
        while True:
            try:
                # Load but immediately get shape and discard
                emb = np.load(f)
                batch_num += 1
                batch_sizes.append(len(emb))
                total += len(emb)
                if batch_num % 10 == 0:
                    print(f"    Scanned {batch_num} batches, {total:,} embeddings...")
                del emb
            except Exception:
                break

    print(f"  ✓ Found {len(batch_sizes)} batches, {total:,} total embeddings")
    return total, batch_sizes


def load_embeddings_from_temp(emb_path: Path, num_batches: int, embedding_dim: int) -> np.ndarray:
    """Load embeddings from temp file with multiple np.save() calls.

    Memory-efficient: pre-allocates final array, loads batches directly into it.
    Avoids 2x memory peak from np.vstack().
    """
    print(f"Loading embeddings from {emb_path}...")
    print(f"  Expected batches: {num_batches}, dim: {embedding_dim}")

    # First pass: count embeddings
    total_count, batch_sizes = count_embeddings_in_file(emb_path)

    # Pre-allocate final array
    print(
        f"  Pre-allocating array: {total_count:,} x {embedding_dim} ({total_count * embedding_dim * 4 / 1e9:.1f} GB)..."
    )
    embeddings = np.empty((total_count, embedding_dim), dtype=np.float32)

    # Second pass: load directly into pre-allocated array
    print("  Second pass: loading into pre-allocated array...")
    offset = 0

    with open(emb_path, "rb") as f:
        for batch_num, _expected_size in enumerate(batch_sizes, 1):
            emb = np.load(f)
            embeddings[offset : offset + len(emb)] = emb
            offset += len(emb)
            if batch_num % 5 == 0 or batch_num == len(batch_sizes):
                print(f"    Loaded batch {batch_num}/{len(batch_sizes)} ({offset:,} embeddings)")
            del emb
            gc.collect()

    print(f"✓ Loaded {len(embeddings):,} embeddings, shape: {embeddings.shape}")
    return embeddings


def load_chunks_from_temp(chunks_path: Path) -> list:
    """Load chunks from temp file with multiple pickle.dump() calls."""
    print(f"Loading chunks from {chunks_path}...")

    all_chunks = []
    batch_num = 0

    with open(chunks_path, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                batch_num += 1
                all_chunks.extend(batch)
                print(f"  Batch {batch_num}: {len(batch)} chunks (total: {len(all_chunks)})")
            except EOFError:
                break

    print(f"✓ Loaded {len(all_chunks)} chunks")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Recover interrupted embedding build")
    parser.add_argument("--emb-file", required=True, help="Path to temp .npy embeddings file")
    parser.add_argument("--chunks-file", required=True, help="Path to temp .pkl chunks file")
    parser.add_argument(
        "--index-name",
        required=True,
        help="Name for the index (e.g., en_e5_mistral_7b_instruct_c512_o50)",
    )
    parser.add_argument("--embedding-dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--num-batches", type=int, default=36, help="Expected number of batches")
    parser.add_argument("--output-dir", default="artifacts/indexes", help="Output directory")
    args = parser.parse_args()

    emb_path = Path(args.emb_file)
    chunks_path = Path(args.chunks_file)
    output_dir = Path(args.output_dir) / args.index_name

    print(f"\n{'=' * 60}")
    print(f"Recovering embedding index: {args.index_name}")
    print(f"  Embeddings: {emb_path}")
    print(f"  Chunks: {chunks_path}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # Check files exist
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    # Load sequentially with proper error handling
    t_start = time.time()

    try:
        print("Step 1: Loading embeddings...")
        embeddings = load_embeddings_from_temp(emb_path, args.num_batches, args.embedding_dim)
        print(f"  Embeddings loaded in {time.time() - t_start:.1f}s\n")
    except Exception:
        print("\n*** ERROR loading embeddings ***")
        traceback.print_exc()
        raise

    try:
        t_chunks = time.time()
        print("Step 2: Loading chunks...")
        chunks = load_chunks_from_temp(chunks_path)
        print(f"  Chunks loaded in {time.time() - t_chunks:.1f}s\n")
    except Exception:
        print("\n*** ERROR loading chunks ***")
        traceback.print_exc()
        raise

    print(f"✓ Total load time: {time.time() - t_start:.1f}s\n")

    # Verify dimension
    if embeddings.shape[1] != args.embedding_dim:
        raise ValueError(
            f"Dimension mismatch: got {embeddings.shape[1]}, expected {args.embedding_dim}"
        )

    # Verify count match
    if len(embeddings) != len(chunks):
        print(f"WARNING: Count mismatch! {len(embeddings)} embeddings vs {len(chunks)} chunks")
        print(f"  Using minimum: {min(len(embeddings), len(chunks))}")
        n = min(len(embeddings), len(chunks))
        embeddings = embeddings[:n]
        chunks = chunks[:n]

    # Normalize embeddings
    print("Normalizing embeddings...")
    t_start = time.time()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings)
    print(f"✓ Normalized in {time.time() - t_start:.1f}s\n")

    # Create FAISS HNSW index
    print(f"Creating FAISS HNSW index (dim={args.embedding_dim})...")
    t_start = time.time()

    # Use all available CPU cores for FAISS
    import os

    num_threads = os.cpu_count() or 8
    faiss.omp_set_num_threads(num_threads)
    print(f"  Using {num_threads} threads for FAISS")

    index = faiss.IndexHNSWFlat(args.embedding_dim, 32)
    index.hnsw.efConstruction = 200
    print(f"  Index created, adding {len(embeddings)} vectors...")

    # Add in batches to show progress
    batch_size = 100000
    for i in range(0, len(embeddings), batch_size):
        end = min(i + batch_size, len(embeddings))
        index.add(embeddings[i:end].astype("float32"))
        print(f"  Added {end}/{len(embeddings)} vectors ({100 * end / len(embeddings):.1f}%)")

    print(f"✓ Index built with {index.ntotal} vectors in {time.time() - t_start:.1f}s\n")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving FAISS index to {output_dir / 'index.faiss'}...")
    faiss.write_index(index, str(output_dir / "index.faiss"))

    print(f"Saving documents to {output_dir / 'documents.pkl'}...")
    with open(output_dir / "documents.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Save config
    config = {
        "name": args.index_name,
        "type": "embedding",
        "embedding_model": "intfloat/e5-mistral-7b-instruct",
        "embedding_backend": "vllm",
        "index_type": "hnsw",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "chunking_strategy": "recursive",
        "num_documents": len(chunks),
        "embedding_dim": args.embedding_dim,
        "use_gpu": False,
        "recovered": True,
    }

    import json

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✓ Recovery complete!")
    print(f"  Index: {output_dir}")
    print(f"  Vectors: {index.ntotal}")
    print(f"  Documents: {len(chunks)}")
    print(f"{'=' * 60}")

    # Cleanup suggestion
    print("\nYou can now delete the temp files:")
    print(f"  rm {emb_path}")
    print(f"  rm {chunks_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 60)
        print("FATAL ERROR - Full traceback:")
        print("=" * 60)
        traceback.print_exc()
        exit(1)
