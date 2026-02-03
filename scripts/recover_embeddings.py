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
from pathlib import Path

import faiss
import numpy as np


def load_embeddings_from_temp(emb_path: Path, num_batches: int, embedding_dim: int) -> np.ndarray:
    """Load embeddings from temp file with multiple np.save() calls.
    
    The old code appended multiple arrays to one file using np.save().
    We need to load them sequentially.
    """
    print(f"Loading embeddings from {emb_path}...")
    print(f"  Expected batches: {num_batches}, dim: {embedding_dim}")
    
    all_embeddings = []
    total_loaded = 0
    
    with open(emb_path, "rb") as f:
        batch_num = 0
        while True:
            try:
                emb = np.load(f)
                batch_num += 1
                total_loaded += len(emb)
                all_embeddings.append(emb)
                print(f"  Batch {batch_num}: {len(emb)} embeddings (total: {total_loaded})")
            except Exception as e:
                print(f"  Finished loading at batch {batch_num}: {e}")
                break
    
    if not all_embeddings:
        raise ValueError("No embeddings found in file!")
    
    print(f"  Stacking {len(all_embeddings)} batches...")
    embeddings = np.vstack(all_embeddings)
    del all_embeddings
    gc.collect()
    
    print(f"✓ Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
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
    parser.add_argument("--index-name", required=True, help="Name for the index (e.g., en_e5_mistral_7b_instruct_c512_o50)")
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
    
    # Load embeddings
    t_start = time.time()
    embeddings = load_embeddings_from_temp(emb_path, args.num_batches, args.embedding_dim)
    print(f"  Load time: {time.time() - t_start:.1f}s\n")
    
    # Verify dimension
    if embeddings.shape[1] != args.embedding_dim:
        raise ValueError(f"Dimension mismatch: got {embeddings.shape[1]}, expected {args.embedding_dim}")
    
    # Load chunks
    t_start = time.time()
    chunks = load_chunks_from_temp(chunks_path)
    print(f"  Load time: {time.time() - t_start:.1f}s\n")
    
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
    index = faiss.IndexHNSWFlat(args.embedding_dim, 32)
    index.hnsw.efConstruction = 200
    print(f"  Index created, adding {len(embeddings)} vectors...")
    
    # Add in batches to show progress
    batch_size = 100000
    for i in range(0, len(embeddings), batch_size):
        end = min(i + batch_size, len(embeddings))
        index.add(embeddings[i:end].astype("float32"))
        print(f"  Added {end}/{len(embeddings)} vectors ({100*end/len(embeddings):.1f}%)")
    
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
    print(f"✓ Recovery complete!")
    print(f"  Index: {output_dir}")
    print(f"  Vectors: {index.ntotal}")
    print(f"  Documents: {len(chunks)}")
    print(f"{'=' * 60}")
    
    # Cleanup suggestion
    print(f"\nYou can now delete the temp files:")
    print(f"  rm {emb_path}")
    print(f"  rm {chunks_path}")


if __name__ == "__main__":
    main()
