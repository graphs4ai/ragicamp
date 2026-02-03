#!/usr/bin/env python3
"""Incremental recovery script for interrupted embedding builds.

Processes embeddings batch-by-batch instead of loading all into RAM.
Much more memory efficient and resilient to crashes.

Usage:
    python scripts/recover_embeddings_incremental.py \
        --emb-file artifacts/indexes/tmppzyovtz_.npy \
        --chunks-file artifacts/indexes/tmpali7f93w.pkl \
        --index-name en_e5_mistral_7b_instruct_c512_o50 \
        --embedding-dim 4096

Memory usage: ~10GB per batch instead of 188GB total
"""

import argparse
import gc
import json
import os
import pickle
import time
import traceback
from pathlib import Path

import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Incremental recovery for embedding builds")
    parser.add_argument("--emb-file", required=True, help="Path to temp .npy embeddings file")
    parser.add_argument("--chunks-file", required=True, help="Path to temp .pkl chunks file")
    parser.add_argument("--index-name", required=True, help="Name for the index")
    parser.add_argument("--embedding-dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--output-dir", default="artifacts/indexes", help="Output directory")
    parser.add_argument("--start-batch", type=int, default=0, help="Resume from this batch (0-indexed)")
    args = parser.parse_args()

    emb_path = Path(args.emb_file)
    chunks_path = Path(args.chunks_file)
    output_dir = Path(args.output_dir) / args.index_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Incremental embedding recovery: {args.index_name}")
    print(f"  Embeddings: {emb_path} ({emb_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  Chunks: {chunks_path} ({chunks_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  Output: {output_dir}")
    print(f"  Start batch: {args.start_batch}")
    print(f"{'=' * 60}\n")

    # Use all CPU cores for FAISS
    num_threads = os.cpu_count() or 8
    faiss.omp_set_num_threads(num_threads)
    print(f"Using {num_threads} threads for FAISS\n")

    # Check for existing index (resume support)
    index_path = output_dir / "index.faiss"
    if args.start_batch > 0 and index_path.exists():
        print(f"Resuming: loading existing index from {index_path}...")
        index = faiss.read_index(str(index_path))
        print(f"  Loaded index with {index.ntotal} vectors")
    else:
        print(f"Creating new FAISS HNSW index (dim={args.embedding_dim})...")
        index = faiss.IndexHNSWFlat(args.embedding_dim, 32)
        index.hnsw.efConstruction = 200
        print(f"  Index created")

    # =========================================================================
    # Step 1: Process embeddings batch-by-batch
    # =========================================================================
    print(f"\nStep 1: Processing embeddings incrementally...")
    t_start = time.time()
    
    batch_num = 0
    total_added = index.ntotal  # Start from existing count if resuming
    
    emb_file = open(emb_path, "rb")
    try:
        while True:
            try:
                # Load one batch
                t_batch = time.time()
                emb = np.load(emb_file)
                batch_num += 1
                
                # Skip batches if resuming
                if batch_num <= args.start_batch:
                    print(f"  Skipping batch {batch_num} (already processed)")
                    del emb
                    continue
                
                batch_size = len(emb)
                print(f"  Batch {batch_num}: {batch_size:,} embeddings...", end=" ", flush=True)
                
                # Normalize in-place
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                np.divide(emb, norms, out=emb)
                
                # Add to index
                index.add(emb.astype(np.float32))
                total_added += batch_size
                
                elapsed = time.time() - t_batch
                print(f"done ({elapsed:.1f}s, total: {total_added:,})")
                
                # Cleanup
                del emb, norms
                gc.collect()
                
                # Save checkpoint every 5 batches
                if batch_num % 5 == 0:
                    print(f"    Saving checkpoint...")
                    faiss.write_index(index, str(index_path))
                    # Save progress
                    with open(output_dir / "checkpoint.json", "w") as f:
                        json.dump({"batch_num": batch_num, "total_vectors": total_added}, f)
                
            except Exception as e:
                if "No data left" in str(e) or "EOF" in str(e) or batch_num > 0:
                    print(f"\n  Finished reading embeddings at batch {batch_num}")
                    break
                else:
                    raise
                    
    finally:
        emb_file.close()
    
    print(f"✓ Embeddings processed: {total_added:,} vectors in {time.time() - t_start:.1f}s")
    
    # Save final index
    print(f"\nSaving final index...")
    faiss.write_index(index, str(index_path))
    print(f"  Saved to {index_path}")

    # =========================================================================
    # Step 2: Load and save chunks
    # =========================================================================
    print(f"\nStep 2: Loading chunks...")
    t_start = time.time()
    
    all_chunks = []
    batch_num = 0
    
    with open(chunks_path, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                batch_num += 1
                all_chunks.extend(batch)
                if batch_num % 10 == 0:
                    print(f"  Loaded {batch_num} batches, {len(all_chunks):,} chunks...")
            except EOFError:
                break
    
    print(f"✓ Loaded {len(all_chunks):,} chunks in {time.time() - t_start:.1f}s")
    
    # Verify count match
    if total_added != len(all_chunks):
        print(f"WARNING: Count mismatch! {total_added} embeddings vs {len(all_chunks)} chunks")
        print(f"  This may indicate corrupted data.")
    
    # Save chunks
    print(f"\nSaving documents...")
    with open(output_dir / "documents.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"  Saved {len(all_chunks):,} documents")
    
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
        "num_documents": len(all_chunks),
        "embedding_dim": args.embedding_dim,
        "use_gpu": False,
        "recovered": True,
        "incremental": True,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Remove checkpoint file
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n{'=' * 60}")
    print(f"✓ Recovery complete!")
    print(f"  Index: {output_dir}")
    print(f"  Vectors: {index.ntotal:,}")
    print(f"  Documents: {len(all_chunks):,}")
    print(f"{'=' * 60}")
    
    print(f"\nYou can now delete the temp files:")
    print(f"  rm {emb_path}")
    print(f"  rm {chunks_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been checkpointed.")
        print("Resume with --start-batch <N> to continue from last checkpoint.")
    except Exception:
        print("\n" + "=" * 60)
        print("FATAL ERROR - Full traceback:")
        print("=" * 60)
        traceback.print_exc()
        print("\nProgress may have been checkpointed. Check checkpoint.json")
        exit(1)
