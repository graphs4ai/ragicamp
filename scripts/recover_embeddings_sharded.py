#!/usr/bin/env python3
"""Sharded HNSW recovery script for memory-constrained environments.

Builds multiple HNSW shards that each fit in memory.
At query time, all shards are searched and results merged.

Usage:
    python scripts/recover_embeddings_sharded.py \
        --num-shards 3 \
        --embedding-dim 4096

Memory usage: ~65GB per shard instead of 188GB total
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import faiss
import numpy as np


def get_memory_gb():
    """Get current memory usage in GB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Sharded HNSW recovery")
    parser.add_argument("--emb-file", default="artifacts/indexes/tmppzyovtz_.npy",
                        help="Path to temp .npy embeddings file")
    parser.add_argument("--chunks-file", default="artifacts/indexes/tmpali7f93w.pkl",
                        help="Path to temp .pkl chunks file")
    parser.add_argument("--index-name", default="en_e5_mistral_7b_instruct_c512_o50",
                        help="Name for the index")
    parser.add_argument("--embedding-dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--output-dir", default="artifacts/indexes", help="Output directory")
    parser.add_argument("--num-shards", type=int, default=3, help="Number of shards to create")
    parser.add_argument("--num-threads", type=int, default=32, help="FAISS threads")
    parser.add_argument("--current-shard", type=int, default=None, 
                        help="Build only this shard (0-indexed). If not specified, builds all.")
    args = parser.parse_args()

    emb_path = Path(args.emb_file)
    chunks_path = Path(args.chunks_file)
    output_dir = Path(args.output_dir) / args.index_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: count total batches
    print(f"Counting batches in {emb_path}...", flush=True)
    total_batches = 0
    batch_sizes = []
    with open(emb_path, "rb") as f:
        while True:
            try:
                emb = np.load(f)
                batch_sizes.append(len(emb))
                total_batches += 1
                if total_batches % 10 == 0:
                    print(f"  Counted {total_batches} batches...", flush=True)
                del emb
            except (ValueError, EOFError):
                break
    
    total_vectors = sum(batch_sizes)
    print(f"  Found {total_batches} batches, {total_vectors:,} total vectors", flush=True)
    
    # Calculate shard boundaries
    vectors_per_shard = total_vectors // args.num_shards
    shard_boundaries = []  # List of (start_batch, end_batch, start_vector, end_vector)
    
    current_vector = 0
    current_batch = 0
    for shard_idx in range(args.num_shards):
        start_batch = current_batch
        start_vector = current_vector
        
        # Find end of this shard
        if shard_idx == args.num_shards - 1:
            # Last shard gets everything remaining
            end_batch = total_batches
            end_vector = total_vectors
        else:
            # Find batch that crosses the boundary
            target_vector = (shard_idx + 1) * vectors_per_shard
            while current_batch < total_batches and current_vector < target_vector:
                current_vector += batch_sizes[current_batch]
                current_batch += 1
            end_batch = current_batch
            end_vector = current_vector
        
        shard_boundaries.append((start_batch, end_batch, start_vector, end_vector))
        current_batch = end_batch
        current_vector = end_vector
    
    print(f"\nShard plan ({args.num_shards} shards):", flush=True)
    for i, (sb, eb, sv, ev) in enumerate(shard_boundaries):
        shard_vectors = ev - sv
        shard_mem_gb = shard_vectors * args.embedding_dim * 4 / 1e9
        print(f"  Shard {i}: batches {sb+1}-{eb}, vectors {sv:,}-{ev:,} ({shard_vectors:,} vectors, ~{shard_mem_gb:.1f}GB)", flush=True)
    
    # Determine which shards to build
    if args.current_shard is not None:
        shards_to_build = [args.current_shard]
    else:
        shards_to_build = list(range(args.num_shards))
    
    faiss.omp_set_num_threads(args.num_threads)
    print(f"\nUsing {args.num_threads} threads for FAISS", flush=True)
    
    # Build each shard
    for shard_idx in shards_to_build:
        start_batch, end_batch, start_vector, end_vector = shard_boundaries[shard_idx]
        shard_dir = output_dir / f"shard_{shard_idx}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = shard_dir / "index.faiss"
        checkpoint_path = shard_dir / "checkpoint.json"
        
        # Check for checkpoint
        resume_batch = 0
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                cp = json.load(f)
            resume_batch = cp.get("batch_num", 0)
            if resume_batch >= end_batch:
                print(f"\n✓ Shard {shard_idx} already complete, skipping", flush=True)
                continue
        
        print(f"\n{'=' * 60}", flush=True)
        print(f"Building shard {shard_idx} (batches {start_batch+1}-{end_batch})", flush=True)
        print(f"{'=' * 60}", flush=True)
        
        # Load existing index or create new
        if resume_batch > start_batch and index_path.exists():
            print(f"Resuming from batch {resume_batch}...", flush=True)
            index = faiss.read_index(str(index_path))
            print(f"  Loaded {index.ntotal} vectors", flush=True)
        else:
            print(f"Creating new HNSW index (dim={args.embedding_dim})...", flush=True)
            index = faiss.IndexHNSWFlat(args.embedding_dim, 32)
            index.hnsw.efConstruction = 200
            resume_batch = start_batch
        
        # Process batches for this shard
        t_start = time.time()
        batch_num = 0
        shard_chunks = []
        
        with open(emb_path, "rb") as f:
            while batch_num < end_batch:
                try:
                    emb = np.load(f)
                    batch_num += 1
                    
                    # Skip batches before this shard
                    if batch_num <= start_batch:
                        del emb
                        continue
                    
                    # Skip already processed batches
                    if batch_num <= resume_batch:
                        del emb
                        continue
                    
                    batch_size = len(emb)
                    mem_gb = get_memory_gb()
                    print(f"  Batch {batch_num}: {batch_size:,} embeddings (mem: {mem_gb:.1f}GB)...", end=" ", flush=True)
                    
                    # Normalize
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    np.divide(emb, norms, out=emb)
                    del norms
                    
                    # Add to index
                    index.add(emb.astype(np.float32))
                    
                    print(f"done (total: {index.ntotal:,})", flush=True)
                    
                    del emb
                    gc.collect()
                    
                    # Checkpoint every batch
                    faiss.write_index(index, str(index_path))
                    with open(checkpoint_path, "w") as cf:
                        json.dump({"batch_num": batch_num, "total_vectors": index.ntotal}, cf)
                    
                except (ValueError, EOFError):
                    break
        
        elapsed = time.time() - t_start
        print(f"✓ Shard {shard_idx} complete: {index.ntotal:,} vectors in {elapsed:.1f}s", flush=True)
        
        # Save final index
        faiss.write_index(index, str(index_path))
        
        # Save shard config
        shard_config = {
            "shard_idx": shard_idx,
            "start_vector": start_vector,
            "end_vector": end_vector,
            "num_vectors": index.ntotal,
            "embedding_dim": args.embedding_dim,
        }
        with open(shard_dir / "config.json", "w") as f:
            json.dump(shard_config, f, indent=2)
        
        # Remove checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        del index
        gc.collect()
    
    # =========================================================================
    # Load and save chunks (only if all shards are done)
    # =========================================================================
    all_shards_done = all(
        (output_dir / f"shard_{i}" / "index.faiss").exists() 
        for i in range(args.num_shards)
    )
    
    if not all_shards_done:
        print(f"\nNot all shards complete. Run again to build remaining shards.", flush=True)
        print(f"Or run with --current-shard N to build a specific shard.", flush=True)
        return
    
    # Check if chunks already saved
    if (output_dir / "documents.pkl").exists():
        print(f"\n✓ Documents already saved", flush=True)
    else:
        print(f"\nLoading and saving chunks...", flush=True)
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
                        print(f"  Loaded {batch_num} batches, {len(all_chunks):,} chunks...", flush=True)
                except EOFError:
                    break
        
        print(f"✓ Loaded {len(all_chunks):,} chunks in {time.time() - t_start:.1f}s", flush=True)
        
        with open(output_dir / "documents.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
        print(f"  Saved to {output_dir / 'documents.pkl'}", flush=True)
    
    # Save main config
    config = {
        "name": args.index_name,
        "type": "embedding",
        "embedding_model": "intfloat/e5-mistral-7b-instruct",
        "embedding_backend": "vllm",
        "index_type": "hnsw_sharded",
        "num_shards": args.num_shards,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "chunking_strategy": "recursive",
        "num_documents": total_vectors,
        "embedding_dim": args.embedding_dim,
        "use_gpu": False,
        "recovered": True,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'=' * 60}", flush=True)
    print(f"✓ All shards complete!", flush=True)
    print(f"  Index: {output_dir}", flush=True)
    print(f"  Shards: {args.num_shards}", flush=True)
    print(f"  Total vectors: {total_vectors:,}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress checkpointed.", flush=True)
    except Exception:
        print("\n" + "=" * 60, flush=True)
        print("FATAL ERROR:", flush=True)
        print("=" * 60, flush=True)
        traceback.print_exc()
        exit(1)
