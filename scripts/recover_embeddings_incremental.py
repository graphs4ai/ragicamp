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
                    # VmRSS is in kB
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Incremental recovery for embedding builds")
    parser.add_argument("--emb-file", required=True, help="Path to temp .npy embeddings file")
    parser.add_argument("--chunks-file", required=True, help="Path to temp .pkl chunks file")
    parser.add_argument("--index-name", required=True, help="Name for the index")
    parser.add_argument("--embedding-dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--output-dir", default="artifacts/indexes", help="Output directory")
    parser.add_argument("--start-batch", type=int, default=0, help="Resume from this batch (0-indexed)")
    parser.add_argument("--num-threads", type=int, default=32, help="FAISS threads (default: 32)")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint every N batches")
    args = parser.parse_args()

    emb_path = Path(args.emb_file)
    chunks_path = Path(args.chunks_file)
    output_dir = Path(args.output_dir) / args.index_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Incremental embedding recovery: {args.index_name}", flush=True)
    print(f"  Embeddings: {emb_path} ({emb_path.stat().st_size / 1e9:.1f} GB)", flush=True)
    print(f"  Chunks: {chunks_path} ({chunks_path.stat().st_size / 1e9:.1f} GB)", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Start batch: {args.start_batch}", flush=True)
    print(f"  Threads: {args.num_threads}", flush=True)
    print(f"  Checkpoint every: {args.checkpoint_every} batch(es)", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # Reduced thread count - 288 may cause issues
    faiss.omp_set_num_threads(args.num_threads)
    print(f"Using {args.num_threads} threads for FAISS\n", flush=True)

    # Check for existing index (resume support)
    index_path = output_dir / "index.faiss"
    if args.start_batch > 0 and index_path.exists():
        print(f"Resuming: loading existing index from {index_path}...", flush=True)
        index = faiss.read_index(str(index_path))
        print(f"  Loaded index with {index.ntotal} vectors", flush=True)
    else:
        print(f"Creating new FAISS HNSW index (dim={args.embedding_dim})...", flush=True)
        index = faiss.IndexHNSWFlat(args.embedding_dim, 32)
        index.hnsw.efConstruction = 200
        print(f"  Index created", flush=True)

    # =========================================================================
    # Step 1: Process embeddings batch-by-batch
    # =========================================================================
    print(f"\nStep 1: Processing embeddings incrementally...", flush=True)
    t_start = time.time()
    
    batch_num = 0
    total_added = index.ntotal  # Start from existing count if resuming
    last_checkpoint_batch = args.start_batch
    
    emb_file = open(emb_path, "rb")
    try:
        while True:
            # Read one batch
            batch_num += 1
            
            try:
                emb = np.load(emb_file)
            except ValueError as e:
                # End of file
                print(f"\n  Finished reading embeddings at batch {batch_num - 1}", flush=True)
                break
            
            # Skip batches if resuming
            if batch_num <= args.start_batch:
                print(f"  Skipping batch {batch_num} (already processed)", flush=True)
                del emb
                continue
            
            batch_size = len(emb)
            mem_gb = get_memory_gb()
            print(f"  Batch {batch_num}: {batch_size:,} embeddings (mem: {mem_gb:.1f}GB)...", end=" ", flush=True)
            sys.stdout.flush()
            
            # Normalize in-place
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            np.divide(emb, norms, out=emb)
            del norms
            
            # Add to index - this is where it might crash
            try:
                index.add(emb.astype(np.float32))
            except Exception as e:
                print(f"\n*** FAISS add() failed: {e}", flush=True)
                print(f"    Saving emergency checkpoint at batch {batch_num - 1}...", flush=True)
                faiss.write_index(index, str(index_path))
                with open(output_dir / "checkpoint.json", "w") as f:
                    json.dump({"batch_num": batch_num - 1, "total_vectors": total_added}, f)
                raise
            
            total_added += batch_size
            elapsed = time.time() - t_start
            batch_time = elapsed / (batch_num - args.start_batch)
            
            print(f"done (avg: {batch_time:.1f}s/batch, total: {total_added:,})", flush=True)
            
            # Cleanup
            del emb
            gc.collect()
            
            # Save checkpoint
            if batch_num % args.checkpoint_every == 0:
                print(f"    Saving checkpoint at batch {batch_num}...", flush=True)
                faiss.write_index(index, str(index_path))
                with open(output_dir / "checkpoint.json", "w") as f:
                    json.dump({"batch_num": batch_num, "total_vectors": total_added}, f)
                last_checkpoint_batch = batch_num
                    
    except Exception as e:
        print(f"\n*** Exception during processing: {e}", flush=True)
        traceback.print_exc()
        print(f"\n    Last checkpoint: batch {last_checkpoint_batch}", flush=True)
        raise
    finally:
        emb_file.close()
    
    print(f"✓ Embeddings processed: {total_added:,} vectors in {time.time() - t_start:.1f}s", flush=True)
    
    # Save final index
    print(f"\nSaving final index...", flush=True)
    faiss.write_index(index, str(index_path))
    print(f"  Saved to {index_path}", flush=True)

    # =========================================================================
    # Step 2: Load and save chunks
    # =========================================================================
    print(f"\nStep 2: Loading chunks...", flush=True)
    t_start = time.time()
    
    all_chunks = []
    chunk_batch_num = 0
    
    with open(chunks_path, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                chunk_batch_num += 1
                all_chunks.extend(batch)
                if chunk_batch_num % 10 == 0:
                    print(f"  Loaded {chunk_batch_num} batches, {len(all_chunks):,} chunks...", flush=True)
            except EOFError:
                break
    
    print(f"✓ Loaded {len(all_chunks):,} chunks in {time.time() - t_start:.1f}s", flush=True)
    
    # Verify count match
    if total_added != len(all_chunks):
        print(f"WARNING: Count mismatch! {total_added} embeddings vs {len(all_chunks)} chunks", flush=True)
        print(f"  This may indicate corrupted data.", flush=True)
    
    # Save chunks
    print(f"\nSaving documents...", flush=True)
    with open(output_dir / "documents.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"  Saved {len(all_chunks):,} documents", flush=True)
    
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
    
    # Remove checkpoint file on success
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n{'=' * 60}", flush=True)
    print(f"✓ Recovery complete!", flush=True)
    print(f"  Index: {output_dir}", flush=True)
    print(f"  Vectors: {index.ntotal:,}", flush=True)
    print(f"  Documents: {len(all_chunks):,}", flush=True)
    print(f"{'=' * 60}", flush=True)
    
    print(f"\nYou can now delete the temp files:", flush=True)
    print(f"  rm {emb_path}", flush=True)
    print(f"  rm {chunks_path}", flush=True)


if __name__ == "__main__":
    # Flush stdout/stderr on every write
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been checkpointed.", flush=True)
        print("Resume with --start-batch <N> to continue from last checkpoint.", flush=True)
    except Exception:
        print("\n" + "=" * 60, flush=True)
        print("FATAL ERROR - Full traceback:", flush=True)
        print("=" * 60, flush=True)
        traceback.print_exc()
        print("\nProgress may have been checkpointed. Check checkpoint.json", flush=True)
        exit(1)
