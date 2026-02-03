"""Embedding index builder.

Builds dense vector indexes for semantic search.
Supports both sentence-transformers and vLLM embedding backends.

Uses pipelined processing to overlap GPU embedding with CPU post-processing.
"""

import gc
import os
import pickle
import tempfile
import time
from typing import Any, Iterator

# Disable tokenizers parallelism to avoid fork warnings
# Must be set before importing transformers/tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)

# Maximum document size in characters (larger docs are truncated)
MAX_DOC_CHARS = 100_000


def chunk_documents(
    doc_batch: list,
    chunker,
    Document,
    tqdm,
) -> tuple[list, int]:
    """Chunk a batch of documents.

    Args:
        doc_batch: List of documents to chunk
        chunker: DocumentChunker instance
        Document: Document class for creating truncated docs
        tqdm: tqdm function for progress bar

    Returns:
        Tuple of (chunks list, truncated count)
    """
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

    return batch_chunks, truncated


def generate_doc_batches(
    corpus,
    doc_batch_size: int,
) -> Iterator[tuple[list, bool]]:
    """Generate batches of documents from corpus.

    Args:
        corpus: Corpus to load documents from
        doc_batch_size: Number of documents per batch

    Yields:
        Tuples of (doc_batch, is_final)
    """
    doc_batch = []

    for doc in corpus.load():
        doc_batch.append(doc)

        if len(doc_batch) >= doc_batch_size:
            yield doc_batch, False
            doc_batch = []

    # Yield remaining docs as final batch
    if doc_batch:
        yield doc_batch, True


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

    # Initialize encoder based on backend
    print(f"Loading embedding model ({embedding_backend})...")

    if embedding_backend == "vllm":
        # vLLM backend - uses continuous batching for high throughput
        from ragicamp.models.vllm_embedder import VLLMEmbedder

        encoder = VLLMEmbedder(
            model_name=embedding_model,
            gpu_memory_fraction=vllm_gpu_memory_fraction,
            enforce_eager=False,  # Use CUDA graphs for speed
        )
        embedding_dim = encoder.get_sentence_embedding_dimension()
        print(
            f"  vLLM embedder loaded (dim={embedding_dim}, gpu_mem={vllm_gpu_memory_fraction:.0%})"
        )
    elif embedding_backend == "vllm_server":
        # vLLM server backend - subprocess server with async HTTP for overlap
        from ragicamp.models.vllm_embedder import VLLMServerEmbedder

        encoder = VLLMServerEmbedder(
            model_name=embedding_model,
            gpu_memory_fraction=vllm_gpu_memory_fraction,
        )
        embedding_dim = encoder.get_sentence_embedding_dimension()
        print(
            f"  vLLM server started (dim={embedding_dim}, gpu_mem={vllm_gpu_memory_fraction:.0%})"
        )
    else:
        # sentence_transformers backend - works with any HuggingFace model
        from sentence_transformers import SentenceTransformer

        model_kwargs = {}
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Flash Attention 2 available, enabling")
        except ImportError:
            print("  flash-attn not installed, using default attention")

        encoder = SentenceTransformer(
            embedding_model,
            trust_remote_code=True,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        embedding_dim = encoder.get_sentence_embedding_dimension()

        # Apply torch.compile() for additional speedup (PyTorch 2.0+)
        try:
            import torch

            if hasattr(torch, "compile") and torch.cuda.is_available():
                encoder = torch.compile(encoder, mode="reduce-overhead")
                print("  Applied torch.compile() to embedding model")
        except Exception as e:
            print(f"  torch.compile() not applied: {e}")

    # Initialize FAISS index (we'll build on CPU first, then optionally move to GPU after all vectors are added)
    # This is more efficient than adding vectors to GPU index incrementally
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
    is_trained = index_type == "flat" or index_type == "hnsw"  # These don't need training

    # Chunking config
    chunk_config = ChunkConfig(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunker = DocumentChunker(chunk_config)

    # Process documents in batches - save incrementally to avoid memory bloat
    temp_docs_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pkl", dir=str(manager.indexes_dir)
    )
    temp_docs_file.close()
    docs_file = open(temp_docs_file.name, "ab")

    total_docs = 0
    total_chunks = 0

    # Pipeline overlaps GPU embedding with CPU post-processing
    from ragicamp.indexes.builders.pipeline import EmbeddingPipeline, EmbeddingResult

    def process_batch(result: EmbeddingResult) -> None:
        """Process embeddings: train index, add vectors, save chunks."""
        nonlocal is_trained

        embeddings = result.embeddings

        # Train IVF index on first batch if needed
        if not is_trained and index_type in ("ivf", "ivfpq"):
            print(f"    Training IVF index on {len(embeddings)} vectors...")
            index.train(embeddings)
            is_trained = True

        # Add to index and save (CPU work - overlaps with next GPU batch)
        index.add(embeddings)
        pickle.dump(result.chunks, docs_file)
        docs_file.flush()

        print(f"    ✓ Indexed {len(embeddings)} vectors (total: {index.ntotal})")
        gc.collect()

    print("Processing documents in batches (pipelined)...")

    try:
        with EmbeddingPipeline(
            encoder=encoder,
            process_fn=process_batch,
            embedding_batch_size=embedding_batch_size,
            normalize=True,
        ) as pipeline:
            batch_num = 0

            for doc_batch, is_final in generate_doc_batches(corpus, doc_batch_size):
                batch_num += 1
                batch_size = len(doc_batch)
                suffix = " (final)" if is_final else ""
                print(f"\n  [Batch {batch_num}] Processing {batch_size} documents{suffix}...")

                # Chunking phase
                t_chunk = time.time()
                batch_chunks, truncated = chunk_documents(
                    doc_batch, chunker, Document, tqdm
                )
                chunk_elapsed = time.time() - t_chunk

                msg = f"    ✓ {len(batch_chunks)} chunks in {chunk_elapsed:.1f}s ({batch_size / chunk_elapsed:.0f} docs/s)"
                if truncated > 0:
                    msg += f" (truncated {truncated})"
                print(msg)

                total_docs += batch_size
                total_chunks += len(batch_chunks)

                if batch_chunks:
                    texts = [c.text for c in batch_chunks]
                    print(f"    Embedding {len(texts)} chunks...")

                    # Submit to pipeline - GPU work starts, previous batch processed
                    pipeline.submit(texts, batch_chunks)

                print(f"  ✓ Total: {total_docs} docs → {total_chunks} chunks")
                del doc_batch, batch_chunks, texts
                gc.collect()

    finally:
        docs_file.close()

    print(f"✓ Embedding complete: {total_docs} docs → {total_chunks} chunks")

    # Set nprobe for IVF indexes
    if index_type in ("ivf", "ivfpq") and hasattr(index, "nprobe"):
        index.nprobe = nprobe
        print(f"  Set nprobe={nprobe} for IVF index")

    # Save to shared indexes directory
    artifact_path = manager.get_embedding_index_path(index_name)

    # Save FAISS index (always save CPU version for portability)
    faiss.write_index(index, str(artifact_path / "index.faiss"))

    # Combine all batches into single pickle file
    print("Combining batches for final save...")
    all_documents = []
    with open(temp_docs_file.name, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                all_documents.extend(batch)
            except EOFError:
                break

    with open(artifact_path / "documents.pkl", "wb") as f:
        pickle.dump(all_documents, f)

    os.unlink(temp_docs_file.name)
    del all_documents
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
        # GPU FAISS settings (used when loading)
        "use_gpu": use_gpu,
        "nlist": nlist,
        "nprobe": nprobe,
    }
    manager.save_json(config, artifact_path / "config.json")

    del index, encoder
    gc.collect()

    print(f"✓ Saved to: {artifact_path}")
    if use_gpu:
        print(f"  Index will be loaded to GPU on next use (use_gpu={use_gpu})")
    return str(artifact_path)
