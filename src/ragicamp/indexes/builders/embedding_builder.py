"""Embedding index builder.

Builds dense vector indexes for semantic search using sentence transformers.
"""

import gc
import os
import pickle
import tempfile
from typing import Any, Dict

from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


def build_embedding_index(
    index_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_config: Dict[str, Any],
    chunking_strategy: str = "recursive",
    doc_batch_size: int = 5000,
    embedding_batch_size: int = 64,
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

    Returns:
        Path to the saved index
    """
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from ragicamp.corpus import ChunkConfig, CorpusConfig, DocumentChunker, WikipediaCorpus

    manager = get_artifact_manager()

    print(f"\n{'='*60}")
    print(f"Building shared embedding index: {index_name}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    print(f"  Chunking strategy: {chunking_strategy}")
    print(f"  Doc batch size: {doc_batch_size}, embedding batch size: {embedding_batch_size}")
    print(f"{'='*60}")

    # Initialize corpus (streaming mode)
    # Build metadata dict for WikiRank filter and other options
    corpus_metadata = {}
    if corpus_config.get("wikirank_top_k"):
        corpus_metadata["wikirank_top_k"] = corpus_config["wikirank_top_k"]
    if corpus_config.get("min_chars"):
        corpus_metadata["min_chars"] = corpus_config["min_chars"]

    corpus_cfg = CorpusConfig(
        name=f"wikipedia_{corpus_config.get('version', 'simple')}",
        source=corpus_config.get("source", "wikimedia/wikipedia"),
        version=corpus_config.get("version", "20231101.simple"),
        max_docs=corpus_config.get("max_docs"),
        metadata=corpus_metadata,
    )
    corpus = WikipediaCorpus(corpus_cfg)

    # Initialize encoder
    print("Loading embedding model...")
    encoder = SentenceTransformer(embedding_model)
    embedding_dim = encoder.get_sentence_embedding_dimension()

    # Initialize FAISS index
    index = faiss.IndexFlatIP(embedding_dim)

    # Chunking config
    chunk_config = ChunkConfig(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunker = DocumentChunker(chunk_config)

    # Process documents in batches - save incrementally to avoid memory bloat
    temp_docs_file = tempfile.NamedTemporaryFile(
        delete=False, suffix='.pkl', dir=str(manager.indexes_dir)
    )
    temp_docs_file.close()
    docs_file = open(temp_docs_file.name, 'ab')

    total_docs = 0
    total_chunks = 0
    doc_batch = []

    print("Processing documents in batches...")
    try:
        for doc in corpus.load():
            doc_batch.append(doc)

            if len(doc_batch) >= doc_batch_size:
                batch_chunks = list(chunker.chunk_documents(iter(doc_batch), show_progress=False))
                total_docs += len(doc_batch)
                total_chunks += len(batch_chunks)

                if batch_chunks:
                    texts = [c.text for c in batch_chunks]
                    embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=embedding_batch_size)
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    index.add(embeddings.astype("float32"))

                    # Save chunks incrementally to disk
                    pickle.dump(batch_chunks, docs_file)
                    docs_file.flush()

                print(f"  Processed {total_docs} docs → {total_chunks} chunks (index: {index.ntotal})")
                doc_batch = []
                del batch_chunks, texts, embeddings
                gc.collect()

        # Process remaining docs
        if doc_batch:
            batch_chunks = list(chunker.chunk_documents(iter(doc_batch), show_progress=False))
            total_docs += len(doc_batch)
            total_chunks += len(batch_chunks)

            if batch_chunks:
                texts = [c.text for c in batch_chunks]
                embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=embedding_batch_size)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                index.add(embeddings.astype("float32"))
                pickle.dump(batch_chunks, docs_file)
                docs_file.flush()

            del doc_batch, batch_chunks, texts, embeddings
            gc.collect()
    finally:
        docs_file.close()

    print(f"✓ Embedding complete: {total_docs} docs → {total_chunks} chunks")

    # Save to shared indexes directory
    artifact_path = manager.get_embedding_index_path(index_name)

    faiss.write_index(index, str(artifact_path / "index.faiss"))

    # Combine all batches into single pickle file (for compatibility with load method)
    print("Combining batches for final save...")
    all_documents = []
    with open(temp_docs_file.name, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
                all_documents.extend(batch)
            except EOFError:
                break

    # Save combined documents
    with open(artifact_path / "documents.pkl", "wb") as f:
        pickle.dump(all_documents, f)

    os.unlink(temp_docs_file.name)
    del all_documents
    gc.collect()

    config = {
        "name": index_name,
        "type": "embedding",
        "embedding_model": embedding_model,
        "index_type": "flat",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": chunking_strategy,
        "corpus_version": corpus_config.get("version"),
        "corpus_max_docs": corpus_config.get("max_docs"),
        "num_documents": total_chunks,
        "embedding_dim": embedding_dim,
    }
    manager.save_json(config, artifact_path / "config.json")

    del index, encoder
    gc.collect()

    print(f"✓ Saved to: {artifact_path}")
    return str(artifact_path)
