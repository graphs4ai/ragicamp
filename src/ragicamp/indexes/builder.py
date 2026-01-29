"""Index building utilities.

This module handles building and managing indexes:
- EmbeddingIndex: Dense vector indexes for semantic search
- HierarchicalIndex: Parent-child chunk indexes
- Retriever configs that reference shared indexes
"""

import gc
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager
from ragicamp.utils.resource_manager import ResourceManager

logger = get_logger(__name__)


def get_embedding_index_name(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_version: str,
) -> str:
    """Compute canonical name for a shared embedding index.

    Indexes with the same config get the same name, enabling reuse.

    Args:
        embedding_model: Embedding model name
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        corpus_version: Corpus version string

    Returns:
        Canonical index name
    """
    # Normalize model name
    model_short = embedding_model.split("/")[-1].replace("-", "_").lower()
    # Extract corpus short name (e.g., "20231101.en" -> "en")
    corpus_short = corpus_version.split(".")[-1] if "." in corpus_version else corpus_version
    return f"{corpus_short}_{model_short}_c{chunk_size}_o{chunk_overlap}"


def build_embedding_index(
    index_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_config: Dict[str, Any],
    doc_batch_size: int = 5000,
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
        doc_batch_size: Documents to process per batch

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
    print(f"  Batch size: {doc_batch_size} docs")
    print(f"{'='*60}")

    # Initialize corpus (streaming mode)
    corpus_cfg = CorpusConfig(
        name=f"wikipedia_{corpus_config.get('version', 'simple')}",
        source=corpus_config.get("source", "wikimedia/wikipedia"),
        version=corpus_config.get("version", "20231101.simple"),
        max_docs=corpus_config.get("max_docs"),
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
        strategy="recursive",
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
                    embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64)
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
                embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64)
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


def build_hierarchical_index(
    retriever_config: Dict[str, Any],
    corpus_config: Dict[str, Any],
    doc_batch_size: int = 1000,
) -> str:
    """Build a hierarchical index with batched processing.

    Hierarchical indexes have unique parent/child chunking, so they
    can't share indexes with other retrievers.

    Args:
        retriever_config: Retriever configuration with chunk sizes
        corpus_config: Corpus configuration
        doc_batch_size: Documents to process per batch

    Returns:
        Retriever name
    """
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from ragicamp.corpus import CorpusConfig, WikipediaCorpus
    from ragicamp.rag.chunking.hierarchical import HierarchicalChunker
    from ragicamp.retrievers.base import Document

    manager = get_artifact_manager()
    retriever_name = retriever_config["name"]
    embedding_model = retriever_config.get("embedding_model", "all-MiniLM-L6-v2")

    parent_chunk_size = retriever_config.get("parent_chunk_size", 1024)
    child_chunk_size = retriever_config.get("child_chunk_size", 256)
    parent_overlap = retriever_config.get("parent_overlap", 100)
    child_overlap = retriever_config.get("child_overlap", 50)

    print(f"\n{'='*60}")
    print(f"Building hierarchical index: {retriever_name}")
    print(f"  Parent chunks: {parent_chunk_size}")
    print(f"  Child chunks: {child_chunk_size}")
    print(f"  Batch size: {doc_batch_size} docs")
    print(f"{'='*60}")

    corpus_cfg = CorpusConfig(
        name=f"wikipedia_{corpus_config.get('version', 'simple')}",
        source=corpus_config.get("source", "wikimedia/wikipedia"),
        version=corpus_config.get("version", "20231101.simple"),
        max_docs=corpus_config.get("max_docs"),
    )
    corpus = WikipediaCorpus(corpus_cfg)

    # Initialize chunker
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_overlap,
    )

    print("Processing documents in batches...")
    encoder = SentenceTransformer(embedding_model)
    embedding_dim = encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)

    # Save to indexes directory
    temp_dir = manager.get_embedding_index_path(retriever_name)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Temporary files for incremental saving
    temp_parent_file = tempfile.NamedTemporaryFile(
        delete=False, suffix='.pkl', dir=str(temp_dir)
    )
    temp_child_file = tempfile.NamedTemporaryFile(
        delete=False, suffix='.pkl', dir=str(temp_dir)
    )
    temp_mapping_file = tempfile.NamedTemporaryFile(
        delete=False, suffix='.pkl', dir=str(temp_dir)
    )
    temp_parent_file.close()
    temp_child_file.close()
    temp_mapping_file.close()

    parent_file = open(temp_parent_file.name, 'ab')
    child_file = open(temp_child_file.name, 'ab')
    mapping_file = open(temp_mapping_file.name, 'ab')

    total_docs = 0
    doc_batch = []
    parent_count = 0
    child_count = 0

    try:
        for doc in corpus.load():
            doc_with_id = Document(
                id=f"wiki_{total_docs}",
                text=doc.text,
                metadata=doc.metadata,
            )
            doc_batch.append(doc_with_id)
            total_docs += 1

            if len(doc_batch) >= doc_batch_size:
                parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                    iter(doc_batch)
                )

                pickle.dump(batch_child_to_parent, mapping_file)
                mapping_file.flush()
                del batch_child_to_parent

                if child_chunks:
                    child_texts = [c.text for c in child_chunks]
                    embeddings = encoder.encode(child_texts, show_progress_bar=False, batch_size=64)
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    index.add(embeddings.astype("float32"))

                    del child_texts, embeddings
                    gc.collect()

                    pickle.dump(child_chunks, child_file)
                    child_file.flush()
                    child_count += len(child_chunks)
                    del child_chunks
                    gc.collect()

                pickle.dump(parent_chunks, parent_file)
                parent_file.flush()
                parent_count += len(parent_chunks)
                del parent_chunks
                gc.collect()

                print(
                    f"  Processed {total_docs} docs → {parent_count} parents, "
                    f"{child_count} children (index: {index.ntotal})"
                )
                doc_batch = []

        # Process remaining docs
        if doc_batch:
            parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                iter(doc_batch)
            )

            pickle.dump(batch_child_to_parent, mapping_file)

            if child_chunks:
                child_texts = [c.text for c in child_chunks]
                embeddings = encoder.encode(child_texts, show_progress_bar=False, batch_size=64)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                index.add(embeddings.astype("float32"))
                pickle.dump(child_chunks, child_file)
                child_count += len(child_chunks)
                del child_texts, embeddings

            pickle.dump(parent_chunks, parent_file)
            parent_count += len(parent_chunks)

            del doc_batch, parent_chunks, child_chunks, batch_child_to_parent

        print(f"✓ Processing complete: {total_docs} docs → {parent_count} parents, {child_count} children")

    finally:
        parent_file.close()
        child_file.close()
        mapping_file.close()

    # Copy temp files to final location
    print("Copying files to final location...")
    index_path = manager.get_embedding_index_path(retriever_name)
    index_path.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(temp_parent_file.name, index_path / "parent_docs.pkl")
    shutil.copy2(temp_child_file.name, index_path / "child_docs.pkl")

    # Load and reconstruct data for final save
    print("Loading documents for final save...")
    all_parent_docs = []
    all_child_docs = []
    child_to_parent = {}
    parent_id_to_idx = {}

    idx = 0
    with open(temp_parent_file.name, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
                all_parent_docs.extend(batch)
                for doc in batch:
                    parent_id_to_idx[doc.id] = idx
                    idx += 1
            except EOFError:
                break

    with open(temp_child_file.name, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
                all_child_docs.extend(batch)
            except EOFError:
                break

    with open(temp_mapping_file.name, 'rb') as f:
        while True:
            try:
                batch_mapping = pickle.load(f)
                child_to_parent.update(batch_mapping)
            except EOFError:
                break

    # Save child_to_parent mapping
    with open(index_path / "child_to_parent.pkl", "wb") as f:
        pickle.dump(child_to_parent, f)

    # Save FAISS index
    faiss.write_index(index, str(index_path / "child_index.faiss"))

    # Save index config
    index_config = {
        "name": retriever_name,
        "type": "hierarchical",
        "embedding_model": embedding_model,
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
        "hierarchical_index": retriever_name,
        "embedding_model": embedding_model,
        "parent_chunk_size": parent_chunk_size,
        "child_chunk_size": child_chunk_size,
        "num_parents": parent_count,
        "num_children": child_count,
    }
    manager.save_json(retriever_config_data, retriever_path / "config.json")

    # Clean up temp files
    os.unlink(temp_parent_file.name)
    os.unlink(temp_child_file.name)
    os.unlink(temp_mapping_file.name)

    # Clean up
    del all_parent_docs, all_child_docs, child_to_parent, parent_id_to_idx
    del index, encoder, chunker
    gc.collect()

    print(f"✓ Saved hierarchical index: {retriever_name}")
    return retriever_name


def build_retriever_from_index(
    retriever_config: Dict[str, Any],
    embedding_index_name: str,
) -> str:
    """Build a retriever config that uses a shared embedding index.

    For dense retrievers, this just creates a config pointing to the index.
    For hybrid retrievers, this also builds the BM25 sparse index.

    Args:
        retriever_config: Retriever configuration
        embedding_index_name: Name of the shared embedding index to use

    Returns:
        Retriever name
    """
    manager = get_artifact_manager()
    retriever_name = retriever_config["name"]
    retriever_type = retriever_config.get("type", "dense")

    print(f"  Creating retriever: {retriever_name} (type: {retriever_type})")

    # Load the shared index config
    index_path = manager.get_embedding_index_path(embedding_index_name)
    index_config = manager.load_json(index_path / "config.json")

    # Create retriever directory
    retriever_path = manager.get_retriever_path(retriever_name)

    # Create retriever config that references the shared index
    retriever_cfg = {
        "name": retriever_name,
        "type": retriever_type,
        "embedding_index": embedding_index_name,
        "embedding_model": index_config["embedding_model"],
        "embedding_dim": index_config["embedding_dim"],
        "num_documents": index_config["num_documents"],
        "index_type": "flat",
    }

    if retriever_type == "hybrid":
        retriever_cfg["alpha"] = retriever_config.get("alpha", 0.5)
        # For hybrid, we need to build the BM25 index
        print(f"    Building BM25 index for hybrid retriever...")
        with open(index_path / "documents.pkl", "rb") as f:
            documents = pickle.load(f)

        from ragicamp.retrievers.sparse import SparseRetriever
        sparse = SparseRetriever(name=f"{retriever_name}_sparse")
        sparse.index_documents(documents)

        # Save sparse index components
        with open(retriever_path / "sparse_vectorizer.pkl", "wb") as f:
            pickle.dump(sparse.vectorizer, f)
        with open(retriever_path / "sparse_matrix.pkl", "wb") as f:
            pickle.dump(sparse.doc_vectors, f)

        retriever_cfg["has_sparse"] = True
        del documents, sparse

    manager.save_json(retriever_cfg, retriever_path / "config.json")

    return retriever_name


def ensure_indexes_exist(
    retriever_configs: List[Dict[str, Any]],
    corpus_config: Dict[str, Any],
    build_if_missing: bool = True,
) -> List[str]:
    """Ensure all required indexes exist, building missing ones if configured.

    Uses shared embedding indexes to avoid redundant computation.

    Args:
        retriever_configs: List of retriever configuration dicts
        corpus_config: Corpus configuration for building indexes
        build_if_missing: Whether to build missing indexes

    Returns:
        List of retriever names that are ready to use
    """
    manager = get_artifact_manager()
    corpus_version = corpus_config.get("version", "20231101.simple")

    # Step 1: Identify unique embedding indexes needed
    index_to_retrievers: Dict[str, List[Dict[str, Any]]] = {}

    for config in retriever_configs:
        retriever_type = config.get("type", "dense")

        if retriever_type == "hierarchical":
            index_name = config["name"]
        else:
            index_name = get_embedding_index_name(
                embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50),
                corpus_version=corpus_version,
            )

        if index_name not in index_to_retrievers:
            index_to_retrievers[index_name] = []
        index_to_retrievers[index_name].append(config)

    print(f"\n📊 Index analysis:")
    print(f"   {len(retriever_configs)} retrievers → {len(index_to_retrievers)} unique embedding indexes")

    # Step 2: Build missing embedding indexes
    ready_indexes = []
    for index_name, retrievers in index_to_retrievers.items():
        first_retriever = retrievers[0]
        retriever_type = first_retriever.get("type", "dense")

        if retriever_type == "hierarchical":
            if manager.embedding_index_exists(index_name):
                print(f"   ✓ {index_name} (hierarchical, exists)")
                ready_indexes.append(index_name)
            elif build_if_missing:
                print(f"   📦 Building {index_name} (hierarchical)...")
                build_hierarchical_index(first_retriever, corpus_config)
                ready_indexes.append(index_name)
            else:
                raise FileNotFoundError(f"Missing hierarchical index: {index_name}")
        else:
            if manager.embedding_index_exists(index_name):
                print(f"   ✓ {index_name} (shared, exists) → {len(retrievers)} retriever(s)")
                ready_indexes.append(index_name)
            elif build_if_missing:
                print(f"   📦 Building {index_name} (shared) → {len(retrievers)} retriever(s)")
                build_embedding_index(
                    index_name=index_name,
                    embedding_model=first_retriever.get("embedding_model", "all-MiniLM-L6-v2"),
                    chunk_size=first_retriever.get("chunk_size", 512),
                    chunk_overlap=first_retriever.get("chunk_overlap", 50),
                    corpus_config=corpus_config,
                )
                ready_indexes.append(index_name)
                ResourceManager.clear_gpu_memory()
            else:
                raise FileNotFoundError(f"Missing embedding index: {index_name}")

    # Step 3: Create retriever configs that reference the shared indexes
    ready_retrievers = []
    for index_name, retrievers in index_to_retrievers.items():
        first_retriever = retrievers[0]
        retriever_type = first_retriever.get("type", "dense")

        if retriever_type == "hierarchical":
            ready_retrievers.append(first_retriever["name"])
        else:
            for config in retrievers:
                if not manager.index_exists(config["name"]):
                    build_retriever_from_index(config, index_name)
                ready_retrievers.append(config["name"])

    return ready_retrievers
