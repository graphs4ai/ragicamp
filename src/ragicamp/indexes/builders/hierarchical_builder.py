"""Hierarchical index builder.

Builds hierarchical indexes with parent-child chunk relationships.
"""

import gc
import os
import pickle
import tempfile
from typing import Any, Dict

from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


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
