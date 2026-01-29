"""Study runner for CLI.

Handles running studies from YAML config files.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ragicamp import Experiment

# ============================================================================
# Validation
# ============================================================================

VALID_DATASETS = {"nq", "triviaqa", "hotpotqa", "techqa", "pubmedqa"}
VALID_PROVIDERS = {"hf", "openai", "vllm"}
VALID_QUANTIZATIONS = {"4bit", "8bit", "none"}


class ConfigError(ValueError):
    """Configuration validation error."""

    pass


def validate_model_spec(spec: str) -> None:
    """Validate model specification format.

    Args:
        spec: Model spec like 'hf:google/gemma-2b-it' or 'openai:gpt-4o-mini'

    Raises:
        ConfigError: If spec format is invalid
    """
    if ":" not in spec:
        raise ConfigError(
            f"Invalid model spec: '{spec}'. "
            f"Expected format: 'provider:model_name' (e.g., 'hf:google/gemma-2b-it', 'openai:gpt-4o-mini')"
        )
    provider = spec.split(":")[0]
    if provider not in VALID_PROVIDERS:
        raise ConfigError(
            f"Unknown model provider: '{provider}'. " f"Valid providers: {VALID_PROVIDERS}"
        )


def validate_dataset(name: str) -> None:
    """Validate dataset name.

    Args:
        name: Dataset name like 'nq', 'triviaqa', 'hotpotqa'

    Raises:
        ConfigError: If dataset name is invalid
    """
    if name not in VALID_DATASETS:
        raise ConfigError(f"Unknown dataset: '{name}'. " f"Valid datasets: {VALID_DATASETS}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate study configuration.

    Args:
        config: Study config dict

    Returns:
        List of warning messages (empty if all valid)

    Raises:
        ConfigError: If required fields are missing or invalid
    """
    warnings = []

    # Required fields
    if "name" not in config:
        raise ConfigError("Config missing required field: 'name'")

    # Validate datasets
    datasets = config.get("datasets", [])
    if not datasets:
        warnings.append("No datasets specified")
    for ds in datasets:
        validate_dataset(ds)

    # Validate direct experiments
    direct = config.get("direct", {})
    if direct.get("enabled"):
        if not direct.get("models"):
            warnings.append("Direct experiments enabled but no models specified")
        for model in direct.get("models", []):
            validate_model_spec(model)
        for q in direct.get("quantization", []):
            if q not in VALID_QUANTIZATIONS:
                raise ConfigError(f"Invalid quantization: '{q}'. Valid: {VALID_QUANTIZATIONS}")

    # Validate RAG experiments
    rag = config.get("rag", {})
    if rag.get("enabled"):
        if not rag.get("models"):
            warnings.append("RAG experiments enabled but no models specified")
        if not rag.get("retrievers"):
            warnings.append("RAG experiments enabled but no retrievers specified")
        for model in rag.get("models", []):
            validate_model_spec(model)
        for q in rag.get("quantization", []):
            if q not in VALID_QUANTIZATIONS:
                raise ConfigError(f"Invalid quantization: '{q}'. Valid: {VALID_QUANTIZATIONS}")

    return warnings


from ragicamp.agents import DirectLLMAgent, FixedRAGAgent
from ragicamp.datasets import HotpotQADataset, NaturalQuestionsDataset, TriviaQADataset
from ragicamp.factory import ComponentFactory
from ragicamp.models import HuggingFaceModel, OpenAIModel
from ragicamp.retrievers import DenseRetriever
from ragicamp.utils.artifacts import get_artifact_manager
from ragicamp.utils.resource_manager import ResourceManager

# ============================================================================
# Index building - Shared Embedding Indexes
# ============================================================================


def get_embedding_index_name(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_version: str,
) -> str:
    """Compute canonical name for a shared embedding index.

    Indexes with the same config get the same name, enabling reuse.
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
    import gc
    import pickle

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
    import tempfile
    temp_docs_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=str(manager.indexes_dir))
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
    
    import os
    os.unlink(temp_docs_file.name)
    del all_documents
    gc.collect()

    config = {
        "name": index_name,
        "type": "embedding",  # For EmbeddingIndex.load() compatibility
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


def build_retriever_from_index(
    retriever_config: Dict[str, Any],
    embedding_index_name: str,
) -> str:
    """Build a retriever that uses a shared embedding index.

    For dense retrievers, this just creates a config pointing to the index.
    For hybrid retrievers, this also builds the BM25 sparse index.

    Args:
        retriever_config: Retriever configuration
        embedding_index_name: Name of the shared embedding index to use

    Returns:
        Retriever name
    """
    import pickle

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
        # Load documents and build sparse index
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

    # No symlinks needed - the load() method reads embedding_index from config
    # and loads files directly from the shared index path

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
    # Group retrievers by their embedding index requirements
    index_to_retrievers: Dict[str, List[Dict[str, Any]]] = {}

    for config in retriever_configs:
        retriever_type = config.get("type", "dense")

        # Hierarchical retrievers have different chunking, handle separately
        if retriever_type == "hierarchical":
            # Use retriever name as index name (unique chunking per hierarchical config)
            index_name = config["name"]
        else:
            # Compute canonical index name for dense/hybrid
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

        # Check if this is a hierarchical retriever (special handling)
        if retriever_type == "hierarchical":
            # Check in indexes directory (new format)
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
            # Dense/Hybrid - use shared embedding index
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
            # Hierarchical already saved as retriever
            ready_retrievers.append(first_retriever["name"])
        else:
            # Build retriever configs pointing to shared index
            for config in retrievers:
                if not manager.index_exists(config["name"]):
                    build_retriever_from_index(config, index_name)
                ready_retrievers.append(config["name"])

    return ready_retrievers


def build_hierarchical_index(
    retriever_config: Dict[str, Any],
    corpus_config: Dict[str, Any],
    doc_batch_size: int = 1000,  # Smaller batches to avoid memory spikes
) -> str:
    """Build a hierarchical retriever index with batched processing.

    Hierarchical retrievers have unique parent/child chunking, so they
    can't share indexes with other retrievers. Processes documents in batches
    to avoid memory issues.
    """
    import gc
    import pickle

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from ragicamp.corpus import CorpusConfig, WikipediaCorpus
    from ragicamp.retrievers.base import Document
    from ragicamp.utils.artifacts import get_artifact_manager

    retriever_name = retriever_config["name"]
    embedding_model = retriever_config.get("embedding_model", "all-MiniLM-L6-v2")

    print(f"\n{'='*60}")
    print(f"Building hierarchical index: {retriever_name}")
    print(f"  Parent chunks: {retriever_config.get('parent_chunk_size', 1024)}")
    print(f"  Child chunks: {retriever_config.get('child_chunk_size', 256)}")
    print(f"  Batch size: {doc_batch_size} docs")
    print(f"{'='*60}")

    corpus_cfg = CorpusConfig(
        name=f"wikipedia_{corpus_config.get('version', 'simple')}",
        source=corpus_config.get("source", "wikimedia/wikipedia"),
        version=corpus_config.get("version", "20231101.simple"),
        max_docs=corpus_config.get("max_docs"),
    )
    corpus = WikipediaCorpus(corpus_cfg)

    # Initialize chunker for hierarchical chunking
    from ragicamp.rag.chunking.hierarchical import HierarchicalChunker
    
    parent_chunk_size = retriever_config.get("parent_chunk_size", 1024)
    child_chunk_size = retriever_config.get("child_chunk_size", 256)
    parent_overlap = retriever_config.get("parent_overlap", 100)
    child_overlap = retriever_config.get("child_overlap", 50)
    
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_overlap,
    )

    # Process documents in batches: chunk -> embed -> add to index
    # Save incrementally to disk to avoid memory bloat
    print("Processing documents in batches...")
    encoder = SentenceTransformer(embedding_model)
    embedding_dim = encoder.get_sentence_embedding_dimension()
    
    # Estimate final index size to avoid reallocations
    # Rough estimate: 150k docs * ~7 children per doc = ~1M children
    # Pre-allocate with some headroom to avoid reallocations
    estimated_children = corpus_config.get("max_docs", 150000) * 7
    index = faiss.IndexFlatIP(embedding_dim)
    # FAISS doesn't support pre-allocation, but we can reserve by adding empty vectors
    # Actually, better to just let it grow - FAISS handles reallocation efficiently
    # The reallocation overhead is small compared to embedding time

    manager = get_artifact_manager()
    # Save to indexes directory (shared index location)
    temp_dir = manager.get_embedding_index_path(retriever_name)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporary files for incremental saving
    import tempfile
    temp_parent_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=str(temp_dir))
    temp_child_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=str(temp_dir))
    temp_mapping_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=str(temp_dir))
    temp_parent_file.close()
    temp_child_file.close()
    temp_mapping_file.close()
    
    import pickle
    parent_file = open(temp_parent_file.name, 'ab')
    child_file = open(temp_child_file.name, 'ab')
    mapping_file = open(temp_mapping_file.name, 'ab')
    
    total_docs = 0
    doc_batch = []
    batch_num = 0
    parent_count = 0
    child_count = 0

    try:
        for doc in corpus.load():
            # Create new document with sequential ID
            from ragicamp.retrievers.base import Document
            doc_with_id = Document(
                id=f"wiki_{total_docs}",
                text=doc.text,
                metadata=doc.metadata,
            )
            doc_batch.append(doc_with_id)
            total_docs += 1

            if len(doc_batch) >= doc_batch_size:
                # Chunk this batch
                parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                    iter(doc_batch)
                )

                # Save mapping incrementally (small, do first)
                pickle.dump(batch_child_to_parent, mapping_file)
                mapping_file.flush()
                del batch_child_to_parent
                
                # Embed child chunks in batch (GPU work) - do this first to free GPU memory
                if child_chunks:
                    child_texts = [c.text for c in child_chunks]
                    embeddings = encoder.encode(child_texts, show_progress_bar=False, batch_size=64)
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    index.add(embeddings.astype("float32"))
                    
                    # Delete large objects immediately
                    del child_texts, embeddings
                    gc.collect()

                    # Save child chunks to disk
                    pickle.dump(child_chunks, child_file)
                    child_file.flush()
                    child_count += len(child_chunks)
                    del child_chunks  # Delete immediately after pickle
                    gc.collect()

                # Save parent chunks to disk (smaller, do last)
                pickle.dump(parent_chunks, parent_file)
                parent_file.flush()
                parent_count += len(parent_chunks)
                del parent_chunks
                gc.collect()

                print(f"  Processed {total_docs} docs → {parent_count} parents, {child_count} children (index: {index.ntotal})")

                # Clear batch
                doc_batch = []
                batch_num += 1

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

    # Copy temp files to final location first (before loading)
    print("Copying files to final location...")
    manager = get_artifact_manager()
    # Save index to indexes directory
    index_path = manager.get_embedding_index_path(retriever_name)
    index_path.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy2(temp_parent_file.name, index_path / "parent_docs.pkl")
    shutil.copy2(temp_child_file.name, index_path / "child_docs.pkl")
    
    # Load documents efficiently (streaming, but we need them for retriever)
    print("Loading documents for retriever...")
    all_parent_docs = []
    all_child_docs = []
    child_to_parent = {}
    parent_id_to_idx = {}
    
    # Load parent docs and build index mapping
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
    
    # Load child docs
    with open(temp_child_file.name, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
                all_child_docs.extend(batch)
            except EOFError:
                break
    
    # Reconstruct child_to_parent from saved batches
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
    
    # Save index config - compatible with HierarchicalIndex.load()
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
        "hierarchical_index": retriever_name,  # References index by name
        "embedding_model": embedding_model,
        "parent_chunk_size": parent_chunk_size,
        "child_chunk_size": child_chunk_size,
        "num_parents": parent_count,
        "num_children": child_count,
    }
    manager.save_json(retriever_config_data, retriever_path / "config.json")
    
    # Clean up temp files
    import os
    os.unlink(temp_parent_file.name)
    os.unlink(temp_child_file.name)
    os.unlink(temp_mapping_file.name)

    # Clean up
    del all_parent_docs, all_child_docs, child_to_parent, parent_id_to_idx
    del index, encoder, chunker
    gc.collect()

    print(f"✓ Saved hierarchical index: {retriever_name}")
    return retriever_name


# ============================================================================
# Prompt configuration
# ============================================================================


def get_prompt_builder(prompt_type: str, dataset: str):
    """Get a PromptBuilder configured for the given prompt type and dataset.

    This is the single entry point for prompt configuration. Uses the centralized
    PromptBuilder from utils/prompts.py.

    Args:
        prompt_type: One of "default", "concise", "fewshot", "fewshot_3", "fewshot_1"
        dataset: Dataset name (for loading appropriate fewshot examples)

    Returns:
        Configured PromptBuilder instance
    """
    from ragicamp.utils.prompts import PromptBuilder

    return PromptBuilder.from_config(prompt_type, dataset=dataset)


# ============================================================================
# Component creation
# ============================================================================


def create_model(spec: str, quant: str = "4bit"):
    """Create model from spec using ComponentFactory."""
    validate_model_spec(spec)
    config = ComponentFactory.parse_model_spec(spec, quantization=quant)
    return ComponentFactory.create_model(config)


def create_dataset(name: str, limit: Optional[int] = None):
    """Create dataset using ComponentFactory."""
    config = ComponentFactory.parse_dataset_spec(name, limit=limit)
    return ComponentFactory.create_dataset(config)


def create_judge_model(llm_judge_config: Optional[Dict[str, Any]]):
    """Create LLM judge model from config."""
    if not llm_judge_config:
        return None

    model_spec = llm_judge_config.get("model", "openai:gpt-4o-mini")

    if model_spec.startswith("openai:"):
        model_name = model_spec.split(":", 1)[1]
        return OpenAIModel(model_name=model_name)
    return None


def load_retriever(retriever_name: str):
    """Load a retriever by name, automatically detecting the type.

    Args:
        retriever_name: Name of the retriever to load

    Returns:
        Loaded retriever instance (DenseRetriever, HybridRetriever, or HierarchicalRetriever)
    """
    from ragicamp.retrievers import DenseRetriever, HybridRetriever, HierarchicalRetriever

    manager = get_artifact_manager()
    retriever_path = manager.get_retriever_path(retriever_name)
    config_path = retriever_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Retriever config not found: {config_path}")

    config = manager.load_json(config_path)
    retriever_type = config.get("type", "dense")

    if retriever_type == "hierarchical":
        return HierarchicalRetriever.load(retriever_name)
    elif retriever_type == "hybrid":
        return HybridRetriever.load(retriever_name)
    else:
        return DenseRetriever.load(retriever_name)


# ============================================================================
# Study execution
# ============================================================================


@dataclass
class ExpSpec:
    """Experiment specification."""

    name: str
    exp_type: str
    model: str
    dataset: str
    prompt: str
    quant: str = "4bit"
    retriever: Optional[str] = None
    top_k: int = 5
    batch_size: int = 8
    min_batch_size: int = 1  # Floor for auto batch size reduction on CUDA errors


def build_specs(config: Dict[str, Any]) -> List[ExpSpec]:
    """Build experiment specs from config."""
    specs = []
    datasets = config.get("datasets", ["nq"])
    batch = config.get("batch_size", 8)
    min_batch = config.get("min_batch_size", 1)

    # Direct experiments
    direct = config.get("direct", {})
    if direct.get("enabled"):
        for model in direct.get("models", []):
            for prompt in direct.get("prompts", ["default"]):
                for quant in direct.get("quantization", ["4bit"]):
                    if model.startswith("openai:") and quant != "4bit":
                        continue
                    for ds in datasets:
                        name = _name("direct", model, prompt, ds, quant)
                        specs.append(
                            ExpSpec(
                                name,
                                "direct",
                                model,
                                ds,
                                prompt,
                                quant,
                                batch_size=batch,
                                min_batch_size=min_batch,
                            )
                        )

    # RAG experiments
    rag = config.get("rag", {})
    if rag.get("enabled"):
        for model in rag.get("models", []):
            for ret_config in rag.get("retrievers", []):
                # Handle both dict configs and string names for backward compatibility
                ret_name = ret_config["name"] if isinstance(ret_config, dict) else ret_config
                for k in rag.get("top_k_values", [5]):
                    for prompt in rag.get("prompts", ["default"]):
                        for quant in rag.get("quantization", ["4bit"]):
                            if model.startswith("openai:") and quant != "4bit":
                                continue
                            for ds in datasets:
                                name = _name("rag", model, prompt, ds, quant, ret_name, k)
                                specs.append(
                                    ExpSpec(
                                        name,
                                        "rag",
                                        model,
                                        ds,
                                        prompt,
                                        quant,
                                        ret_name,
                                        k,
                                        batch_size=batch,
                                        min_batch_size=min_batch,
                                    )
                                )

    return specs


def _name(t, m, p, d, q, r=None, k=None):
    """Generate experiment name."""
    m = m.replace(":", "_").replace("/", "_").replace("-", "")
    s = f"_{q}" if q != "4bit" else ""
    return f"{t}_{m}_{p}_{d}{s}" if t == "direct" else f"{t}_{m}_{r}_k{k}_{p}_{d}{s}"


def run_spec_subprocess(
    spec: ExpSpec,
    limit: Optional[int],
    metrics: List[str],
    out: Path,
    llm_judge_config: Optional[Dict[str, Any]] = None,
    timeout: int = 7200,  # 2 hour default timeout
) -> str:
    """Run experiment in subprocess (isolated from CUDA crashes).
    
    If the subprocess crashes (e.g., CUDA error), retries with halved batch size
    until min_batch_size is reached.
    
    Returns:
        Status string: 'complete', 'resumed', 'ran', 'failed', 'crashed', 'timeout'
    """
    import subprocess
    import sys
    
    from ragicamp.experiment_state import ExperimentPhase, check_health
    
    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)
    
    # Check health before running
    health = check_health(exp_out, metrics)
    
    if health.is_complete:
        print(f"✓ {spec.name} (complete)")
        return "complete"
    
    if health.phase == ExperimentPhase.FAILED:
        print(f"✗ {spec.name} (failed previously: {health.error[:50] if health.error else 'unknown'})")
        print(f"  Retrying in subprocess...")
    
    # Determine action
    if health.can_resume:
        action = f"↻ Resuming from {health.resume_phase.value}"
        if health.needs_generation:
            action += f" ({health.predictions_complete}/{health.total_questions} predictions)"
    else:
        action = "▶ Starting"
    
    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{action}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "experiments" / "run_single_experiment.py"
    
    # Dynamic batch size reduction on crash
    current_batch_size = spec.batch_size
    min_batch_size = spec.min_batch_size
    attempt = 0
    
    while current_batch_size >= min_batch_size:
        attempt += 1
        
        # Build spec JSON for subprocess with current batch size
        spec_dict = {
            "name": spec.name,
            "exp_type": spec.exp_type,
            "model": spec.model,
            "dataset": spec.dataset,
            "prompt": spec.prompt,
            "quant": spec.quant,
            "retriever": spec.retriever,
            "top_k": spec.top_k,
            "batch_size": current_batch_size,
            "min_batch_size": min_batch_size,
        }
        
        cmd = [
            sys.executable,
            str(script_path),
            "--spec-json", json.dumps(spec_dict),
            "--output-dir", str(out),
            "--metrics", ",".join(metrics),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
        if llm_judge_config:
            cmd.extend(["--llm-judge-config", json.dumps(llm_judge_config)])
        
        if attempt > 1:
            print(f"🔄 Retrying with batch_size={current_batch_size} (attempt {attempt})")
        
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=False,  # Let output stream to console
            )
            
            if result.returncode == 0:
                return "resumed" if health.can_resume else "ran"
            else:
                # Subprocess crashed - check if we should retry with smaller batch
                if current_batch_size > min_batch_size:
                    new_batch_size = max(current_batch_size // 2, min_batch_size)
                    print(f"💥 Crashed (exit code {result.returncode}), reducing batch: {current_batch_size} → {new_batch_size}")
                    current_batch_size = new_batch_size
                    # Clear GPU memory before retry
                    try:
                        ResourceManager.clear_gpu_memory()
                    except:
                        pass
                    continue
                else:
                    # Already at min batch size, give up
                    error_log = exp_out / "error.log"
                    if not error_log.exists():
                        with open(error_log, "w") as f:
                            f.write(f"Subprocess crashed with exit code: {result.returncode}\n")
                            f.write(f"Tried batch sizes: {spec.batch_size} → {min_batch_size}\n")
                            f.write(f"This is typically a CUDA/bitsandbytes fatal error.\n")
                            f.write(f"Experiment: {spec.name}\n")
                            f.write(f"Model: {spec.model}\n")
                            f.write(f"Quantization: {spec.quant}\n")
                    
                    print(f"❌ Failed after {attempt} attempts (min batch={min_batch_size}). See: {error_log}")
                    
                    # Mark as failed in state
                    try:
                        from ragicamp.experiment_state import ExperimentState
                        state_path = exp_out / "state.json"
                        if state_path.exists():
                            state = ExperimentState.load(state_path)
                        else:
                            state = ExperimentState.new(metrics=metrics)
                        state.set_error(f"Crashed after {attempt} attempts (batch sizes {spec.batch_size}→{min_batch_size})")
                        state.save(state_path)
                    except:
                        pass
                    
                    return "crashed"
                
        except subprocess.TimeoutExpired:
            print(f"⏱️  Experiment TIMEOUT after {timeout}s")
            
            # Mark as timed out
            with open(exp_out / "error.log", "w") as f:
                f.write(f"Experiment timed out after {timeout} seconds\n")
                f.write(f"Experiment: {spec.name}\n")
            
            try:
                from ragicamp.experiment_state import ExperimentState
                state_path = exp_out / "state.json"
                if state_path.exists():
                    state = ExperimentState.load(state_path)
                else:
                    state = ExperimentState.new(metrics=metrics)
                state.set_error(f"Timeout after {timeout}s")
                state.save(state_path)
            except:
                pass
            
            return "timeout"
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            raise
    
    # Should never reach here, but just in case
    return "failed"


def run_spec(
    spec: ExpSpec,
    limit: Optional[int],
    metrics: List[str],
    out: Path,
    judge_model=None,
    llm_judge_config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    use_subprocess: bool = True,
) -> str:
    """Run single experiment with health-aware execution.

    Returns:
        Status string: 'complete', 'resumed', 'ran', 'failed', 'skipped', 'crashed', 'timeout'
    """
    # Use subprocess isolation for robustness against CUDA crashes
    if use_subprocess:
        return run_spec_subprocess(spec, limit, metrics, out, llm_judge_config=llm_judge_config)
    
    # Original in-process execution (kept for backwards compatibility)
    import time

    from ragicamp.experiment_state import ExperimentPhase, check_health

    exp_out = out / spec.name
    exp_out.mkdir(parents=True, exist_ok=True)

    # Check health before running
    health = check_health(exp_out, metrics)

    if health.is_complete and not force:
        print(f"✓ {spec.name} (complete)")
        return "complete"

    if health.phase == ExperimentPhase.FAILED and not force:
        print(f"✗ {spec.name} (failed: {health.error})")
        print(f"  Use --force to retry")
        return "skipped"

    # Determine action
    if health.can_resume:
        action = f"↻ Resuming from {health.resume_phase.value}"
        if health.needs_generation:
            action += f" ({health.predictions_complete}/{health.total_questions} predictions)"
        if health.needs_metrics:
            action += f" (missing: {', '.join(health.metrics_missing)})"
    else:
        action = "▶ Starting"

    print(f"\n{'='*60}")
    print(f"{spec.exp_type.upper()}: {spec.name}")
    print(f"{action}")
    print(f"{'='*60}")

    start = time.time()

    try:
        ResourceManager.clear_gpu_memory()

        dataset = create_dataset(spec.dataset, limit)
        print(f"Dataset: {len(dataset)} examples")

        model = create_model(spec.model, spec.quant)

        # Get prompt builder for this experiment
        prompt_builder = get_prompt_builder(spec.prompt, spec.dataset)

        if spec.exp_type == "direct":
            agent = DirectLLMAgent(name=spec.name, model=model, prompt_builder=prompt_builder)
        else:
            # Load retriever based on type from config
            retriever = load_retriever(spec.retriever)
            agent = FixedRAGAgent(
                spec.name, model, retriever, spec.top_k, prompt_builder=prompt_builder
            )

        metric_objs = ComponentFactory.create_metrics(metrics, judge_model=judge_model)

        exp = Experiment(spec.name, agent, dataset, metric_objs, out, _model=model)
        result = exp.run(
            batch_size=spec.batch_size,
            min_batch_size=spec.min_batch_size,
            checkpoint_every=50,
            resume=True,
        )

        # Save metadata
        meta = {
            "name": spec.name,
            "type": spec.exp_type,
            "model": spec.model,
            "prompt": spec.prompt,
            "dataset": spec.dataset,
            "quantization": spec.quant,
            "retriever": spec.retriever,
            "top_k": spec.top_k,
            "metrics": result.metrics,
            "duration": time.time() - start,
            "timestamp": datetime.now().isoformat(),
        }
        with open(exp_out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return "resumed" if health.can_resume else "ran"

    except KeyboardInterrupt:
        # User interrupted - save state and exit gracefully
        print(f"\n⚠️  Interrupted by user")
        # Try to save state as interrupted (not failed)
        try:
            from ragicamp.experiment_state import ExperimentState
            state_path = exp_out / "state.json"
            if state_path.exists():
                state = ExperimentState.load(state_path)
                state.error = "Interrupted by user (Ctrl+C)"
                state.save(state_path)
        except:
            pass
        raise  # Re-raise to stop the study
    
    except Exception as e:
        # Check if it's the metrics incomplete error
        if type(e).__name__ == "_MetricsIncompleteError":
            print(f"⚠ Incomplete: missing metrics {getattr(e, 'missing_metrics', [])}")
            return "incomplete"
        
        # Log the error
        error_msg = str(e)
        print(f"❌ Failed: {error_msg[:100]}")
        
        # Save detailed error to state
        try:
            from ragicamp.experiment_state import ExperimentState
            state_path = exp_out / "state.json"
            
            if state_path.exists():
                state = ExperimentState.load(state_path)
            else:
                state = ExperimentState.new(metrics=metrics)
            
            state.set_error(error_msg)
            state.save(state_path)
            
            # Also save error details to a separate file for debugging
            error_log_path = exp_out / "error.log"
            with open(error_log_path, "w") as f:
                import traceback
                f.write(f"Error: {error_msg}\n\n")
                f.write(f"Experiment: {spec.name}\n")
                f.write(f"Model: {spec.model}\n")
                f.write(f"Quantization: {spec.quant}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            
            print(f"  Error details saved to: {error_log_path}")
        except Exception as save_error:
            print(f"  Warning: Could not save error state: {save_error}")
        
        # Don't print full traceback to console (it's in error.log)
        # But do log it for debugging
        import traceback
        logger = __import__('logging').getLogger(__name__)
        logger.debug("Full traceback:", exc_info=True)
        
        return "failed"
    
    finally:
        # Always clean up GPU memory after experiment (success or failure)
        try:
            ResourceManager.clear_gpu_memory()
        except:
            pass


def run_study(
    config: Dict[str, Any],
    dry_run: bool = False,
    skip_existing: bool = False,
    validate_only: bool = False,
):
    """Run complete study."""
    # Validate config first
    try:
        warnings = validate_config(config)
        for w in warnings:
            print(f"⚠️  {w}")
    except ConfigError as e:
        print(f"❌ Config error: {e}")
        return

    if validate_only:
        print("✓ Config validation passed")
        return

    print("\n" + "=" * 70)
    print(f"Study: {config['name']}")
    print(f"  {config.get('description', '')}")
    print("=" * 70)

    # Build missing indexes if configured
    rag_config = config.get("rag", {})
    options = config.get("options", {})
    build_if_missing = options.get("build_index_if_missing", False)

    if rag_config.get("enabled") and rag_config.get("retrievers"):
        retriever_configs = rag_config["retrievers"]
        corpus_config = rag_config.get("corpus", {})

        try:
            ensure_indexes_exist(
                retriever_configs=retriever_configs,
                corpus_config=corpus_config,
                build_if_missing=build_if_missing,
            )
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

    specs = build_specs(config)
    limit = config.get("num_questions")
    metrics = config.get("metrics", ["f1", "exact_match"])
    llm_judge_config = config.get("llm_judge")
    out = Path(config.get("output_dir", "outputs"))
    out.mkdir(parents=True, exist_ok=True)

    # Create judge model once if needed
    judge_model = None
    if llm_judge_config and "llm_judge" in metrics:
        judge_model = create_judge_model(llm_judge_config)
        print(f"LLM Judge: {llm_judge_config.get('model', 'openai:gpt-4o-mini')}")

    print(f"\nExperiments: {len(specs)}")
    print(f"Questions: {limit or 'all'}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Output: {out}")

    if dry_run:
        print("\n[DRY RUN] - Checking experiment status:")
        from ragicamp.experiment_state import check_health

        for s in specs:
            health = check_health(out / s.name, metrics)
            print(f"  {health.summary()} - {s.name}")
        return

    # Track results by status
    status_counts = {
        "complete": 0,
        "resumed": 0,
        "ran": 0,
        "failed": 0,
        "skipped": 0,
        "incomplete": 0,
        "crashed": 0,  # CUDA/subprocess crashes
        "timeout": 0,  # Experiments that timed out
    }

    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] ", end="")

        status = run_spec(spec, limit, metrics, out, judge_model, llm_judge_config=llm_judge_config, force=not skip_existing)
        status_counts[status] += 1

    # Comparison
    compare(out)

    print("\n" + "=" * 70)
    print(
        f"Done! Ran: {status_counts['ran']}, Resumed: {status_counts['resumed']}, "
        f"Complete: {status_counts['complete']}, Incomplete: {status_counts['incomplete']}, "
        f"Failed: {status_counts['failed']}, Crashed: {status_counts['crashed']}, "
        f"Timeout: {status_counts['timeout']}"
    )
    if status_counts['crashed'] > 0:
        print(f"⚠️  {status_counts['crashed']} experiments crashed (CUDA errors) - see error.log files")
    print("=" * 70)


def compare(out: Path):
    """Print comparison table."""
    results = []
    for d in out.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            with open(d / "metadata.json") as f:
                results.append(json.load(f))

    if not results:
        return

    results.sort(key=lambda x: x.get("metrics", {}).get("f1", 0), reverse=True)

    print(f"\n{'='*80}")
    print("Results (by F1)")
    print("=" * 80)
    print(f"{'Experiment':<50} {'F1':>10} {'EM':>10}")
    print("-" * 80)

    for r in results[:20]:
        n = r["name"][:48] + ".." if len(r["name"]) > 48 else r["name"]
        f1 = r.get("metrics", {}).get("f1", 0) * 100
        em = r.get("metrics", {}).get("exact_match", 0) * 100
        print(f"{n:<50} {f1:>9.1f}% {em:>9.1f}%")

    # Save study summary (for quick access to aggregated results)
    with open(out / "study_summary.json", "w") as f:
        json.dump(
            {
                "experiments": results,
                "count": len(results),
            },
            f,
            indent=2,
        )
