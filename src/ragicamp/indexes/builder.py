"""Index building utilities.

This module provides the orchestration layer for building and managing indexes:
- ensure_indexes_exist: Main entry point for ensuring all required indexes exist
- get_embedding_index_name: Compute canonical names for shared indexes
- build_retriever_from_index: Create retriever configs from shared indexes

The actual index building logic is in the builders/ subpackage.
"""

import pickle
from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.indexes.builders import build_embedding_index, build_hierarchical_index
from ragicamp.utils.artifacts import get_artifact_manager
from ragicamp.utils.resource_manager import ResourceManager

logger = get_logger(__name__)


def get_embedding_index_name(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_version: str,
    chunking_strategy: str = "recursive",
) -> str:
    """Compute canonical name for a shared embedding index.

    Indexes with the same config get the same name, enabling reuse.

    Args:
        embedding_model: Embedding model name
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        corpus_version: Corpus version string
        chunking_strategy: Chunking strategy (recursive, fixed, sentence, paragraph)

    Returns:
        Canonical index name
    """
    # Normalize model name
    model_short = embedding_model.split("/")[-1].replace("-", "_").lower()
    # Extract corpus short name (e.g., "20231101.en" -> "en")
    corpus_short = corpus_version.split(".")[-1] if "." in corpus_version else corpus_version

    # Only include strategy in name if not default (for backwards compatibility)
    if chunking_strategy == "recursive":
        return f"{corpus_short}_{model_short}_c{chunk_size}_o{chunk_overlap}"
    else:
        # Use short strategy name: recursive->rec, paragraph->para, sentence->sent, fixed->fix
        strategy_short = {"paragraph": "para", "sentence": "sent", "fixed": "fix"}.get(
            chunking_strategy, chunking_strategy[:3]
        )
        return f"{corpus_short}_{model_short}_c{chunk_size}_o{chunk_overlap}_{strategy_short}"


def build_retriever_from_index(
    retriever_config: dict[str, Any],
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
        "index_type": index_config.get("index_type", "flat"),
    }

    if retriever_type == "hybrid":
        retriever_cfg["alpha"] = retriever_config.get("alpha", 0.5)
        # For hybrid, we need to build the BM25 index
        print("    Building BM25 index for hybrid retriever...")
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
    retriever_configs: list[dict[str, Any]],
    corpus_config: dict[str, Any],
    build_if_missing: bool = True,
) -> list[str]:
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
    index_to_retrievers: dict[str, list[dict[str, Any]]] = {}

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
                chunking_strategy=config.get("chunking_strategy", "recursive"),
            )

        if index_name not in index_to_retrievers:
            index_to_retrievers[index_name] = []
        index_to_retrievers[index_name].append(config)

    print("\n📊 Index analysis:")
    print(
        f"   {len(retriever_configs)} retrievers → {len(index_to_retrievers)} unique embedding indexes"
    )

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
                build_hierarchical_index(
                    first_retriever,
                    corpus_config,
                    doc_batch_size=corpus_config.get("doc_batch_size", 1000),
                    embedding_batch_size=corpus_config.get("embedding_batch_size", 64),
                )
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
                    chunking_strategy=first_retriever.get("chunking_strategy", "recursive"),
                    doc_batch_size=corpus_config.get("doc_batch_size", 5000),
                    embedding_batch_size=corpus_config.get("embedding_batch_size", 64),
                    index_type=first_retriever.get("index_type"),  # Pass index_type from config
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
