"""Hierarchical index builder.

Builds hierarchical indexes with parent-child chunk relationships.
Uses pipelined processing to overlap GPU embedding with CPU post-processing.
"""

import gc
import os
import pickle
import tempfile
from typing import Any, Iterator

# Disable tokenizers parallelism to avoid fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ragicamp.core.logging import get_logger
from ragicamp.utils.artifacts import get_artifact_manager

logger = get_logger(__name__)


def generate_doc_batches(
    corpus,
    doc_batch_size: int,
    Document,
) -> Iterator[tuple[list, int, bool]]:
    """Generate batches of documents from corpus.

    Args:
        corpus: Corpus to load documents from
        doc_batch_size: Number of documents per batch
        Document: Document class for creating docs with IDs

    Yields:
        Tuples of (doc_batch, total_docs_so_far, is_final)
    """
    doc_batch = []
    total_docs = 0

    for doc in corpus.load():
        doc_with_id = Document(
            id=f"wiki_{total_docs}",
            text=doc.text,
            metadata=doc.metadata,
        )
        doc_batch.append(doc_with_id)
        total_docs += 1

        if len(doc_batch) >= doc_batch_size:
            yield doc_batch, total_docs, False
            doc_batch = []

    if doc_batch:
        yield doc_batch, total_docs, True


def build_hierarchical_index(
    retriever_config: dict[str, Any],
    corpus_config: dict[str, Any],
    doc_batch_size: int = 1000,
    embedding_batch_size: int = 64,
    embedding_backend: str = "vllm",
    vllm_gpu_memory_fraction: float = 0.9,
) -> str:
    """Build a hierarchical index with batched processing.

    Hierarchical indexes have unique parent/child chunking, so they
    can't share indexes with other retrievers.

    Args:
        retriever_config: Retriever configuration with chunk sizes
        corpus_config: Corpus configuration
        doc_batch_size: Documents to process per batch (default 1000)
        embedding_batch_size: Texts to embed per GPU batch (default 64)
        embedding_backend: 'vllm' or 'sentence_transformers'
        vllm_gpu_memory_fraction: GPU memory fraction for vLLM (default 0.9)

    Returns:
        Retriever name
    """
    import faiss
    import numpy as np

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

    print(f"\n{'=' * 60}")
    print(f"Building hierarchical index: {retriever_name}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Embedding backend: {embedding_backend}")
    print(f"  Parent chunks: {parent_chunk_size}")
    print(f"  Child chunks: {child_chunk_size}")
    print(f"  Doc batch size: {doc_batch_size}, embedding batch size: {embedding_batch_size}")
    print(f"{'=' * 60}")

    # Build metadata dict for WikiRank filter and other options
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

    # Initialize chunker
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_overlap,
    )

    # Initialize encoder based on backend
    print(f"Loading embedding model ({embedding_backend})...")

    if embedding_backend == "vllm":
        from ragicamp.models.vllm_embedder import VLLMEmbedder

        encoder = VLLMEmbedder(
            model_name=embedding_model,
            gpu_memory_fraction=vllm_gpu_memory_fraction,
            enforce_eager=False,
        )
        embedding_dim = encoder.get_sentence_embedding_dimension()
        print(
            f"  vLLM embedder loaded (dim={embedding_dim}, gpu_mem={vllm_gpu_memory_fraction:.0%})"
        )
    elif embedding_backend == "vllm_server":
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

    print("Processing documents in batches (pipelined)...")

    index = faiss.IndexFlatIP(embedding_dim)

    # Save to indexes directory
    temp_dir = manager.get_embedding_index_path(retriever_name)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Temporary files for incremental saving
    temp_parent_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=str(temp_dir))
    temp_child_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=str(temp_dir))
    temp_mapping_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=str(temp_dir))
    temp_parent_file.close()
    temp_child_file.close()
    temp_mapping_file.close()

    parent_file = open(temp_parent_file.name, "ab")
    child_file = open(temp_child_file.name, "ab")
    mapping_file = open(temp_mapping_file.name, "ab")

    parent_count = 0
    child_count = 0
    total_docs = 0

    # Pipeline for overlapping GPU embedding with CPU post-processing
    from ragicamp.indexes.builders.pipeline import EmbeddingPipeline, EmbeddingResult

    # Store parent chunks separately (they don't get embedded)
    pending_parents: list = []

    def process_batch(result: EmbeddingResult) -> None:
        """Process embeddings: add to index, save chunks."""
        nonlocal child_count, parent_count

        embeddings = result.embeddings
        child_chunks = result.chunks

        # Add to index
        index.add(embeddings.astype("float32"))

        # Save child chunks
        pickle.dump(child_chunks, child_file)
        child_file.flush()
        child_count += len(child_chunks)

        # Save corresponding parent chunks
        if pending_parents:
            parents = pending_parents.pop(0)
            pickle.dump(parents, parent_file)
            parent_file.flush()
            parent_count += len(parents)

        print(f"    ✓ Indexed {len(embeddings)} vectors (total: {index.ntotal})")
        gc.collect()

    try:
        with EmbeddingPipeline(
            encoder=encoder,
            process_fn=process_batch,
            embedding_batch_size=embedding_batch_size,
            normalize=True,
        ) as pipeline:
            batch_num = 0

            for doc_batch, docs_so_far, is_final in generate_doc_batches(
                corpus, doc_batch_size, Document
            ):
                batch_num += 1
                batch_size = len(doc_batch)
                total_docs = docs_so_far
                suffix = " (final)" if is_final else ""
                print(f"\n  [Batch {batch_num}] Processing {batch_size} documents{suffix}...")

                print("    Chunking (hierarchical)...", end=" ", flush=True)
                parent_chunks, child_chunks, batch_child_to_parent = chunker.chunk_documents(
                    iter(doc_batch)
                )
                print(f"→ {len(parent_chunks)} parents, {len(child_chunks)} children")

                # Save mapping immediately
                pickle.dump(batch_child_to_parent, mapping_file)
                mapping_file.flush()

                if child_chunks:
                    child_texts = [c.text for c in child_chunks]
                    print(f"    Embedding {len(child_texts)} child chunks...")

                    # Queue parent chunks for saving after embedding completes
                    pending_parents.append(parent_chunks)

                    # Submit to pipeline - GPU work starts, previous batch processed
                    pipeline.submit(child_texts, child_chunks)
                else:
                    # No children, just save parents
                    pickle.dump(parent_chunks, parent_file)
                    parent_file.flush()
                    parent_count += len(parent_chunks)

                print(f"  ✓ Total: {total_docs} docs processed")
                del doc_batch, batch_child_to_parent
                gc.collect()

        # Save any remaining parents after pipeline finishes
        for parents in pending_parents:
            pickle.dump(parents, parent_file)
            parent_file.flush()
            parent_count += len(parents)

        print(
            f"✓ Processing complete: {total_docs} docs → {parent_count} parents, {child_count} children"
        )

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
    with open(temp_parent_file.name, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                all_parent_docs.extend(batch)
                for doc in batch:
                    parent_id_to_idx[doc.id] = idx
                    idx += 1
            except EOFError:
                break

    with open(temp_child_file.name, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                all_child_docs.extend(batch)
            except EOFError:
                break

    with open(temp_mapping_file.name, "rb") as f:
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
        "embedding_backend": embedding_backend,
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
