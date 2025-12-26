#!/usr/bin/env python3
"""
Index a document corpus for RAG retrieval.

Uses the new DocumentCorpus abstraction for clean, data-leak-free indexing.
Supports document chunking for better retrieval quality.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.corpus import ChunkConfig, CorpusConfig, DocumentChunker, WikipediaCorpus
from ragicamp.retrievers.base import Document
from ragicamp.retrievers.dense import DenseRetriever


def main():
    """Index a document corpus."""
    parser = argparse.ArgumentParser(description="Index document corpus for RAG retrieval")
    parser.add_argument(
        "--corpus-name",
        type=str,
        default="wikipedia_simple",
        help="Corpus identifier (default: wikipedia_simple)",
    )
    parser.add_argument(
        "--corpus-source",
        type=str,
        default="wikimedia/wikipedia",
        help="Corpus source (default: wikimedia/wikipedia)",
    )
    parser.add_argument(
        "--corpus-version",
        type=str,
        default="20231101.simple",
        help="Corpus version (default: 20231101.simple)",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Maximum documents to index (for testing)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model for dense retrieval",
    )
    parser.add_argument(
        "--artifact-name", type=str, required=True, help="Name for the saved retriever artifact"
    )

    # Chunking arguments
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default=None,
        choices=["fixed", "sentence", "paragraph", "recursive"],
        help="Chunking strategy (default: no chunking, index full documents)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in chars (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks in chars (default: 50)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Document Corpus Indexing")
    print("=" * 70)
    print(f"\nCorpus: {args.corpus_name}")
    print(f"Source: {args.corpus_source}")
    print(f"Version: {args.corpus_version}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Artifact name: {args.artifact_name}")
    if args.max_docs:
        print(f"Max documents: {args.max_docs} (testing)")
    if args.chunk_strategy:
        print(f"\n📄 Chunking enabled:")
        print(f"   Strategy: {args.chunk_strategy}")
        print(f"   Chunk size: {args.chunk_size} chars")
        print(f"   Overlap: {args.chunk_overlap} chars")
    else:
        print(f"\n📄 Chunking: disabled (indexing full documents)")
    print("\n" + "=" * 70 + "\n")

    # Create corpus configuration
    corpus_config = CorpusConfig(
        name=args.corpus_name,
        source=args.corpus_source,
        version=args.corpus_version,
        max_docs=args.max_docs,
    )

    # Initialize corpus
    print(f"Initializing corpus: {corpus_config}")
    corpus = WikipediaCorpus(corpus_config)

    # Create retriever
    print(f"\nCreating dense retriever...")
    retriever = DenseRetriever(
        name=f"{args.corpus_name}_retriever",
        embedding_model=args.embedding_model,
        index_type="flat",
    )

    # Load documents from corpus
    print(f"\nLoading documents from corpus...")
    raw_documents = []
    for doc in corpus.load(max_docs=args.max_docs):
        raw_documents.append(doc)

        if len(raw_documents) % 1000 == 0:
            print(f"  Loaded {len(raw_documents)} documents...")

    print(f"\n✓ Loaded {len(raw_documents)} raw documents")

    if len(raw_documents) == 0:
        print("✗ No documents collected. Exiting.")
        sys.exit(1)

    # Apply chunking if enabled
    if args.chunk_strategy:
        print(f"\n📄 Chunking documents with '{args.chunk_strategy}' strategy...")
        chunk_config = ChunkConfig(
            strategy=args.chunk_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        chunker = DocumentChunker(chunk_config)
        documents = list(chunker.chunk_documents(iter(raw_documents)))
        print(f"✓ Created {len(documents)} chunks from {len(raw_documents)} documents")
        print(f"  Avg chunks per doc: {len(documents) / len(raw_documents):.1f}")
    else:
        documents = raw_documents
        print(f"\n📄 Using {len(documents)} full documents (no chunking)")

    # Index all documents/chunks
    print("\n⚠️  Computing embeddings (this will take time)...")
    retriever.index_documents(documents)

    # Save the index
    print(f"\nSaving retriever artifact...")
    artifact_path = retriever.save_index(args.artifact_name)

    # Save comprehensive metadata for later analysis
    from datetime import datetime
    import json
    
    corpus_info = corpus.get_info()
    
    full_metadata = {
        "artifact_name": args.artifact_name,
        "created_at": datetime.now().isoformat(),
        
        # Corpus info
        "corpus": {
            "name": args.corpus_name,
            "source": args.corpus_source,
            "version": args.corpus_version,
            "raw_documents": len(raw_documents),
            **corpus_info,
        },
        
        # Embedding info
        "embedding": {
            "model": args.embedding_model,
            "dimensions": retriever.embedding_dim,
        },
        
        # Chunking info
        "chunking": {
            "enabled": args.chunk_strategy is not None,
            "strategy": args.chunk_strategy,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "total_chunks": len(documents),
            "avg_chunks_per_doc": len(documents) / len(raw_documents) if raw_documents else 0,
        },
        
        # Index info
        "index": {
            "type": "faiss_flat",
            "num_vectors": len(documents),
        },
    }
    
    metadata_path = Path(artifact_path) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)
    
    print(f"\n✓ Full metadata saved to: {metadata_path}")
    print(f"\n✓ Corpus info:")
    for key, value in corpus_info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"\nArtifact: {args.artifact_name}")
    if args.chunk_strategy:
        print(f"Raw documents: {len(raw_documents)}")
        print(f"Chunks indexed: {len(documents)}")
        print(f"Chunk strategy: {args.chunk_strategy}")
    else:
        print(f"Documents indexed: {len(documents)}")
    print(f"Saved to: {artifact_path}")
    print("\n⚠️  IMPORTANT: Documents contain NO answer information!")
    print("This is correct for RAG - the model must extract answers from context.")
    print("=" * 70)


if __name__ == "__main__":
    main()
