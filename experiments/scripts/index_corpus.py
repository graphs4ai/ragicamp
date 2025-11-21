#!/usr/bin/env python3
"""
Index a document corpus for RAG retrieval.

Uses the new DocumentCorpus abstraction for clean, data-leak-free indexing.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.corpus import WikipediaCorpus, CorpusConfig
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.base import Document


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

    # Index documents
    print(f"\nIndexing documents from corpus...")
    print("⚠️  This will take time (computing embeddings)")

    documents = []
    for doc in corpus.load(max_docs=args.max_docs):
        documents.append(doc)

        if len(documents) % 1000 == 0:
            print(f"  Collected {len(documents)} documents...")

    print(f"\n✓ Collected {len(documents)} documents")

    if len(documents) == 0:
        print("✗ No documents collected. Exiting.")
        sys.exit(1)

    # Index all documents
    print("\nIndexing documents with retriever...")
    retriever.index_documents(documents)

    # Save the index
    print(f"\nSaving retriever artifact...")
    artifact_path = retriever.save_index(args.artifact_name)

    # Save corpus metadata
    corpus_info = corpus.get_info()
    print(f"\n✓ Corpus info:")
    for key, value in corpus_info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"\nArtifact: {args.artifact_name}")
    print(f"Documents indexed: {len(documents)}")
    print(f"Saved to: {artifact_path}")
    print("\n⚠️  IMPORTANT: Documents contain NO answer information!")
    print("This is correct for RAG - the model must extract answers from context.")
    print("=" * 70)


if __name__ == "__main__":
    main()
