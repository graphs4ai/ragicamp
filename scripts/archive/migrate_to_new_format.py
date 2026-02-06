#!/usr/bin/env python3
"""Migrate indexes and retrievers to new separated format.

The new architecture separates:
- Indexes (expensive, reusable): artifacts/indexes/
- Retrievers (cheap, configurable): artifacts/retrievers/ (config only)

This script migrates old-format data to the new format:
1. Moves hierarchical data from retrievers/ to indexes/
2. Creates retriever configs that reference indexes

Usage:
    python scripts/migrate_to_new_format.py [--dry-run]
"""

import json
import shutil
import sys
from pathlib import Path


def get_artifacts_dir() -> Path:
    """Get artifacts directory."""
    # Look for artifacts relative to script or cwd
    for base in [Path(__file__).parent.parent, Path.cwd()]:
        artifacts = base / "artifacts"
        if artifacts.exists():
            return artifacts
    raise FileNotFoundError("Could not find artifacts directory")


def migrate_hierarchical_retrievers(artifacts_dir: Path, dry_run: bool = False):
    """Migrate hierarchical retrievers to new index format."""
    retrievers_dir = artifacts_dir / "retrievers"
    indexes_dir = artifacts_dir / "indexes"

    if not retrievers_dir.exists():
        print("No retrievers directory found")
        return

    # Find hierarchical retrievers with data (not just config)
    hierarchical_files = [
        "child_docs.pkl",
        "child_index.faiss",
        "parent_docs.pkl",
        "child_to_parent.pkl",
    ]

    migrated = 0
    for retriever_path in retrievers_dir.iterdir():
        if not retriever_path.is_dir():
            continue

        config_path = retriever_path / "config.json"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        if config.get("type") != "hierarchical":
            continue

        # Check if has data files (old format)
        has_data = all((retriever_path / f).exists() for f in hierarchical_files)

        if not has_data:
            # Already migrated or just config
            print(f"  ✓ {retriever_path.name}: already migrated (config only)")
            continue

        # Check if already has hierarchical_index reference
        if config.get("hierarchical_index"):
            print(f"  ✓ {retriever_path.name}: already references index")
            continue

        index_name = retriever_path.name
        index_path = indexes_dir / index_name

        print(f"\n  📦 Migrating: {index_name}")
        print(f"     From: {retriever_path}")
        print(f"     To:   {index_path}")

        if dry_run:
            print("     [DRY RUN - no changes made]")
            continue

        # Create index directory
        index_path.mkdir(parents=True, exist_ok=True)

        # Move data files to index directory
        for fname in hierarchical_files:
            src = retriever_path / fname
            dst = index_path / fname
            if src.exists():
                print(f"     Moving: {fname}")
                shutil.move(str(src), str(dst))

        # Create index config
        index_config = {
            "name": index_name,
            "type": "hierarchical",
            "embedding_model": config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
            "parent_chunk_size": config.get("parent_chunk_size", 1024),
            "child_chunk_size": config.get("child_chunk_size", 256),
            "parent_overlap": config.get("parent_overlap", 100),
            "child_overlap": config.get("child_overlap", 50),
            "num_parents": config.get("num_parents", 0),
            "num_children": config.get("num_children", 0),
            "embedding_dim": config.get("embedding_dim"),
        }

        with open(index_path / "config.json", "w") as f:
            json.dump(index_config, f, indent=2)
        print(f"     Created: {index_path / 'config.json'}")

        # Update retriever config to reference index
        retriever_config = {
            "name": index_name,
            "type": "hierarchical",
            "hierarchical_index": index_name,
            "embedding_model": config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
            "parent_chunk_size": config.get("parent_chunk_size", 1024),
            "child_chunk_size": config.get("child_chunk_size", 256),
            "num_parents": config.get("num_parents", 0),
            "num_children": config.get("num_children", 0),
        }

        with open(retriever_path / "config.json", "w") as f:
            json.dump(retriever_config, f, indent=2)
        print(f"     Updated: {retriever_path / 'config.json'}")

        migrated += 1

    return migrated


def verify_dense_hybrid_retrievers(artifacts_dir: Path):
    """Verify dense/hybrid retrievers reference indexes correctly."""
    retrievers_dir = artifacts_dir / "retrievers"
    indexes_dir = artifacts_dir / "indexes"

    if not retrievers_dir.exists():
        return

    for retriever_path in retrievers_dir.iterdir():
        if not retriever_path.is_dir():
            continue

        config_path = retriever_path / "config.json"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        rtype = config.get("type", "dense")

        if rtype in ("dense", "hybrid"):
            index_name = config.get("embedding_index")
            if index_name:
                index_path = indexes_dir / index_name
                if index_path.exists():
                    print(f"  ✓ {retriever_path.name}: references {index_name}")
                else:
                    print(f"  ⚠ {retriever_path.name}: missing index {index_name}")
            else:
                print(f"  ⚠ {retriever_path.name}: no embedding_index reference")


def show_summary(artifacts_dir: Path):
    """Show current state of artifacts."""
    indexes_dir = artifacts_dir / "indexes"
    retrievers_dir = artifacts_dir / "retrievers"

    print("\n" + "=" * 60)
    print("CURRENT STATE")
    print("=" * 60)

    if indexes_dir.exists():
        print("\nIndexes (artifacts/indexes/):")
        for idx_path in sorted(indexes_dir.iterdir()):
            if idx_path.is_dir():
                config_path = idx_path / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    idx_type = config.get("type", "embedding")
                    if idx_type == "hierarchical":
                        print(
                            f"  📚 {idx_path.name} (hierarchical, {config.get('num_parents', '?')} parents)"
                        )
                    else:
                        print(f"  📚 {idx_path.name} ({config.get('num_documents', '?')} docs)")

    if retrievers_dir.exists():
        print("\nRetrievers (artifacts/retrievers/):")
        for ret_path in sorted(retrievers_dir.iterdir()):
            if ret_path.is_dir():
                config_path = ret_path / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)

                    rtype = config.get("type", "dense")
                    index_ref = config.get("embedding_index") or config.get("hierarchical_index")

                    # Check if has data files (old format)
                    has_data = any(
                        (ret_path / f).exists()
                        for f in ["index.faiss", "documents.pkl", "child_index.faiss"]
                    )

                    if has_data:
                        print(f"  ⚠ {ret_path.name} ({rtype}, OLD FORMAT with data)")
                    elif index_ref:
                        print(f"  ✓ {ret_path.name} ({rtype}) → {index_ref}")
                    else:
                        print(f"  ? {ret_path.name} ({rtype}, no index reference)")


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("RAG Index Migration Tool")
    print("=" * 60)
    print("\nMigrates to new architecture:")
    print("  - Indexes: artifacts/indexes/ (reusable, expensive to build)")
    print("  - Retrievers: artifacts/retrievers/ (config only, cheap)")

    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")

    try:
        artifacts_dir = get_artifacts_dir()
        print(f"\nArtifacts directory: {artifacts_dir}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1

    # Show current state
    show_summary(artifacts_dir)

    print("\n" + "=" * 60)
    print("MIGRATION")
    print("=" * 60)

    # Migrate hierarchical retrievers
    print("\nMigrating hierarchical retrievers:")
    migrated = migrate_hierarchical_retrievers(artifacts_dir, dry_run)

    print(f"\n✓ Migrated {migrated} hierarchical retriever(s)")

    # Verify dense/hybrid
    print("\nVerifying dense/hybrid retrievers:")
    verify_dense_hybrid_retrievers(artifacts_dir)

    if not dry_run and migrated > 0:
        print("\n" + "=" * 60)
        print("AFTER MIGRATION")
        print("=" * 60)
        show_summary(artifacts_dir)

    print("\n✓ Migration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
