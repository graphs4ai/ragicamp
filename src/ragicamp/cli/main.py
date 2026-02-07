#!/usr/bin/env python3
"""RAGiCamp CLI - Unified command-line interface.

Commands:
    run       Run a study from config
    index     Build retrieval indexes
    compare   Compare experiment results
    evaluate  Compute metrics on predictions
    backup    Backup artifacts to Backblaze B2
    download  Download artifacts from Backblaze B2
    prune     Remove orphaned remote files not present locally
"""

# ============================================================================
# CRITICAL: Configure environment BEFORE any library imports!
# ============================================================================
import os

# TensorFlow: Prevent grabbing all GPU memory on import
if "TF_FORCE_GPU_ALLOW_GROWTH" not in os.environ:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# vLLM: Use 'spawn' for multiprocessing to avoid CUDA fork issues
# See: https://github.com/vllm-project/vllm/issues/6152
if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import sys
from pathlib import Path
from typing import Optional

from ragicamp.cli.commands import (
    cmd_backup,
    cmd_cache,
    cmd_compare,
    cmd_download,
    cmd_evaluate,
    cmd_health,
    cmd_index,
    cmd_metrics,
    cmd_migrate_indexes,
    cmd_prune,
    cmd_resume,
    cmd_run,
)


def _add_run_parser(subparsers) -> None:
    """Add 'run' subcommand."""
    p = subparsers.add_parser("run", help="Run a study from config")
    p.add_argument("config", type=Path, help="Study config YAML")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--skip-existing", action="store_true", help="Skip completed")
    p.add_argument("--validate", action="store_true", help="Validate config only")
    p.add_argument("--sample", "-s", type=int, metavar="N", help="Sample N experiments (enables search)")
    p.add_argument("--sample-mode", choices=["random", "tpe"], default="random", help="Sampling mode")
    p.add_argument("--sample-seed", type=int, default=None, help="Random seed for sampling")
    p.add_argument("--optimize-metric", type=str, default="f1", help="Metric to optimize (default: f1)")
    p.set_defaults(func=cmd_run)


def _add_index_parser(subparsers) -> None:
    """Add 'index' subcommand."""
    p = subparsers.add_parser("index", help="Build retrieval index")
    p.add_argument("--corpus", default="simple", help="Corpus: simple, en, or full version string")
    p.add_argument("--embedding", default="minilm", help="Embedding: minilm, e5, mpnet, or full model name")
    p.add_argument("--chunk-size", type=int, default=512, help="Chunk size in chars")
    p.add_argument("--max-docs", type=int, default=None, help="Max documents to index")
    p.set_defaults(func=cmd_index)


def _add_compare_parser(subparsers) -> None:
    """Add 'compare' subcommand."""
    p = subparsers.add_parser("compare", help="Compare results")
    p.add_argument("output_dir", type=Path, help="Output directory")
    p.add_argument("--top", type=int, default=10, help="Show top N results")
    p.add_argument("--metric", "-m", default="f1", help="Metric to compare")
    p.add_argument(
        "--group-by", "-g", default="model",
        choices=["model", "dataset", "prompt", "retriever", "quantization", "type"],
        help="Dimension to group by",
    )
    p.add_argument("--pivot", nargs=2, metavar=("ROWS", "COLS"), help="Create pivot table")
    p.set_defaults(func=cmd_compare)


def _add_evaluate_parser(subparsers) -> None:
    """Add 'evaluate' subcommand."""
    p = subparsers.add_parser("evaluate", help="Compute metrics")
    p.add_argument("predictions", type=Path, help="Predictions JSON")
    p.add_argument("--metrics", nargs="+", default=["f1", "exact_match"], help="Metrics to compute")
    p.add_argument("--output", type=Path, help="Output file")
    p.add_argument("--judge-model", default="gpt-4o-mini", help="Model for LLM judge")
    p.add_argument("--judgment-type", choices=["binary", "ternary"], default="binary")
    p.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent LLM judge calls")
    p.set_defaults(func=cmd_evaluate)


def _add_health_parser(subparsers) -> None:
    """Add 'health' subcommand."""
    p = subparsers.add_parser("health", help="Check experiment health")
    p.add_argument("output_dir", type=Path, help="Output directory")
    p.add_argument("--metrics", default=None, help="Comma-separated metrics to check")
    p.set_defaults(func=cmd_health)


def _add_resume_parser(subparsers) -> None:
    """Add 'resume' subcommand."""
    p = subparsers.add_parser("resume", help="Resume incomplete experiments")
    p.add_argument("output_dir", type=Path, help="Output directory")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.set_defaults(func=cmd_resume)


def _add_metrics_parser(subparsers) -> None:
    """Add 'metrics' subcommand."""
    p = subparsers.add_parser("metrics", help="Recompute metrics for an experiment")
    p.add_argument("exp_dir", type=Path, help="Experiment directory")
    p.add_argument("--metrics", "-m", required=True, help="Comma-separated metrics")
    p.add_argument("--judge-model", default="gpt-4o-mini", help="Model for LLM judge")
    p.set_defaults(func=cmd_metrics)


def _add_backup_parser(subparsers) -> None:
    """Add 'backup' subcommand."""
    p = subparsers.add_parser("backup", help="Backup artifacts and outputs to Backblaze B2")
    p.add_argument("path", type=Path, nargs="?", default=None, help="Directory to backup")
    p.add_argument("--bucket", "-b", default="masters-bucket", help="B2 bucket name")
    p.add_argument("--prefix", "-p", default=None, help="S3 key prefix")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--continue-on-error", action="store_true", help="Continue on upload errors")
    p.add_argument("--workers", "-w", type=int, default=12, help="Parallel upload threads")
    p.add_argument("--sync", "-s", action="store_true", help="Only upload new/modified files")
    p.add_argument("--latest", "-l", action="store_true", help="Use latest backup as target prefix")
    p.set_defaults(func=cmd_backup)


def _add_download_parser(subparsers) -> None:
    """Add 'download' subcommand."""
    p = subparsers.add_parser("download", help="Download artifacts from Backblaze B2")
    p.add_argument("--list", "-l", action="store_true", help="List available backups")
    p.add_argument("--backup", "-b", default=None, help="Backup name to download")
    p.add_argument("--bucket", default="masters-bucket", help="B2 bucket name")
    p.add_argument("--artifacts-only", action="store_true", help="Only download artifacts/")
    p.add_argument("--outputs-only", action="store_true", help="Only download outputs/")
    p.add_argument("--indexes-only", action="store_true", help="Only download artifacts/indexes/")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--continue-on-error", action="store_true", help="Continue on download errors")
    p.add_argument("--workers", "-w", type=int, default=12, help="Parallel download threads")
    p.add_argument("--skip-existing", action="store_true", help="Skip files already on disk")
    p.add_argument("--no-migrate", action="store_true", help="Skip index migration after download")
    p.set_defaults(func=cmd_download)


def _add_cache_parser(subparsers) -> None:
    """Add 'cache' subcommand."""
    p = subparsers.add_parser("cache", help="Manage the embedding cache")
    sub = p.add_subparsers(dest="cache_action", help="Cache actions")
    sub.required = True
    sub.add_parser("stats", help="Show cache statistics")
    clear = sub.add_parser("clear", help="Clear cached embeddings")
    clear.add_argument("--model", default=None, help="Only clear for this model")
    p.set_defaults(func=cmd_cache)


def _add_prune_parser(subparsers) -> None:
    """Add 'prune' subcommand."""
    p = subparsers.add_parser(
        "prune",
        help="Remove orphaned remote files that no longer exist locally",
    )
    p.add_argument("path", type=Path, nargs="?", default=None, help="Local directory to compare against")
    p.add_argument("--bucket", default="masters-bucket", help="B2 bucket name")
    p.add_argument("--prefix", "-p", default=None, help="S3 key prefix to prune (default: latest backup)")
    p.add_argument("--dry-run", action="store_true", help="Preview only, do not delete")
    p.add_argument("--workers", "-w", type=int, default=12, help="Parallel delete threads")
    p.set_defaults(func=cmd_prune)


def _add_migrate_parser(subparsers) -> None:
    """Add 'migrate-indexes' subcommand."""
    p = subparsers.add_parser("migrate-indexes", help="Migrate old index format to new format")
    p.add_argument("--index-name", "-n", default=None, help="Specific index to migrate")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--force", "-f", action="store_true", help="Re-migrate even if already migrated")
    p.set_defaults(func=cmd_migrate_indexes)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ragicamp",
        description="RAGiCamp - RAG Experimentation Framework",
    )
    from ragicamp import __version__

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_run_parser(subparsers)
    _add_index_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_health_parser(subparsers)
    _add_resume_parser(subparsers)
    _add_metrics_parser(subparsers)
    _add_backup_parser(subparsers)
    _add_download_parser(subparsers)
    _add_prune_parser(subparsers)
    _add_cache_parser(subparsers)
    _add_migrate_parser(subparsers)

    return parser


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
