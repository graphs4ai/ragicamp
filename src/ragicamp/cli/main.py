#!/usr/bin/env python3
"""RAGiCamp CLI - Unified command-line interface.

Commands:
    run       Run a study from config
    index     Build retrieval indexes
    compare   Compare experiment results
    evaluate  Compute metrics on predictions
    backup    Backup artifacts to Backblaze B2
    download  Download artifacts from Backblaze B2
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
    cmd_compare,
    cmd_download,
    cmd_evaluate,
    cmd_health,
    cmd_index,
    cmd_metrics,
    cmd_resume,
    cmd_run,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ragicamp",
        description="RAGiCamp - RAG Experimentation Framework",
    )
    from ragicamp import __version__

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a study from config")
    run_parser.add_argument("config", type=Path, help="Study config YAML")
    run_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    run_parser.add_argument("--skip-existing", action="store_true", help="Skip completed")
    run_parser.add_argument("--validate", action="store_true", help="Validate config only")
    # Random search options (override config)
    run_parser.add_argument(
        "--sample",
        "-s",
        type=int,
        metavar="N",
        help="Random sample N experiments from grid (enables random search)",
    )
    run_parser.add_argument(
        "--sample-mode",
        choices=["random", "stratified"],
        default="random",
        help="Sampling mode: random (uniform) or stratified (ensure coverage)",
    )
    run_parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    run_parser.add_argument(
        "--stratify-by",
        type=str,
        default="model,retriever",
        help="Dimensions to stratify by (comma-separated, for stratified mode)",
    )
    run_parser.set_defaults(func=cmd_run)

    # Index command
    index_parser = subparsers.add_parser("index", help="Build retrieval index")
    index_parser.add_argument(
        "--corpus", default="simple", help="Corpus: simple, en, or full version string"
    )
    index_parser.add_argument(
        "--embedding",
        default="minilm",
        help="Embedding: minilm, e5, mpnet, or full model name",
    )
    index_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in chars")
    index_parser.add_argument("--max-docs", type=int, default=None, help="Max documents to index")
    index_parser.set_defaults(func=cmd_index)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare results")
    compare_parser.add_argument("output_dir", type=Path, help="Output directory")
    compare_parser.add_argument("--top", type=int, default=10, help="Show top N results")
    compare_parser.add_argument("--metric", "-m", default="f1", help="Metric to compare")
    compare_parser.add_argument(
        "--group-by",
        "-g",
        default="model",
        choices=["model", "dataset", "prompt", "retriever", "quantization", "type"],
        help="Dimension to group by",
    )
    compare_parser.add_argument(
        "--pivot",
        nargs=2,
        metavar=("ROWS", "COLS"),
        help="Create pivot table",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Compute metrics")
    eval_parser.add_argument("predictions", type=Path, help="Predictions JSON")
    eval_parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1", "exact_match"],
        help="Metrics: f1, exact_match, llm_judge_qa, bertscore, bleurt",
    )
    eval_parser.add_argument("--output", type=Path, help="Output file")
    eval_parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Model for LLM judge (default: gpt-4o-mini)",
    )
    eval_parser.add_argument(
        "--judgment-type",
        choices=["binary", "ternary"],
        default="binary",
        help="LLM judge type: binary or ternary",
    )
    eval_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max concurrent API calls for LLM judge (default: 20)",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Health command
    health_parser = subparsers.add_parser("health", help="Check experiment health")
    health_parser.add_argument("output_dir", type=Path, help="Output directory")
    health_parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metrics to check (e.g., f1,exact_match,llm_judge)",
    )
    health_parser.set_defaults(func=cmd_health)

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume incomplete experiments")
    resume_parser.add_argument("output_dir", type=Path, help="Output directory")
    resume_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    resume_parser.set_defaults(func=cmd_resume)

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Recompute metrics for an experiment")
    metrics_parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    metrics_parser.add_argument(
        "--metrics",
        "-m",
        required=True,
        help="Comma-separated metrics (e.g., f1,exact_match,llm_judge)",
    )
    metrics_parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Model for LLM judge (default: gpt-4o-mini)",
    )
    metrics_parser.set_defaults(func=cmd_metrics)

    # Backup command
    backup_parser = subparsers.add_parser(
        "backup", help="Backup artifacts and outputs to Backblaze B2"
    )
    backup_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Directory to backup (default: artifacts/ and outputs/)",
    )
    backup_parser.add_argument(
        "--bucket",
        "-b",
        default="masters-bucket",
        help="B2 bucket name (default: masters-bucket)",
    )
    backup_parser.add_argument(
        "--prefix",
        "-p",
        default=None,
        help="S3 key prefix (default: ragicamp-backup/YYYYMMDD-HHMMSS)",
    )
    backup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without uploading",
    )
    backup_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue uploading if some files fail",
    )
    backup_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=12,
        help="Number of parallel upload threads (default: 12)",
    )
    backup_parser.set_defaults(func=cmd_backup)

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download artifacts and outputs from Backblaze B2 backup"
    )
    download_parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available backups",
    )
    download_parser.add_argument(
        "--backup",
        "-b",
        default=None,
        help="Backup name to download (default: most recent)",
    )
    download_parser.add_argument(
        "--bucket",
        default="masters-bucket",
        help="B2 bucket name (default: masters-bucket)",
    )
    download_parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Only download artifacts/ (indexes, retrievers)",
    )
    download_parser.add_argument(
        "--outputs-only",
        action="store_true",
        help="Only download outputs/ (experiment results)",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without downloading",
    )
    download_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue downloading if some files fail",
    )
    download_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=12,
        help="Number of parallel download threads (default: 12)",
    )
    download_parser.set_defaults(func=cmd_download)

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
