#!/usr/bin/env python3
"""RAGiCamp CLI - Unified command-line interface.

Commands:
    run       Run a study from config
    index     Build retrieval indexes
    compare   Compare experiment results
    evaluate  Compute metrics on predictions
"""

# ============================================================================
# CRITICAL: Configure TensorFlow BEFORE any library imports!
# TensorFlow is transitively imported by transformers/sentence-transformers.
# By default, TF allocates ALL GPU memory on import, causing OOM.
# ============================================================================
import os

if "TF_FORCE_GPU_ALLOW_GROWTH" not in os.environ:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml


def cmd_run(args: argparse.Namespace) -> int:
    """Run a study from config file."""
    from ragicamp.cli.study import run_study

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        return 1

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_study(
        config,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        validate_only=args.validate,
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Build retrieval indexes."""
    from ragicamp.corpus import ChunkConfig, CorpusConfig, DocumentChunker, WikipediaCorpus
    from ragicamp.retrievers import DenseRetriever

    # Map short names to embedding models
    embedding_models = {
        "minilm": "all-MiniLM-L6-v2",
        "e5": "intfloat/e5-small-v2",
        "mpnet": "all-mpnet-base-v2",
    }
    embedding_model = embedding_models.get(args.embedding, args.embedding)

    # Map short names to corpus versions
    corpus_versions = {
        "simple": "20231101.simple",
        "en": "20231101.en",
    }
    corpus_version = corpus_versions.get(args.corpus, args.corpus)

    index_name = f"{args.corpus}_{args.embedding}_recursive_{args.chunk_size}"
    print(f"Building index: {index_name}")

    # Load corpus
    corpus_config = CorpusConfig(
        name=f"wikipedia_{args.corpus}",
        source="wikimedia/wikipedia",
        version=corpus_version,
        max_docs=args.max_docs,
    )
    corpus = WikipediaCorpus(corpus_config)
    docs = list(corpus.load())
    print(f"Loaded {len(docs)} documents")

    # Chunk documents
    chunk_config = ChunkConfig(
        strategy="recursive",
        chunk_size=args.chunk_size,
        chunk_overlap=50,
    )
    chunker = DocumentChunker(chunk_config)
    chunks = list(chunker.chunk_documents(docs, show_progress=True))
    print(f"Created {len(chunks)} chunks")

    # Build index
    retriever = DenseRetriever(
        name=index_name,
        embedding_model=embedding_model,
    )
    retriever.index_documents(chunks)
    retriever.save(index_name)

    print(f"Index saved: {index_name}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare experiment results."""
    from ragicamp.analysis import (
        ResultsLoader,
        best_by,
        compare_results,
        format_comparison_table,
        pivot_results,
        summarize_results,
    )

    loader = ResultsLoader(args.output_dir)
    results = loader.load_all()

    if not results:
        print("No results found")
        return 1

    print(f"Loaded {len(results)} experiments\n")

    # Summary
    summary = summarize_results(results)
    print(f"Models: {', '.join(summary['models'])}")
    print(f"Datasets: {', '.join(summary['datasets'])}")
    print(f"Best F1: {summary['best_f1']['value']:.4f} ({summary['best_f1']['model']})")
    print()

    # Comparison by requested dimension
    stats = compare_results(results, group_by=args.group_by, metric=args.metric)
    print(format_comparison_table(stats, title=f"By {args.group_by}", metric=args.metric))

    # Pivot table if requested
    if args.pivot:
        pivot = pivot_results(results, rows=args.pivot[0], cols=args.pivot[1], metric=args.metric)
        print(f"\nPivot: {args.pivot[0]} x {args.pivot[1]}")
        for row, cols in sorted(pivot.items()):
            print(f"  {row[:20]}: {', '.join(f'{c}={v:.3f}' for c, v in sorted(cols.items()))}")

    # Top N
    print(f"\nTop {args.top} by {args.metric}:")
    for i, r in enumerate(best_by(results, metric=args.metric, n=args.top), 1):
        val = getattr(r, args.metric, 0)
        print(f"  {i}. {r.name[:50]} = {val:.4f}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compute metrics on predictions file."""
    import os

    from ragicamp.evaluation import compute_metrics_from_file
    from ragicamp.metrics import ExactMatchMetric, F1Metric

    metrics = []
    for name in args.metrics:
        if name == "f1":
            metrics.append(F1Metric())
        elif name == "exact_match":
            metrics.append(ExactMatchMetric())
        elif name in ("llm_judge", "llm_judge_qa"):
            # LLM Judge requires OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY not set. Required for llm_judge_qa.")
                print("Set it with: export OPENAI_API_KEY='your-key'")
                return 1

            from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric
            from ragicamp.models.openai import OpenAIModel

            judge_model_name = args.judge_model
            print(f"Using judge model: {judge_model_name} (max_concurrent={args.max_concurrent})")
            judge_model = OpenAIModel(judge_model_name, temperature=0.0)
            metrics.append(
                LLMJudgeQAMetric(
                    judge_model=judge_model,
                    judgment_type=args.judgment_type,
                    max_concurrent=args.max_concurrent,
                )
            )
        elif name == "bertscore":
            from ragicamp.metrics import BertScoreMetric

            metrics.append(BertScoreMetric())
        elif name == "bleurt":
            from ragicamp.metrics import BLEURTMetric

            metrics.append(BLEURTMetric())

    if not metrics:
        print(
            "No valid metrics specified. Available: f1, exact_match, llm_judge_qa, bertscore, bleurt"
        )
        return 1

    print(f"Computing metrics: {[m.name for m in metrics]}")
    print(f"Predictions file: {args.predictions}")

    results = compute_metrics_from_file(
        predictions_path=str(args.predictions),
        metrics=metrics,
        output_path=str(args.output) if args.output else None,
    )

    print("\nResults:")
    for key, value in results.get("aggregate", {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check health of experiments in a directory."""
    from ragicamp.experiment_state import check_health

    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        return 1

    # Find experiment directories
    exp_dirs = [
        d
        for d in output_dir.iterdir()
        if d.is_dir()
        and (d / "state.json").exists()
        or (d / "predictions.json").exists()
        or (d / "results.json").exists()
    ]

    if not exp_dirs:
        print(f"No experiments found in {output_dir}")
        return 1

    print(f"Checking {len(exp_dirs)} experiments in {output_dir}\n")

    # Status counts
    complete = 0
    incomplete = 0
    failed = 0

    for exp_dir in sorted(exp_dirs):
        health = check_health(exp_dir, args.metrics.split(",") if args.metrics else None)
        print(f"  {health.summary()} - {exp_dir.name}")

        if health.is_complete:
            complete += 1
        elif health.phase.value == "failed":
            failed += 1
        else:
            incomplete += 1

    print(f"\nSummary: {complete} complete, {incomplete} incomplete, {failed} failed")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume incomplete experiments."""
    from ragicamp.experiment_state import check_health

    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        return 1

    # Find experiment directories
    exp_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    if not exp_dirs:
        print(f"No experiments found in {output_dir}")
        return 1

    # Find incomplete experiments
    to_resume = []
    for exp_dir in sorted(exp_dirs):
        health = check_health(exp_dir)
        if health.can_resume and not health.is_complete:
            to_resume.append((exp_dir, health))

    if not to_resume:
        print("All experiments are complete or failed.")
        return 0

    print(f"Found {len(to_resume)} experiments to resume:\n")
    for exp_dir, health in to_resume:
        print(f"  {health.summary()} - {exp_dir.name}")

    if args.dry_run:
        print("\n[DRY RUN] - no changes made")
        return 0

    # Note: Actually resuming would require loading the original config
    # For now, just report what needs to be done
    print("\nTo resume, run the original study config with --skip-existing=False")
    print("Or use `ragicamp metrics <dir>` to recompute just metrics")
    return 0


def _get_b2_client():
    """Get B2 S3 client and credentials."""
    import os

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("Error: boto3 not installed. Install with: uv pip install boto3")
        return None, None, None, None

    key_id = os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    app_key = os.environ.get("B2_APP_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")

    if not key_id or not app_key:
        print("Error: Backblaze credentials not set.")
        print("Set environment variables:")
        print("  export B2_KEY_ID='your-key-id'")
        print("  export B2_APP_KEY='your-application-key'")
        print("  export B2_ENDPOINT='https://s3.us-east-005.backblazeb2.com'  # optional")
        return None, None, None, None

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(signature_version="s3v4"),
    )

    return s3, endpoint, key_id, app_key


def cmd_backup(args: argparse.Namespace) -> int:
    """Backup artifacts and outputs to Backblaze B2."""
    import os
    import time
    from datetime import datetime

    s3, endpoint, _, _ = _get_b2_client()
    if s3 is None:
        return 1

    # Determine directories to backup
    if args.path:
        dirs_to_backup = [args.path]
    else:
        dirs_to_backup = []
        for default_dir in ["artifacts", "outputs"]:
            p = Path(default_dir)
            if p.exists():
                dirs_to_backup.append(p)

        if not dirs_to_backup:
            print("No artifacts/ or outputs/ directories found.")
            print("Specify a path explicitly: ragicamp backup <path>")
            return 1

    bucket = args.bucket
    prefix = args.prefix or f"ragicamp-backup/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"Backing up to: s3://{bucket}/{prefix}/")
    print(f"Source directories: {', '.join(str(d) for d in dirs_to_backup)}")

    # Collect files to upload
    files_to_upload = []
    for backup_dir in dirs_to_backup:
        for root, dirs, files in os.walk(backup_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for file in files:
                if file.startswith("."):
                    continue
                local_path = Path(root) / file
                relative_path = local_path.relative_to(backup_dir.parent)
                s3_key = f"{prefix}/{relative_path}"
                files_to_upload.append((local_path, s3_key))

    if not files_to_upload:
        print("No files to upload.")
        return 0

    total_bytes = sum(f[0].stat().st_size for f in files_to_upload)
    print(f"Found {len(files_to_upload)} files ({total_bytes / (1024**3):.2f} GB)")

    if args.dry_run:
        print("\n[DRY RUN] Would upload:")
        for local_path, s3_key in files_to_upload[:10]:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  {s3_key} ({size_mb:.1f} MB)")
        if len(files_to_upload) > 10:
            print(f"  ... and {len(files_to_upload) - 10} more files")
        return 0

    # Upload files with byte-level progress
    from tqdm import tqdm

    uploaded_bytes = 0
    uploaded_files = 0
    errors = []
    start_time = time.time()

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Uploading") as pbar:
        for local_path, s3_key in files_to_upload:
            file_size = local_path.stat().st_size
            try:
                s3.upload_file(str(local_path), bucket, s3_key)
                uploaded_files += 1
                uploaded_bytes += file_size
                pbar.update(file_size)
            except Exception as e:
                errors.append((local_path, str(e)))
                pbar.update(file_size)  # Still update progress
                if not args.continue_on_error:
                    print(f"\nError uploading {local_path}: {e}")
                    return 1

    elapsed = time.time() - start_time
    speed_mbps = (uploaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Uploaded {uploaded_files}/{len(files_to_upload)} files to s3://{bucket}/{prefix}/")
    print(f"  Total: {uploaded_bytes / (1024**3):.2f} GB in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)")

    if errors:
        print(f"\n⚠️  {len(errors)} errors:")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download artifacts and outputs from Backblaze B2 backup."""
    import time

    s3, endpoint, _, _ = _get_b2_client()
    if s3 is None:
        return 1

    bucket = args.bucket

    # List available backups
    if args.list:
        print(f"Available backups in s3://{bucket}/ragicamp-backup/:")
        print("=" * 50)
        try:
            response = s3.list_objects_v2(
                Bucket=bucket, Prefix="ragicamp-backup/", Delimiter="/"
            )
            prefixes = response.get("CommonPrefixes", [])
            if not prefixes:
                print("  No backups found.")
                return 0

            # Sort by name (timestamp) descending
            backup_names = sorted(
                [p["Prefix"].split("/")[1] for p in prefixes], reverse=True
            )
            for name in backup_names[:20]:
                print(f"  {name}")
            if len(backup_names) > 20:
                print(f"  ... and {len(backup_names) - 20} more")
        except Exception as e:
            print(f"Error listing backups: {e}")
            return 1
        return 0

    # Determine which backup to download
    if args.backup:
        backup_name = args.backup
    else:
        # Get most recent backup
        try:
            response = s3.list_objects_v2(
                Bucket=bucket, Prefix="ragicamp-backup/", Delimiter="/"
            )
            prefixes = response.get("CommonPrefixes", [])
            if not prefixes:
                print("No backups found.")
                return 1

            backup_names = sorted(
                [p["Prefix"].split("/")[1] for p in prefixes], reverse=True
            )
            backup_name = backup_names[0]
            print(f"Most recent backup: {backup_name}")
        except Exception as e:
            print(f"Error finding backups: {e}")
            return 1

    prefix = f"ragicamp-backup/{backup_name}"
    print(f"Downloading from: s3://{bucket}/{prefix}/")

    # List all files in the backup
    files_to_download = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                size = obj["Size"]
                # Extract local path: remove prefix, keep artifacts/... or outputs/...
                relative_path = s3_key[len(prefix) + 1 :]  # +1 for trailing /
                if not relative_path:
                    continue

                # Filter by artifacts-only or outputs-only
                if args.artifacts_only and not relative_path.startswith("artifacts"):
                    continue
                if args.outputs_only and not relative_path.startswith("outputs"):
                    continue

                local_path = Path(relative_path)
                files_to_download.append((s3_key, local_path, size))
    except Exception as e:
        print(f"Error listing backup contents: {e}")
        return 1

    if not files_to_download:
        print("No files to download.")
        return 0

    total_bytes = sum(f[2] for f in files_to_download)
    print(f"Found {len(files_to_download)} files ({total_bytes / (1024**3):.2f} GB)")

    if args.dry_run:
        print("\n[DRY RUN] Would download:")
        for s3_key, local_path, size in files_to_download[:15]:
            print(f"  {local_path} ({size / (1024*1024):.1f} MB)")
        if len(files_to_download) > 15:
            print(f"  ... and {len(files_to_download) - 15} more files")
        return 0

    # Download files with progress
    from tqdm import tqdm

    downloaded_bytes = 0
    downloaded_files = 0
    errors = []
    start_time = time.time()

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for s3_key, local_path, size in files_to_download:
            try:
                # Create parent directories
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, s3_key, str(local_path))
                downloaded_files += 1
                downloaded_bytes += size
                pbar.update(size)
            except Exception as e:
                errors.append((local_path, str(e)))
                pbar.update(size)
                if not args.continue_on_error:
                    print(f"\nError downloading {s3_key}: {e}")
                    return 1

    elapsed = time.time() - start_time
    speed_mbps = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Downloaded {downloaded_files}/{len(files_to_download)} files")
    print(f"  Total: {downloaded_bytes / (1024**3):.2f} GB in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)")

    if errors:
        print(f"\n⚠️  {len(errors)} errors:")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    # Show what was downloaded
    print("\nDownloaded contents:")
    if Path("artifacts").exists():
        print(f"  artifacts/: {sum(1 for _ in Path('artifacts').rglob('*') if _.is_file())} files")
    if Path("outputs").exists():
        print(f"  outputs/: {sum(1 for _ in Path('outputs').rglob('*') if _.is_file())} files")

    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Recompute metrics for an experiment."""
    import os

    from ragicamp.evaluation import compute_metrics_from_file
    from ragicamp.factory import ComponentFactory

    exp_dir = args.exp_dir
    predictions_path = exp_dir / "predictions.json"

    if not predictions_path.exists():
        print(f"Predictions not found: {predictions_path}")
        return 1

    # Parse metrics
    metric_names = [m.strip() for m in args.metrics.split(",")]

    # Build judge model if needed
    judge_model = None
    if any(m in ("llm_judge", "llm_judge_qa") for m in metric_names):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set. Required for llm_judge_qa.")
            return 1
        from ragicamp.models.openai import OpenAIModel

        judge_model = OpenAIModel(args.judge_model, temperature=0.0)

    metrics = ComponentFactory.create_metrics(metric_names, judge_model=judge_model)

    print(f"Computing metrics: {metric_names}")
    print(f"Experiment: {exp_dir.name}")

    results = compute_metrics_from_file(
        predictions_path=str(predictions_path),
        metrics=metrics,
        output_path=str(predictions_path),  # Update in place
    )

    print("\nResults:")
    for key, value in results.get("aggregate", {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Update results.json if it exists
    results_path = exp_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            result_data = json.load(f)
        result_data["metrics"].update(results.get("aggregate", {}))
        with open(results_path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n✓ Updated {results_path}")

    return 0


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
