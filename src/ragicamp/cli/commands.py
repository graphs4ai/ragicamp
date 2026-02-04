"""CLI command implementations."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml


def cmd_run(args: argparse.Namespace) -> int:
    """Run a study from config file."""
    from ragicamp.cli.study import run_study

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        return 1

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build sampling override from CLI args
    sampling_override = None
    if args.sample:
        sampling_override = {
            "mode": args.sample_mode,
            "n_experiments": args.sample,
            "seed": args.sample_seed,
        }
        if args.sample_mode == "stratified":
            sampling_override["stratify_by"] = [s.strip() for s in args.stratify_by.split(",")]
        print(f"🎲 Sampling mode: {args.sample_mode}, n={args.sample}")
        if args.sample_seed:
            print(f"   Seed: {args.sample_seed}")

    run_study(
        config,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        validate_only=args.validate,
        sampling_override=sampling_override,
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

    print("\nTo resume, run the original study config with --skip-existing=False")
    print("Or use `ragicamp metrics <dir>` to recompute just metrics")
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


def cmd_backup(args: argparse.Namespace) -> int:
    """Backup artifacts and outputs to Backblaze B2."""
    from ragicamp.cli.backup import backup

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

    prefix = args.prefix or f"ragicamp-backup/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    return backup(
        dirs_to_backup=dirs_to_backup,
        bucket=args.bucket,
        prefix=prefix,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        max_workers=args.workers,
    )


def cmd_download(args: argparse.Namespace) -> int:
    """Download artifacts and outputs from Backblaze B2 backup."""
    from ragicamp.cli.backup import download, list_backups

    # List mode
    if args.list:
        print(f"Available backups in s3://{args.bucket}/ragicamp-backup/:")
        print("=" * 50)
        backups = list_backups(args.bucket, limit=20)
        if not backups:
            print("  No backups found.")
        else:
            for name in backups:
                print(f"  {name}")
        return 0

    return download(
        bucket=args.bucket,
        backup_name=args.backup,
        artifacts_only=args.artifacts_only,
        outputs_only=args.outputs_only,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        max_workers=args.workers,
    )
