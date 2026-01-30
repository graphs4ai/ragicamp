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
    from ragicamp.experiment_state import ExperimentPhase, check_health

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
    print(f"\nTo resume, run the original study config with --skip-existing=False")
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
