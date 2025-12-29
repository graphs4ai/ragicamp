#!/usr/bin/env python3
"""RAGiCamp CLI - Unified command-line interface.

Commands:
    run       Run a study from config
    index     Build retrieval indexes
    compare   Compare experiment results
    evaluate  Compute metrics on predictions
"""

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

    # MLflow
    if args.mlflow:
        try:
            from ragicamp.analysis import MLflowTracker

            exp_name = args.mlflow_experiment or args.output_dir.name
            tracker = MLflowTracker(exp_name)
            logged = tracker.backfill_from_results(results, skip_existing=True)
            print(f"\n✓ Logged {logged} experiments to MLflow ({exp_name})")
        except ImportError:
            print("\n⚠️  MLflow not installed")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compute metrics on predictions file."""
    from ragicamp.evaluation import Evaluator
    from ragicamp.metrics import ExactMatchMetric, F1Metric

    metrics = []
    for name in args.metrics:
        if name == "f1":
            metrics.append(F1Metric())
        elif name == "exact_match":
            metrics.append(ExactMatchMetric())

    if not metrics:
        print("No valid metrics specified")
        return 1

    results = Evaluator.compute_metrics_from_file(
        predictions_path=str(args.predictions),
        metrics=metrics,
        output_path=str(args.output) if args.output else None,
    )

    print("\nResults:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ragicamp",
        description="RAGiCamp - RAG Experimentation Framework",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.3.0")

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
    compare_parser.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    compare_parser.add_argument("--mlflow-experiment", help="MLflow experiment name")
    compare_parser.set_defaults(func=cmd_compare)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Compute metrics")
    eval_parser.add_argument("predictions", type=Path, help="Predictions JSON")
    eval_parser.add_argument("--metrics", nargs="+", default=["f1", "exact_match"], help="Metrics")
    eval_parser.add_argument("--output", type=Path, help="Output file")
    eval_parser.set_defaults(func=cmd_evaluate)

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
