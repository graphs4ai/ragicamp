# RAGiCamp

A modular framework for benchmarking RAG (Retrieval-Augmented Generation) strategies across models, retrievers, and datasets. Designed for controlled experiments at scale with Optuna-powered hyperparameter optimization.

## Features

- **Phased Execution** — INIT → GENERATING → COMPUTING_METRICS → COMPLETE with automatic resume
- **Subprocess Isolation** — Each experiment runs in its own process for CUDA crash safety
- **Optuna TPE Search** — Bayesian optimization over the full RAG grid (models × retrievers × top_k × prompts × agents)
- **Multiple Agents** — DirectLLM (baseline), FixedRAG, IterativeRAG (multi-iteration), SelfRAG (adaptive retrieval)
- **Multiple Backends** — vLLM (primary), HuggingFace, OpenAI
- **Advanced Retrieval** — Dense (FAISS), Hybrid (dense + BM25/TF-IDF), Hierarchical (parent-child chunks)
- **Query Transforms** — HyDE (hypothetical document embeddings), MultiQuery expansion
- **Cross-Encoder Reranking** — BGE reranker with configurable fetch_k
- **SQLite Caching** — Embedding and retrieval result caching across experiments
- **Comprehensive Metrics** — F1, Exact Match, BERTScore, BLEURT, LLM-as-Judge, Faithfulness, Hallucination

## Quick Start

```bash
# Install
uv sync

# Run a study (dry-run to preview)
uv run ragicamp run conf/study/smart_retrieval_slm.yaml --dry-run

# Run for real (resumes automatically on crash)
uv run ragicamp run conf/study/smart_retrieval_slm.yaml

# Check experiment health
uv run ragicamp health outputs/smart_retrieval_slm
```

## How It Works

```
YAML config
  → cli/study.py:run_study()
  → spec/builder.py:build_specs() → list[ExperimentSpec]
  → execution/runner.py:run_spec() per spec (subprocess)
  → experiment.py:Experiment.from_spec() → agent.run()
```

1. **Config validated** — Pydantic schemas at `config/schemas.py`
2. **Indexes ensured** — Dense (FAISS HNSW) and sparse (BM25/TF-IDF) built if missing
3. **Specs built** — Grid search, Optuna TPE sampling, or explicit singletons
4. **Each spec executed** in a subprocess for CUDA crash isolation
5. **Phases per experiment** — INIT → GENERATING → COMPUTING_METRICS → COMPLETE
6. **Phase-aware resume** — Crashed in COMPUTING_METRICS? Skips model loading entirely

## Project Structure

```
ragicamp/
├── src/ragicamp/
│   ├── agents/             # Agent implementations
│   │   ├── base.py         # Base class, shared embed+search logic
│   │   ├── fixed_rag.py    # Standard single-pass RAG
│   │   ├── iterative_rag.py # Multi-iteration query refinement
│   │   └── self_rag.py     # Adaptive retrieval (model decides when to retrieve)
│   ├── cache/              # SQLite-backed caching
│   │   ├── embedding_store.py   # (model, text_hash) → float32 vectors
│   │   └── retrieval_store.py   # (retriever, query_hash, top_k) → search results
│   ├── cli/                # CLI entry points
│   │   ├── main.py         # Entry: ragicamp.cli.main:main
│   │   ├── study.py        # run_study() orchestration
│   │   ├── commands.py     # CLI command handlers
│   │   └── trace.py        # Experiment step tracing
│   ├── config/             # Pydantic schemas and validation
│   ├── execution/          # Experiment execution pipeline
│   │   ├── runner.py       # run_spec(), run_generation(), run_metrics_only()
│   │   ├── executor.py     # ResilientExecutor for batch processing
│   │   └── phases/         # Phase handlers (init, generation, metrics)
│   ├── experiment.py       # Experiment class: from_spec() factory, run()
│   ├── factory/            # Component factories (agents, providers, metrics)
│   ├── indexes/            # FAISS vector indexes, BM25 sparse, hierarchical
│   ├── metrics/            # Evaluation metrics
│   ├── models/             # Model backends
│   │   ├── vllm.py         # vLLM backend (primary)
│   │   ├── huggingface.py  # HuggingFace transformers
│   │   ├── openai.py       # OpenAI API
│   │   └── providers/      # Lazy-loading providers with GPU lifecycle
│   ├── optimization/       # Optuna hyperparameter search
│   ├── rag/                # Query transforms, chunking, rerankers
│   ├── retrievers/         # Dense, hybrid (RRF fusion), hierarchical
│   ├── spec/               # Experiment specification (frozen dataclass)
│   │   ├── experiment.py   # ExperimentSpec — single source of truth
│   │   ├── builder.py      # build_specs() from YAML
│   │   └── naming.py       # Hash-based experiment naming
│   ├── state/              # Experiment state and health checks
│   └── utils/              # Atomic IO, prompt builder, resource manager
├── conf/study/             # Study YAML configs
├── scripts/                # Utility and migration scripts
├── notebooks/              # Analysis notebooks
├── artifacts/              # Indexes and caches
└── outputs/                # Experiment results
```

## CLI Commands

### Run a Study

```bash
ragicamp run <config.yaml> [OPTIONS]
  --dry-run              Preview experiments and their status
  --skip-existing        Skip completed experiments
  --validate             Validate config only
  --limit N              Max examples per experiment
  --force                Force re-run even if complete/failed
  --sample N             Sample N experiments (random or TPE)
  --sample-mode          random | tpe (default: random)
  --optimize-metric      Metric to optimize (default: f1)
```

### Check Health

```bash
ragicamp health <output_dir> [--metrics f1,exact_match]
```

### Resume Incomplete

```bash
ragicamp resume <output_dir> [--dry-run]
```

### Recompute Metrics

```bash
ragicamp metrics <exp_dir> -m f1,bertscore,llm_judge
  --judge-model          Model for LLM judge (default: gpt-4o-mini)
```

### Compare Results

```bash
ragicamp compare <output_dir> [OPTIONS]
  --metric, -m           Metric to compare (default: f1)
  --group-by, -g         model | dataset | prompt | retriever | type
  --pivot ROWS COLS      Create pivot table
  --top N                Show top N results (default: 10)
```

### Evaluate Predictions

```bash
ragicamp evaluate <predictions.json> --metrics f1 exact_match llm_judge_qa
  --judge-model          Model for LLM judge
  --output               Output file path
```

### Build Index

```bash
ragicamp index [OPTIONS]
  --corpus               simple | en | full version string
  --embedding            minilm | e5 | mpnet | full model name
  --chunk-size           Chunk size in chars (default: 512)
  --max-docs             Max documents to index
```

### Cache Management

```bash
ragicamp cache stats              # Show cache statistics
ragicamp cache clear [--model X]  # Clear embeddings (optionally by model)
```

### Backup / Download (Backblaze B2)

```bash
ragicamp backup [path] --bucket masters-bucket --sync --dry-run
ragicamp download --list                          # List available backups
ragicamp download --backup <name> --artifacts-only
```

## Study Configuration

Studies are defined in YAML configs that specify the full experimental grid:

```yaml
name: my_study
description: "My RAG experiments"
num_questions: 1000
datasets: [nq, triviaqa, hotpotqa]

metrics:
  - f1
  - exact_match
  - bertscore

# Direct baselines (no retrieval)
direct:
  enabled: true
  models:
    - vllm:Qwen/Qwen2.5-7B-Instruct
    - vllm:meta-llama/Llama-3.2-3B-Instruct
  prompts: [concise, fewshot_3]

# RAG experiments
rag:
  enabled: true
  models:
    - vllm:Qwen/Qwen2.5-7B-Instruct

  # Optuna TPE optimization
  sampling:
    mode: tpe
    n_experiments: 1000
    optimize_metric: f1
    seed: 42
    agent_types: [fixed_rag, iterative_rag, self_rag]
    agent_params:
      iterative_rag:
        max_iterations: [1, 2, 3]
      self_rag:
        retrieval_threshold: [0.3, 0.5, 0.7]
        verify_answer: [true, false]

  # Retrievers
  retrievers:
    - type: dense
      name: dense_bge_large_512
      embedding_model: BAAI/bge-large-en-v1.5
      # ...
    - type: hybrid
      name: hybrid_bge_large_bm25
      sparse_method: bm25
      alpha: 0.5  # Overridden by Optuna
      # ...

  retriever_names: [dense_bge_large_512, hybrid_bge_large_bm25]
  top_k_values: [3, 5, 10, 15, 20]
  query_transform: [none, hyde, multiquery]
  alpha_values: [0.3, 0.5, 0.7, 0.9]  # Hybrid-only, explored by Optuna

  reranker:
    configs:
      - enabled: false
        name: none
      - enabled: true
        name: bge

  prompts: [concise, extractive, cot, fewshot_3]

output_dir: outputs/my_study
batch_size: 128
```

See `conf/study/smart_retrieval_slm.yaml` for a complete production example.

### Model Spec Format

| Provider | Format | Example |
|----------|--------|---------|
| vLLM | `vllm:org/model` | `vllm:Qwen/Qwen2.5-7B-Instruct` |
| HuggingFace | `hf:org/model` | `hf:google/gemma-2-2b-it` |
| OpenAI | `openai:model` | `openai:gpt-4o-mini` |

## Experiment Output

Each experiment produces:

```
outputs/my_study/rag_Qwen257BI_nq_a3f8c2d1/
├── state.json          # Phase tracking
├── questions.json      # Exported questions
├── metadata.json       # Full experiment config
├── predictions.json    # Answers + per-item metrics + timing
├── results.json        # Aggregate metrics + timing profile
└── experiment.log      # Subprocess stdout/stderr
```

Experiment names use a hash-based scheme: `{type}_{model_short}_{dataset}_{hash8}`.

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/validate_cache.py` | Validate cache integrity (structure, dimensions, cross-reference) |
| `scripts/clean_study.py` | Diagnose and fix incorrectly named experiments |
| `scripts/migrate_naming.py` | Migrate old long names to hash-based names |
| `scripts/migrate_audit_fixes.py` | Apply audit fix migrations to existing results |
| `scripts/repair_metadata.py` | Repair metadata.json files |
| `scripts/quarantine_affected_exps.py` | Quarantine experiments affected by bugs |
| `scripts/analyze_retrieval.py` | Analyze retrieval quality |

## Analysis Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_overview.ipynb` | Dataset statistics and distributions |
| `02_rag_effectiveness.ipynb` | RAG vs Direct LLM comparison |
| `03_component_analysis.ipynb` | Ablation by retriever, prompt, model |
| `04_next_experiments.ipynb` | Experiment planning from results |
| `05_failure_analysis.ipynb` | Error patterns and failure modes |
| `06_pipeline_profiling.ipynb` | Phase and metric timing analysis |
| `rag_strategy_analysis.ipynb` | Agent strategy comparison |
| `smart_retrieval_analysis.ipynb` | Smart retrieval study results |

## Development

```bash
# Install
uv sync

# Tests (no GPU needed)
uv run pytest tests/ -x -q

# Lint & format
make lint       # ruff format --check + ruff check
make format     # ruff format + ruff check --fix

# Pre-push check
make pre-push   # format + lint + test
```

## Documentation

- [Metrics Guide](docs/guides/METRICS.md)
- [Optuna Study Design](docs/OPTUNA_STUDY_DESIGN.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Experiment Configurations](docs/EXPERIMENT_CONFIGURATIONS.md)
- [Future Work](docs/FUTURE_WORK.md)
- [Cheatsheet](CHEATSHEET.md)

## License

MIT
