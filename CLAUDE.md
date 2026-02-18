# CLAUDE.md — RAGiCamp AI Agent Guide

## Project Overview

RAGiCamp is a RAG (Retrieval-Augmented Generation) benchmarking framework for running controlled experiments across different retrieval strategies, models, and datasets. Python 3.10+, managed with `uv`.

## Quick Commands

```bash
# Tests (~39 test files, no GPU needed)
uv run pytest tests/ -x -q

# Lint & format
make lint             # ruff format --check + ruff check
make format           # ruff format + ruff check --fix

# Run a study
uv run ragicamp study conf/study/your_config.yaml

# Index building
make index-simple     # 1 small index (500 docs)
make index-full       # All standard indexes

# Quick experiment
make run-baseline-simple     # 10-question test
make run-baseline-full       # Full 100+ question baseline

# Analysis
make compare DIR=outputs/simple
make evaluate DIR=outputs/simple METRICS=bertscore,bleurt
```

## Source Layout

```
src/ragicamp/
├── agents/             # Agent implementations (direct_llm, fixed_rag, iterative_rag, self_rag)
│   └── base.py         # Agent base class, batch_transform_embed_and_search()
├── analysis/           # Post-experiment comparison and visualization
├── cache/              # SQLite-backed embedding and retrieval caching
│   └── embedding_store.py  # EmbeddingStore: (model, text_hash) → float32 vectors
├── cli/                # CLI entry points
│   ├── main.py         # Entry: ragicamp.cli.main:main
│   ├── study.py        # run_study() orchestration
│   ├── commands.py     # CLI command handlers
│   └── trace.py        # trace_experiment(), format_trace_report()
├── config/             # Pydantic schemas and validation
│   └── schemas.py      # ModelConfig, DatasetConfig, ChunkingConfig, RetrieverConfig
├── core/               # Core types and utilities
│   ├── constants.py    # AgentType enum, MetricType, Defaults
│   ├── step_types.py   # Step type constants (GENERATE, BATCH_SEARCH, etc.)
│   ├── logging.py      # get_logger(), configure_logging()
│   └── exceptions.py   # Custom exceptions
├── corpus/             # Corpus loading and chunking
├── datasets/           # QA dataset loaders (NQ, TriviaQA, HotpotQA, TechQA, PubMedQA)
├── execution/          # Experiment execution pipeline
│   ├── runner.py       # run_spec(), run_generation(), run_metrics_only()
│   ├── executor.py     # ResilientExecutor for batch processing
│   └── phases/         # Phase implementations (init, generation, metrics)
├── experiment.py       # Experiment class: from_spec() factory, run()
├── factory/            # Factory classes
│   ├── agents.py       # AgentFactory.from_spec() — wires providers + index → agent
│   ├── providers.py    # ProviderFactory for generators/embedders/rerankers
│   └── metrics.py      # MetricFactory
├── indexes/            # FAISS vector indexes, BM25 sparse indexes, hierarchical indexes
│   └── index_builder.py # IndexBuilder, build_index()
├── metrics/            # Evaluation metrics (exact_match, bertscore, bleurt, llm_judge, etc.)
├── models/             # Model interfaces and backends
│   ├── vllm.py         # vLLM backend
│   ├── huggingface.py  # HuggingFace transformers
│   ├── openai.py       # OpenAI API
│   └── providers/      # Lazy-loading providers with GPU lifecycle management
│       ├── generator.py    # GeneratorProvider (vllm/hf)
│       ├── embedder.py     # EmbedderProvider (vllm/sentence_transformers)
│       └── reranker.py     # RerankerProvider (cross-encoder)
├── optimization/       # Optuna hyperparameter search
├── rag/                # RAG components
│   ├── query_transform/    # HyDE, multi-query expansion
│   ├── chunking/           # Hierarchical chunking
│   └── rerankers/          # Cross-encoder reranking
├── retrievers/         # Retrieval backends
│   ├── lazy.py         # LazySearchBackend — defers index load until first search
│   ├── hybrid.py       # HybridSearcher (dense + sparse fusion)
│   └── hierarchical.py # HierarchicalSearcher (child → parent retrieval)
├── spec/               # Experiment specification (CANONICAL)
│   ├── experiment.py   # ExperimentSpec frozen dataclass — single source of truth
│   ├── builder.py      # build_specs() — grid search, random sampling, singletons
│   └── naming.py       # Experiment naming conventions
├── state/              # Experiment state and health checks
│   └── experiment_state.py  # ExperimentPhase enum, ExperimentState
└── utils/              # Utilities
    ├── experiment_io.py # ExperimentIO — atomic JSON writes (temp → rename)
    ├── prompts.py       # PromptBuilder for prompt templates
    └── resource_manager.py  # ResourceManager.clear_gpu_memory()
```

## Architecture: How a Study Runs

```
YAML config → cli/study.py:run_study()
           → spec/builder.py:build_specs() → list[ExperimentSpec]
           → execution/runner.py:run_spec() per spec (subprocess by default)
           → experiment.py:Experiment.from_spec() → agent.batch_answer()
```

1. **Config validated** — Pydantic schemas at `config/schemas.py`
2. **Indexes ensured** — Dense (FAISS) and sparse (BM25) built if missing
3. **Specs built** — Grid search (Cartesian product), random/TPE sampling, or explicit singletons
4. **Each spec executed** — In a subprocess for CUDA crash isolation
5. **Phases per experiment:**
   - `INIT` → saves questions.json, metadata.json
   - `GENERATING` → predictions.json updated incrementally
   - `GENERATED` → all predictions complete
   - `COMPUTING_METRICS` → metrics added to results
   - `COMPLETE` → results.json saved
6. **Phase-aware resume** — If crashed in COMPUTING_METRICS, skips model loading

Output per experiment: `state.json`, `questions.json`, `predictions.json`, `results.json`, `metadata.json`

## Key Abstractions

### ExperimentSpec (`spec/experiment.py`)
Frozen dataclass — the single source of truth for experiment configuration. Key fields: `name`, `exp_type`, `model`, `dataset`, `prompt`, `retriever`, `top_k`, `query_transform`, `reranker`, `agent_type`, `agent_params`.

### Provider Pattern (`models/providers/`)
`GeneratorProvider`, `EmbedderProvider`, `RerankerProvider` — context managers that lazy-load models onto GPU and guarantee cleanup:
```python
with provider.load(gpu_fraction=0.9) as generator:
    answers = generator.batch_generate(prompts)
# GPU memory freed after context exit
```

### AgentFactory (`factory/agents.py`)
`AgentFactory.from_spec(spec)` wires providers + index → agent instance. Agent types:
- `direct_llm` — No retrieval (baseline)
- `fixed_rag` — Single-shot RAG
- `iterative_rag` — Multi-iteration with sufficiency checking
- `self_rag` — Self-reflective with assessment/verification

### Step Tracing (`core/step_types.py`)
All agents emit `Step` objects with standardized type constants (`GENERATE`, `BATCH_SEARCH`, `QUERY_TRANSFORM`, `RERANK`, etc.) logged to predictions.json. Use `cli/trace.py` for debugging.

### LazySearchBackend (`retrievers/lazy.py`)
Proxy that defers FAISS index loading until first `batch_search()` call. If the retrieval cache has 100% hits, the index never loads — avoids multi-minute deserialization.

## Config Layer

Pydantic schemas at `config/schemas.py`: `ModelConfig`, `DatasetConfig`, `ChunkingConfig`, `RetrieverConfig`. YAML configs live at `conf/study/*.yaml`. Retriever configs (created at index build time) at `artifacts/retrievers/{name}/config.json`.

## Logging

```python
from ragicamp.core.logging import get_logger
logger = get_logger(__name__)
logger.info("Processing %s items", count)  # Use %s style, not f-strings
```

Study logs auto-saved to `{output_dir}/study.log` via `add_file_handler()`.

## Testing

```bash
uv run pytest tests/ -x -q
```

39 test files in `tests/`. No GPU needed. Fixtures in `conftest.py`.

## Gotchas & Conventions

- **Subprocess execution** — Experiments run in subprocesses for CUDA crash isolation
- **Atomic JSON writes** — All writes use temp-then-rename (`ExperimentIO._atomic_write()`)
- **Embedding cache** — Shared SQLite store at `artifacts/cache/ragicamp_cache.db` (override with `RAGICAMP_CACHE_DIR`). Keys: `(model_name, sha256(text)[:32])` → float32 blobs. WAL mode for concurrent reads.
- **Canonical spec** — `spec/experiment.py` is the single source of truth for ExperimentSpec. `core/constants.py` is canonical for AgentType.
- **Docs status** — All docs cleaned up (Feb 2026). Deprecated docs (IMPROVEMENT_PLAN.md, BACKLOG.md, EXPERIMENT_CONFIGURATIONS.md) have been deleted. See `docs/THESIS_STUDY.md` for current study status.
- **Query transforms** — Wired via `batch_transform_embed_and_search()` in `agents/base.py`. Factory creates HyDE/multiquery at `AgentFactory._create_query_transformer()`.
- **Ruff config** — line-length=100, target=py310
