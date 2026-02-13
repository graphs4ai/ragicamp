# Changelog

All notable changes to RAGiCamp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.1] - 2026-02-13

### Engineering Audit & Code Review

Comprehensive codebase audit (67/68 issues fixed) plus external code review.

#### Data Integrity (P0) — 8 issues fixed
- **Fixed** Division-by-zero in embedding normalization (3 locations)
- **Fixed** CrossEncoderReranker mutating caller's Document objects
- **Fixed** Non-atomic JSON writes consolidated into `atomic_write_json()` utility
- **Fixed** BERTScoreMetric OOM retry discarding partial results
- **Fixed** HuggingFace wrong device for quantized models
- **Fixed** OpenAI error strings scored as predictions
- **Fixed** OpenAI hardcoded embedding model name
- **Fixed** `_stratified_sample` using global random instead of seeded RNG

#### Reliability (P1) — 14 issues fixed
- **Added** Provider ref-counting for GPU model reuse (eliminates redundant load/unload)
- **Fixed** SQLite connection leaks in embedding and retrieval caches
- **Fixed** HyDE/MultiQuery loading model on every call (now ref-counted)
- **Added** HyDE iteration-0-only guard in IterativeRAG
- **Fixed** Retrieval cache enabled for transformed queries
- **Optimized** HybridSearcher RRF fusion (50% fewer object allocations)
- **Fixed** Checkpoint callback skipping when batch crosses modulo boundary
- **Fixed** `_tee` thread losing output on subprocess timeout

#### Test Coverage (P2) — 10 gaps filled
- **Added** IterativeRAGAgent functional tests (11 test cases)
- **Added** SelfRAGAgent functional tests (9 test cases)
- **Added** ResilientExecutor, ExperimentState, ExperimentIO, PromptBuilder tests
- **Added** HybridSearcher functional tests, execution phase tests
- **Added** Provider ref-counting tests (3 test classes)
- **Consolidated** mock fixtures into `conftest.py`

#### Interface & Design (P3) — 14 issues fixed
- **Extracted** shared `apply_reranking()` from 3 agents (~90 lines deduplication)
- **Removed** stale `core/schemas.py`
- **Migrated** Pydantic v1 validators to v2 (`@field_validator`, `model_config`)
- **Converted** MetricFactory from if-elif chains to registry pattern
- **Renamed** provider-level `Embedder` → `ManagedEmbedder` (disambiguation)
- **Replaced** 8-level nested loops with `itertools.product` in spec builder

#### Documentation
- **Rewrote** `docs/FUTURE_WORK.md` — comprehensive roadmap with code review findings
- **Deprecated** `docs/IMPROVEMENT_PLAN.md` (superseded by BACKLOG + FUTURE_WORK)
- **Archived** `docs/BACKLOG.md` as completed historical record
- **Improved** LLM judge prompt injection defense (XML tags, sanitization, retry)

---

## [0.4.0] - 2026-01-30

### 🏗️ Architecture Refactoring

Major codebase restructuring for improved modularity and maintainability.

#### New Packages
- **Added** `spec/` package: Immutable experiment specifications
  - `ExperimentSpec` dataclass for experiment configuration
  - `build_specs()` for building specs from YAML configs
  - Naming conventions (`name_direct()`, `name_rag()`)
- **Added** `state/` package: Experiment state management
  - `ExperimentState` and `ExperimentPhase` (moved from `experiment_state.py`)
  - `ExperimentHealth` and `check_health()` for health checking
- **Added** `factory/` package: Specialized component factories
  - `ModelFactory`, `DatasetFactory`, `MetricFactory`, `RetrieverFactory`, `AgentFactory`
  - Replaces monolithic `factory.py` (632 lines → 5 focused modules)
- **Added** `execution/phases/` package: Pluggable phase handlers
  - `PhaseHandler` ABC and `ExecutionContext` dataclass
  - `InitHandler`, `GenerationHandler`, `MetricsHandler`
- **Added** `indexes/builders/` package: Split index builders
  - `build_embedding_index()`, `build_hierarchical_index()`

#### Code Cleanup
- **Deleted** ~1,200 lines of dead code:
  - `core/protocols.py` (281 lines) - unused protocols
  - `rag/chunking/semantic.py` (~200 lines) - unused chunker
  - `utils/prediction_writer.py` (238 lines) - unused utility
  - `metrics/ragas_adapter.py` (~150 lines) - unused adapter
  - `analysis/mlflow_tracker.py` (304 lines) - unused tracker

#### Improvements
- **Refactored** `Experiment` class to use phase handlers (Strategy pattern)
- **Improved** separation of concerns: specification ≠ state ≠ execution
- **Added** backward compatibility shims for smooth migration
- **Updated** all tests (124 passing)

### 📦 Breaking Changes (Internal Only)

Module paths changed (backward-compat shims provided):
- `ragicamp.experiment_state` → `ragicamp.state`
- `ragicamp.factory` (file) → `ragicamp.factory` (package)

---

## [0.3.0] - 2025-12-30

### 🎉 Major Features

#### Phased Experiment Execution
- **Added** `ExperimentPhase` enum: INIT → GENERATING → GENERATED → COMPUTING_METRICS → COMPLETE
- **Added** `ExperimentState` for persistent state tracking (phase, progress, timestamps)
- **Added** `ExperimentHealth` for health checks (missing predictions, missing metrics)
- **Added** Automatic checkpointing during generation with resume capability
- **Added** `state.json`, `questions.json` artifacts for better state management

#### New CLI Commands
- **Added** `ragicamp health <dir>` - Check experiment health status
- **Added** `ragicamp resume <dir>` - Resume incomplete experiments
- **Added** `ragicamp metrics <dir> -m f1,llm_judge` - Recompute specific metrics
- **Enhanced** `--dry-run` now shows health status of each experiment

#### OpenAI API Improvements
- **Fixed** `max_tokens` → `max_completion_tokens` for newer models (o1, o3, gpt-5)
- **Fixed** Skip `temperature`/`top_p` for models that don't support them
- **Added** Automatic detection of model capabilities

### ✨ Enhancements

#### Experiment Management
- **Added** Phase-based callbacks: `on_phase_start`, `on_phase_end`
- **Added** `check_health()` method on Experiment class
- **Added** `run_phase()` for running specific phases only
- **Enhanced** Study runner with health-aware execution
- **Added** Status tracking: complete, resumed, ran, failed, skipped

#### Analysis & Metrics
- **Added** Per-item metric scores in `predictions.json`
- **Added** Full prompt storage in predictions for debugging
- **Enhanced** Analysis notebook with metric-agnostic visualizations
- **Added** LLM-as-Judge async batching with rate limiting

### 🐛 Bug Fixes

- **Fixed** OpenAI API errors with newer models (o1, o3, gpt-5)
- **Fixed** Analysis notebook issues with pivot table returns
- **Fixed** Padding side for HuggingFace decoder-only models

### 🗑️ Removed

- **Deleted** `experiments/` folder (legacy configs and scripts)
- **Deleted** Orphaned Hydra config subdirectories (`conf/agent/`, `conf/model/`, etc.)
- **Deleted** Empty script folders (`scripts/eval/`, `scripts/examples/`, etc.)
- **Deleted** Duplicate `scripts/analysis/compare_baseline.py`
- **Deleted** `src/ragicamp/registry.py` (merged into ComponentFactory)
- **Deleted** `docs/archives/` (outdated documentation)

### 📚 Documentation

- **Updated** README.md with phased execution and new CLI commands
- **Updated** AGENTS.md with current architecture
- **Updated** ARCHITECTURE.md to reflect actual components
- **Updated** CHEATSHEET.md with current workflows
- **Updated** GETTING_STARTED.md with modern examples
- **Cleaned** Removed references to non-existent features (BanditRAG, MDPRAG, Policies, etc.)

---

## [0.2.0] - 2025-11-18

### 🎉 Major Features

#### Unified Experiment API
- **Added** Single `Experiment` class for all evaluations
- **Added** `ExperimentCallbacks` for monitoring hooks
- **Added** `ExperimentResult` dataclass
- **Added** `ComponentFactory` for component creation from configs

#### Study Configuration
- **Added** YAML-based study configs (`conf/study/`)
- **Added** Study runner (`scripts/experiments/run_study.py`)
- **Added** Support for direct and RAG experiments in same study

#### Analysis Tools
- **Added** `ragicamp.analysis` module (ResultsLoader, comparison, visualization)
- **Added** `ragicamp compare` CLI command
- **Added** MLflow integration for experiment tracking
- **Added** Analysis notebook (`notebooks/experiment_analysis.ipynb`)

### ✨ Enhancements

- **Added** Batch processing for model generation
- **Added** Plugin registration with `@ComponentFactory.register_model()`
- **Added** ResourceManager for GPU memory cleanup
- **Added** Stop sequences in HuggingFace model

### 📚 Documentation

- **Added** AGENTS.md (development guidelines)
- **Added** CHEATSHEET.md (quick reference)
- **Added** Comprehensive docstrings

---

## [0.1.0] - 2025-10-01

### Initial Release

- DirectLLMAgent and FixedRAGAgent
- HuggingFace and OpenAI model support
- Dense retriever with FAISS
- NQ, TriviaQA, HotpotQA datasets
- F1, Exact Match, BERTScore, BLEURT, LLM-as-judge metrics
- Basic evaluation scripts

---

## Links

- **Documentation**: See `docs/README.md`
- **Quick Reference**: See `CHEATSHEET.md`
- **Architecture**: See `docs/ARCHITECTURE.md`
