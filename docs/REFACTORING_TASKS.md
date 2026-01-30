# RAGiCamp Architecture Refactoring - Task List

**Source**: ARCHITECTURE_REFACTORING.md v1.0  
**Created**: January 2026  
**Status**: ✅ COMPLETED (January 2026)

> **Note**: This document is now archived. All tasks have been completed.
> See [ARCHITECTURE_REFACTORING.md](./ARCHITECTURE_REFACTORING.md) for the current status.

---

## Overview

This document transforms the architecture refactoring specification into an ordered list of actionable development tasks. Tasks are organized by phase, with complexity ratings, dependencies, and validation steps.

### Complexity Legend

| Symbol | Complexity | Typical Effort | Risk Level |
|--------|------------|----------------|------------|
| 🟢 | Low | < 1 hour | Minimal risk, isolated changes |
| 🟡 | Medium | 1-4 hours | Moderate risk, some dependencies |
| 🔴 | High | 4+ hours | Higher risk, cross-cutting changes |

### Task Status Legend

- [ ] Not started
- [~] In progress
- [x] Completed
- [!] Blocked

---

## Phase 0: Preparation & Foundation

**Goal**: Establish baseline, ensure safety nets before making changes.

### 0.1 Baseline & Documentation

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 0.1.1 | Document current test coverage for affected modules | 🟢 | 30 min |
| 0.1.2 | Create a git branch for refactoring (`refactor/architecture-v2`) | 🟢 | 5 min |
| 0.1.3 | Run full test suite and record baseline results | 🟢 | 15 min |
| 0.1.4 | Document current import graph for affected modules | 🟡 | 1 hour |

#### Tasks

- [ ] **0.1.1** Run coverage report: `uv run pytest --cov=ragicamp --cov-report=html`
- [ ] **0.1.2** Create feature branch: `git checkout -b refactor/architecture-v2`
- [ ] **0.1.3** Run tests and save output: `uv run pytest tests/ > baseline_test_results.txt 2>&1`
- [ ] **0.1.4** Generate import graph using `pydeps` or manual inspection of key files

### 0.2 Add Missing Tests (Critical for Safe Refactoring)

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 0.2.1 | Add tests for `factory.py` core functions | 🟡 | 2-3 hours |
| 0.2.2 | Add tests for `execution/runner.py` spec building | 🟡 | 2-3 hours |
| 0.2.3 | Add tests for `experiment.py` phase orchestration | 🔴 | 3-4 hours |
| 0.2.4 | Add tests for `indexes/builder.py` orchestration | 🟡 | 2 hours |

#### Tasks

- [ ] **0.2.1** Create `tests/test_factory.py` with tests for:
  - [ ] `create_model()` with different specs
  - [ ] `create_dataset()` with different configs
  - [ ] `create_metrics()` list creation
  - [ ] `load_retriever()` loading logic

- [ ] **0.2.2** Create/extend `tests/execution/test_runner.py` with tests for:
  - [ ] `build_specs()` from config dict
  - [ ] `_name_direct()` naming convention
  - [ ] `_name_rag()` naming convention

- [ ] **0.2.3** Extend `tests/test_experiment.py` with tests for:
  - [ ] Phase transitions
  - [ ] State persistence
  - [ ] Resume capability
  
- [ ] **0.2.4** Create `tests/indexes/test_builder.py` with tests for:
  - [ ] `ensure_indexes_exist()` logic
  - [ ] Index detection

#### Validation Checkpoint 0.2

```bash
# All tests must pass before proceeding
uv run pytest tests/ -v
# Coverage should be at acceptable level for refactored modules
uv run pytest --cov=ragicamp.factory --cov=ragicamp.execution.runner --cov-report=term-missing
```

---

## Phase 1: Dead Code Removal (Low Risk)

**Goal**: Remove ~1,200 lines of unused code to reduce maintenance burden.

**Prerequisites**: Phase 0 completed, all tests passing.

### 1.1 Identify and Verify Dead Code

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 1.1.1 | Verify `core/protocols.py` is truly unused | 🟢 | 15 min |
| 1.1.2 | Verify `rag/chunking/semantic.py` is truly unused | 🟢 | 15 min |
| 1.1.3 | Verify `utils/prediction_writer.py` is truly unused | 🟢 | 15 min |
| 1.1.4 | Verify `metrics/ragas_adapter.py` is truly unused | 🟢 | 15 min |
| 1.1.5 | Verify `analysis/mlflow_tracker.py` is truly unused | 🟢 | 15 min |

#### Tasks

- [ ] **1.1.1** Search for usages: `rg "from.*protocols import|import.*protocols" src/`
- [ ] **1.1.2** Search for usages: `rg "SemanticChunker|semantic.py" src/`
- [ ] **1.1.3** Search for usages: `rg "PredictionWriter|prediction_writer" src/`
- [ ] **1.1.4** Search for usages: `rg "ragas_adapter|RagasAdapter" src/`
- [ ] **1.1.5** Search for usages: `rg "mlflow_tracker|MLflowTracker" src/`

### 1.2 Remove Dead Code Files

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 1.2.1 | Delete `core/protocols.py` | 🟢 | 5 min |
| 1.2.2 | Delete `rag/chunking/semantic.py` | 🟢 | 5 min |
| 1.2.3 | Delete `utils/prediction_writer.py` | 🟢 | 5 min |
| 1.2.4 | Delete `metrics/ragas_adapter.py` | 🟢 | 5 min |
| 1.2.5 | Delete `analysis/mlflow_tracker.py` | 🟢 | 5 min |

#### Tasks

- [ ] **1.2.1** `rm src/ragicamp/core/protocols.py`
- [ ] **1.2.2** `rm src/ragicamp/rag/chunking/semantic.py`
- [ ] **1.2.3** `rm src/ragicamp/utils/prediction_writer.py`
- [ ] **1.2.4** `rm src/ragicamp/metrics/ragas_adapter.py`
- [ ] **1.2.5** `rm src/ragicamp/analysis/mlflow_tracker.py`

### 1.3 Clean Up Exports and Imports

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 1.3.1 | Update `core/__init__.py` to remove deleted exports | 🟢 | 10 min |
| 1.3.2 | Update `rag/chunking/__init__.py` to remove deleted exports | 🟢 | 10 min |
| 1.3.3 | Update `utils/__init__.py` to remove deleted exports | 🟢 | 10 min |
| 1.3.4 | Update `metrics/__init__.py` to remove deleted exports | 🟢 | 10 min |
| 1.3.5 | Update `analysis/__init__.py` to remove deleted exports | 🟢 | 10 min |

#### Tasks

- [ ] **1.3.1** Edit `src/ragicamp/core/__init__.py` - remove protocol imports
- [ ] **1.3.2** Edit `src/ragicamp/rag/chunking/__init__.py` - remove semantic imports
- [ ] **1.3.3** Edit `src/ragicamp/utils/__init__.py` - remove prediction_writer imports
- [ ] **1.3.4** Edit `src/ragicamp/metrics/__init__.py` - remove ragas_adapter imports
- [ ] **1.3.5** Edit `src/ragicamp/analysis/__init__.py` - remove mlflow_tracker imports

#### Validation Checkpoint 1

```bash
# Verify no broken imports
uv run python -c "import ragicamp"

# Run full test suite
uv run pytest tests/ -v

# Verify the application still runs
uv run ragicamp --help
```

### 1.4 Commit Phase 1

- [ ] **1.4.1** Stage and commit changes with descriptive message
  ```bash
  git add -A
  git commit -m "refactor: remove ~1,200 lines of dead code

  - Delete unused core/protocols.py (281 lines)
  - Delete unused rag/chunking/semantic.py (~200 lines)  
  - Delete unused utils/prediction_writer.py (238 lines)
  - Delete unused metrics/ragas_adapter.py (~150 lines)
  - Delete unused analysis/mlflow_tracker.py (304 lines)
  - Clean up related exports in __init__.py files"
  ```

---

## Phase 2: Extract Specifications Package (Medium Risk)

**Goal**: Create `spec/` package to separate configuration from runtime concerns.

**Prerequisites**: Phase 1 completed, all tests passing.

### 2.1 Create spec/ Package Structure

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 2.1.1 | Create `spec/` directory and `__init__.py` | 🟢 | 5 min |
| 2.1.2 | Create `spec/experiment.py` with `ExperimentSpec` dataclass | 🟡 | 45 min |
| 2.1.3 | Create `spec/naming.py` with naming functions | 🟢 | 30 min |
| 2.1.4 | Create `spec/builder.py` with `build_specs()` | 🟡 | 1 hour |

#### Tasks

- [ ] **2.1.1** Create package structure:
  ```bash
  mkdir -p src/ragicamp/spec
  touch src/ragicamp/spec/__init__.py
  ```

- [ ] **2.1.2** Create `spec/experiment.py`:
  - [ ] Define `ExperimentSpec` as frozen dataclass
  - [ ] Include all fields: name, exp_type, model, dataset, prompt, quant, retriever, top_k, query_transform, reranker, batch_size, min_batch_size, metrics
  - [ ] Add validation in `__post_init__` if needed
  - [ ] Add docstrings

- [ ] **2.1.3** Extract from `runner.py` to `spec/naming.py`:
  - [ ] Move `_name_direct()` → `name_direct()`
  - [ ] Move `_name_rag()` → `name_rag()`
  - [ ] Add docstrings and type hints

- [ ] **2.1.4** Extract from `runner.py` to `spec/builder.py`:
  - [ ] Move `build_specs()` function
  - [ ] Update to return `List[ExperimentSpec]`
  - [ ] Update internal calls to use new naming module

### 2.2 Create state/ Package Structure

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 2.2.1 | Create `state/` directory and `__init__.py` | 🟢 | 5 min |
| 2.2.2 | Move `experiment_state.py` to `state/experiment_state.py` | 🟢 | 15 min |
| 2.2.3 | Extract `ExperimentHealth` and `check_health()` to `state/health.py` | 🟡 | 45 min |
| 2.2.4 | Create `state/artifacts.py` for path management | 🟡 | 30 min |

#### Tasks

- [ ] **2.2.1** Create package structure:
  ```bash
  mkdir -p src/ragicamp/state
  touch src/ragicamp/state/__init__.py
  ```

- [ ] **2.2.2** Move state module:
  - [ ] `mv src/ragicamp/experiment_state.py src/ragicamp/state/experiment_state.py`
  - [ ] Update imports in the moved file
  - [ ] Create re-export in `src/ragicamp/experiment_state.py` (temporary backward compat)

- [ ] **2.2.3** Create `state/health.py`:
  - [ ] Extract `ExperimentHealth` dataclass from `experiment.py`
  - [ ] Extract `check_health()` function
  - [ ] Add proper type hints and docstrings

- [ ] **2.2.4** Create `state/artifacts.py`:
  - [ ] Extract path management properties from `Experiment` class
  - [ ] Create `ArtifactPaths` dataclass or functions

### 2.3 Update Imports Across Codebase

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 2.3.1 | Update imports in `experiment.py` | 🟡 | 30 min |
| 2.3.2 | Update imports in `execution/runner.py` | 🟡 | 30 min |
| 2.3.3 | Update imports in test files | 🟡 | 30 min |
| 2.3.4 | Update imports in any other affected files | 🟡 | 30 min |

#### Tasks

- [ ] **2.3.1** Update `experiment.py`:
  - [ ] Import `ExperimentSpec` from `spec.experiment`
  - [ ] Import state types from `state.experiment_state`
  - [ ] Import health checking from `state.health`

- [ ] **2.3.2** Update `execution/runner.py`:
  - [ ] Remove local spec building code
  - [ ] Import from `spec.builder`
  - [ ] Import from `spec.naming`

- [ ] **2.3.3** Update test files to use new import paths

- [ ] **2.3.4** Search and update remaining imports:
  ```bash
  rg "from ragicamp import.*ExpSpec" src/
  rg "from ragicamp.experiment_state" src/
  ```

### 2.4 Add Tests for New Packages

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 2.4.1 | Create `tests/spec/test_experiment.py` | 🟢 | 30 min |
| 2.4.2 | Create `tests/spec/test_naming.py` | 🟢 | 20 min |
| 2.4.3 | Create `tests/spec/test_builder.py` | 🟡 | 45 min |
| 2.4.4 | Create `tests/state/test_health.py` | 🟢 | 30 min |

#### Tasks

- [ ] **2.4.1** Test `ExperimentSpec`:
  - [ ] Test immutability (frozen dataclass)
  - [ ] Test default values
  - [ ] Test field validation if any

- [ ] **2.4.2** Test naming functions:
  - [ ] Test `name_direct()` format
  - [ ] Test `name_rag()` format
  - [ ] Test edge cases (special characters, etc.)

- [ ] **2.4.3** Test `build_specs()`:
  - [ ] Test with direct experiment config
  - [ ] Test with RAG experiment config
  - [ ] Test with mixed config
  - [ ] Test error handling

- [ ] **2.4.4** Test `check_health()`:
  - [ ] Test healthy state detection
  - [ ] Test missing work detection
  - [ ] Test resume phase calculation

#### Validation Checkpoint 2

```bash
# Run new tests
uv run pytest tests/spec/ tests/state/ -v

# Run full test suite
uv run pytest tests/ -v

# Test experiment execution still works
uv run ragicamp run conf/study/rag_strategies_test.yaml --dry-run

# Test import paths work
uv run python -c "from ragicamp.spec import ExperimentSpec, build_specs"
uv run python -c "from ragicamp.state import ExperimentState, check_health"
```

### 2.5 Commit Phase 2

- [ ] **2.5.1** Commit spec/ package
- [ ] **2.5.2** Commit state/ package

---

## Phase 3: Split Factory Package (Medium Risk)

**Goal**: Break monolithic `factory.py` (632 lines) into focused factory modules.

**Prerequisites**: Phase 2 completed, all tests passing.

### 3.1 Create factory/ Package Structure

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 3.1.1 | Create `factory/` directory and `__init__.py` | 🟢 | 5 min |
| 3.1.2 | Create `factory/models.py` with `ModelFactory` | 🟡 | 1 hour |
| 3.1.3 | Create `factory/datasets.py` with `DatasetFactory` | 🟡 | 45 min |
| 3.1.4 | Create `factory/metrics.py` with `MetricFactory` | 🟡 | 45 min |
| 3.1.5 | Create `factory/retrievers.py` with `RetrieverFactory` | 🟡 | 1 hour |
| 3.1.6 | Create `factory/agents.py` with `AgentFactory` | 🟡 | 45 min |

#### Tasks

- [ ] **3.1.1** Create package structure:
  ```bash
  mkdir -p src/ragicamp/factory
  touch src/ragicamp/factory/__init__.py
  ```

- [ ] **3.1.2** Create `factory/models.py`:
  - [ ] Extract model creation from `factory.py`
  - [ ] Define `ModelFactory` class with static methods
  - [ ] Include `parse_model_spec()` if coupled
  - [ ] Add type hints and docstrings

- [ ] **3.1.3** Create `factory/datasets.py`:
  - [ ] Extract dataset creation from `factory.py`
  - [ ] Define `DatasetFactory` class
  - [ ] Include `parse_dataset_spec()` if coupled

- [ ] **3.1.4** Create `factory/metrics.py`:
  - [ ] Extract metric creation from `factory.py`
  - [ ] Define `MetricFactory` class

- [ ] **3.1.5** Create `factory/retrievers.py`:
  - [ ] Extract retriever loading from `factory.py`
  - [ ] Extract query transformer creation
  - [ ] Extract reranker creation
  - [ ] Define `RetrieverFactory` class

- [ ] **3.1.6** Create `factory/agents.py`:
  - [ ] Extract agent creation from `factory.py`
  - [ ] Define `AgentFactory` class

### 3.2 Create Unified Exports

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 3.2.1 | Update `factory/__init__.py` with all exports | 🟢 | 15 min |
| 3.2.2 | Create backward-compat re-exports in old `factory.py` | 🟢 | 15 min |

#### Tasks

- [ ] **3.2.1** Update `factory/__init__.py`:
  ```python
  from .models import ModelFactory
  from .datasets import DatasetFactory
  from .metrics import MetricFactory
  from .retrievers import RetrieverFactory
  from .agents import AgentFactory

  __all__ = [
      "ModelFactory",
      "DatasetFactory",
      "MetricFactory",
      "RetrieverFactory",
      "AgentFactory",
  ]
  ```

- [ ] **3.2.2** Update old `factory.py` to re-export (temporary):
  ```python
  # Deprecated: import from ragicamp.factory instead
  from ragicamp.factory import *
  ```

### 3.3 Update All Import Sites

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 3.3.1 | Update imports in `experiment.py` | 🟡 | 20 min |
| 3.3.2 | Update imports in `execution/runner.py` | 🟡 | 20 min |
| 3.3.3 | Update imports in test files | 🟡 | 30 min |
| 3.3.4 | Search and update all remaining imports | 🟡 | 30 min |

#### Tasks

- [ ] **3.3.1-4** Find and update all usages:
  ```bash
  rg "from ragicamp.factory import|from ragicamp import.*Factory" src/ tests/
  ```

### 3.4 Delete Old factory.py

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 3.4.1 | Verify all imports updated | 🟢 | 15 min |
| 3.4.2 | Delete `factory.py` | 🟢 | 5 min |

#### Tasks

- [ ] **3.4.1** Verify no direct imports remain:
  ```bash
  rg "from ragicamp.factory import" src/ --files-with-matches
  # Should show only factory/__init__.py
  ```
- [ ] **3.4.2** `rm src/ragicamp/factory.py`

### 3.5 Add Tests for Factory Modules

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 3.5.1 | Create `tests/factory/test_models.py` | 🟡 | 30 min |
| 3.5.2 | Create `tests/factory/test_datasets.py` | 🟡 | 30 min |
| 3.5.3 | Create `tests/factory/test_metrics.py` | 🟢 | 20 min |
| 3.5.4 | Create `tests/factory/test_retrievers.py` | 🟡 | 30 min |
| 3.5.5 | Create `tests/factory/test_agents.py` | 🟡 | 30 min |

#### Validation Checkpoint 3

```bash
# Run factory tests
uv run pytest tests/factory/ -v

# Run full test suite
uv run pytest tests/ -v

# Verify factories work end-to-end
uv run python -c "
from ragicamp.factory import ModelFactory, DatasetFactory
# Quick smoke test
print('Factory imports work!')
"
```

### 3.6 Commit Phase 3

- [ ] **3.6.1** Commit factory package split

---

## Phase 4: Extract Phase Handlers (Higher Risk)

**Goal**: Create modular phase handlers to replace monolithic orchestration in `experiment.py`.

**Prerequisites**: Phase 3 completed, all tests passing.

### 4.1 Create Phase Handler Infrastructure

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 4.1.1 | Create `execution/phases/` directory | 🟢 | 5 min |
| 4.1.2 | Create `execution/phases/base.py` with ABC | 🟡 | 45 min |
| 4.1.3 | Create `ExecutionContext` dataclass | 🟡 | 30 min |

#### Tasks

- [ ] **4.1.1** Create directory structure:
  ```bash
  mkdir -p src/ragicamp/execution/phases
  touch src/ragicamp/execution/phases/__init__.py
  ```

- [ ] **4.1.2** Create `base.py`:
  - [ ] Define `PhaseHandler` abstract base class
  - [ ] Define `can_handle(phase: ExperimentPhase) -> bool`
  - [ ] Define `execute(spec, state, context) -> ExperimentState`
  - [ ] Add comprehensive docstrings

- [ ] **4.1.3** Create `ExecutionContext`:
  - [ ] Define as dataclass
  - [ ] Include: output_path, agent, dataset, metrics, callbacks
  - [ ] Add helper methods if needed

### 4.2 Implement Concrete Phase Handlers

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 4.2.1 | Create `execution/phases/init.py` with `InitHandler` | 🟡 | 1.5 hours |
| 4.2.2 | Create `execution/phases/generation.py` with `GenerationHandler` | 🔴 | 2-3 hours |
| 4.2.3 | Create `execution/phases/metrics.py` with `MetricsHandler` | 🟡 | 1.5 hours |

#### Tasks

- [ ] **4.2.1** Create `InitHandler`:
  - [ ] Extract `_phase_init` logic from `experiment.py`
  - [ ] Implement `can_handle()` for INIT phase
  - [ ] Implement `execute()` to export questions and save metadata
  - [ ] Return updated state

- [ ] **4.2.2** Create `GenerationHandler`:
  - [ ] Extract `_phase_generate` logic from `experiment.py`
  - [ ] Inject `ResilientExecutor` dependency
  - [ ] Implement checkpointing logic
  - [ ] Handle batch processing
  - [ ] Return updated state with predictions

- [ ] **4.2.3** Create `MetricsHandler`:
  - [ ] Extract `_phase_compute_metrics` logic from `experiment.py`
  - [ ] Use `compute_metrics_batched` 
  - [ ] Return updated state with metrics

### 4.3 Refactor ExperimentRunner

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 4.3.1 | Slim down `execution/runner.py` to use handlers | 🔴 | 2-3 hours |
| 4.3.2 | Update handler registration/discovery | 🟡 | 1 hour |
| 4.3.3 | Extract subprocess logic to `execution/subprocess.py` | 🟡 | 1 hour |

#### Tasks

- [ ] **4.3.1** Refactor `ExperimentRunner`:
  - [ ] Accept `List[PhaseHandler]` in constructor
  - [ ] Accept `ResourceManager` for GPU/memory lifecycle
  - [ ] Implement `run()` that delegates to handlers
  - [ ] Implement phase state transitions
  - [ ] Handle state persistence between phases

- [ ] **4.3.2** Update handler registration:
  - [ ] Create factory function to build default handlers
  - [ ] Allow custom handler injection

- [ ] **4.3.3** Extract subprocess:
  - [ ] Move `run_spec_subprocess()` to `subprocess.py`
  - [ ] Keep it focused on subprocess spawning logic

### 4.4 Update Experiment Facade

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 4.4.1 | Slim down `experiment.py` to facade pattern | 🔴 | 2 hours |
| 4.4.2 | Delegate to `ExperimentRunner` | 🟡 | 1 hour |

#### Tasks

- [ ] **4.4.1** Refactor `experiment.py`:
  - [ ] Keep `Experiment` class as simple facade
  - [ ] Keep `ExperimentResult` and `ExperimentCallbacks` (data contracts)
  - [ ] Remove phase implementations (now in handlers)
  - [ ] Remove duplicated orchestration logic

- [ ] **4.4.2** Delegate to runner:
  - [ ] `Experiment.run()` creates runner and delegates
  - [ ] Maintain backward-compatible API

### 4.5 Add Tests for Phase Handlers

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 4.5.1 | Create `tests/execution/phases/test_init.py` | 🟡 | 45 min |
| 4.5.2 | Create `tests/execution/phases/test_generation.py` | 🔴 | 1.5 hours |
| 4.5.3 | Create `tests/execution/phases/test_metrics.py` | 🟡 | 45 min |
| 4.5.4 | Create integration test for full phase flow | 🔴 | 2 hours |

#### Tasks

- [ ] **4.5.1** Test `InitHandler`:
  - [ ] Test question export
  - [ ] Test metadata saving
  - [ ] Test state update

- [ ] **4.5.2** Test `GenerationHandler`:
  - [ ] Test batch processing
  - [ ] Test checkpointing
  - [ ] Test resume from partial
  - [ ] Test error handling

- [ ] **4.5.3** Test `MetricsHandler`:
  - [ ] Test metric computation
  - [ ] Test state update

- [ ] **4.5.4** Create integration test:
  - [ ] Test INIT → GENERATION → METRICS flow
  - [ ] Test resume at each phase
  - [ ] Test with real (but small) dataset

#### Validation Checkpoint 4

```bash
# Run phase handler tests
uv run pytest tests/execution/phases/ -v

# Run full test suite
uv run pytest tests/ -v

# Test end-to-end experiment
uv run ragicamp run conf/study/rag_strategies_test.yaml

# Verify results compatibility
# Check that existing outputs can still be loaded
uv run python -c "
from ragicamp.state import ExperimentState
import json
# Load existing state file and verify
"
```

### 4.6 Commit Phase 4

- [ ] **4.6.1** Commit phase handlers implementation
- [ ] **4.6.2** Commit runner refactoring
- [ ] **4.6.3** Commit experiment facade slimdown

---

## Phase 5: Split Index Builders (Medium Risk)

**Goal**: Split `indexes/builder.py` (630 lines) into focused builder modules.

**Prerequisites**: Phase 4 completed, all tests passing.

### 5.1 Create Builders Package

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 5.1.1 | Create `indexes/builders/` directory | 🟢 | 5 min |
| 5.1.2 | Create `builders/embedding_builder.py` | 🟡 | 1.5 hours |
| 5.1.3 | Create `builders/hierarchical_builder.py` | 🟡 | 1.5 hours |

#### Tasks

- [ ] **5.1.1** Create directory:
  ```bash
  mkdir -p src/ragicamp/indexes/builders
  touch src/ragicamp/indexes/builders/__init__.py
  ```

- [ ] **5.1.2** Create `embedding_builder.py`:
  - [ ] Extract embedding index building logic
  - [ ] Define `build_embedding_index()` function
  - [ ] Include batch processing logic
  - [ ] Add progress reporting

- [ ] **5.1.3** Create `hierarchical_builder.py`:
  - [ ] Extract hierarchical index building logic
  - [ ] Define `build_hierarchical_index()` function
  - [ ] Include batch processing logic

### 5.2 Slim Down Original Builder

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 5.2.1 | Refactor `indexes/builder.py` to orchestration only | 🟡 | 1 hour |
| 5.2.2 | Import from new builder modules | 🟢 | 15 min |

#### Tasks

- [ ] **5.2.1** Refactor `builder.py`:
  - [ ] Keep `ensure_indexes_exist()` as main entry point
  - [ ] Keep orchestration logic (which indexes to build)
  - [ ] Remove implementation details (now in builders/)

- [ ] **5.2.2** Update imports:
  ```python
  from .builders.embedding_builder import build_embedding_index
  from .builders.hierarchical_builder import build_hierarchical_index
  ```

### 5.3 Add Tests for Builders

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 5.3.1 | Create `tests/indexes/builders/test_embedding_builder.py` | 🟡 | 45 min |
| 5.3.2 | Create `tests/indexes/builders/test_hierarchical_builder.py` | 🟡 | 45 min |

#### Validation Checkpoint 5

```bash
# Run builder tests
uv run pytest tests/indexes/ -v

# Run full test suite  
uv run pytest tests/ -v

# Test index building end-to-end
uv run python -c "
from ragicamp.indexes.builder import ensure_indexes_exist
print('Index builder imports work!')
"
```

### 5.4 Commit Phase 5

- [ ] **5.4.1** Commit index builder split

---

## Phase 6: Final Validation & Cleanup

**Goal**: Ensure all success criteria are met and clean up temporary code.

### 6.1 Success Criteria Verification

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 6.1.1 | Verify quantitative metrics | 🟢 | 30 min |
| 6.1.2 | Verify qualitative criteria | 🟡 | 1 hour |
| 6.1.3 | Verify results compatibility | 🟡 | 1 hour |

#### Tasks

- [ ] **6.1.1** Check quantitative metrics:
  ```bash
  # Largest file should be < 300 lines
  wc -l src/ragicamp/**/*.py | sort -n | tail -10
  
  # Files > 400 lines should be 0
  find src/ragicamp -name "*.py" -exec wc -l {} \; | awk '$1 > 400'
  
  # Dead code should be 0 (files deleted)
  ls src/ragicamp/core/protocols.py 2>/dev/null && echo "FAIL: Dead code exists"
  
  # Test coverage for new modules > 80%
  uv run pytest --cov=ragicamp.spec --cov=ragicamp.state --cov=ragicamp.factory --cov-report=term-missing
  ```

- [ ] **6.1.2** Check qualitative criteria:
  - [ ] Each class has single responsibility (code review)
  - [ ] Dependencies are explicit (injected)
  - [ ] Adding new phase handler requires no runner changes
  - [ ] Adding new factory requires no other factory changes
  - [ ] Experiment can be configured without implementation knowledge

- [ ] **6.1.3** Check results compatibility:
  - [ ] Load existing `predictions.json` files
  - [ ] Load existing `results.json` files
  - [ ] Load existing `state.json` files
  - [ ] Run analysis notebooks
  - [ ] Test experiment resume

### 6.2 Remove Temporary Backward Compatibility

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 6.2.1 | Remove re-export shims from old locations | 🟢 | 30 min |
| 6.2.2 | Update any remaining old imports | 🟢 | 30 min |

#### Tasks

- [ ] **6.2.1** Remove shims:
  - [ ] Delete old `factory.py` re-export shim if exists
  - [ ] Delete old `experiment_state.py` re-export shim if exists

- [ ] **6.2.2** Final import cleanup:
  ```bash
  rg "Deprecated|deprecated|compat" src/ragicamp/
  ```

### 6.3 Documentation Update

| ID | Task | Complexity | Est. Time |
|----|------|------------|-----------|
| 6.3.1 | Update module docstrings | 🟢 | 30 min |
| 6.3.2 | Update ARCHITECTURE_REFACTORING.md status | 🟢 | 15 min |
| 6.3.3 | Add migration notes to CHANGELOG | 🟢 | 15 min |

#### Tasks

- [ ] **6.3.1** Ensure all new modules have proper docstrings
- [ ] **6.3.2** Mark tasks as complete in architecture doc
- [ ] **6.3.3** Document breaking changes in CHANGELOG

### 6.4 Final Commit & Merge

- [ ] **6.4.1** Run final full test suite
- [ ] **6.4.2** Create comprehensive commit message
- [ ] **6.4.3** Create PR for code review
- [ ] **6.4.4** Address review feedback
- [ ] **6.4.5** Merge to main branch

---

## Summary: Task Count by Phase

| Phase | Total Tasks | 🟢 Low | 🟡 Medium | 🔴 High |
|-------|-------------|--------|-----------|---------|
| Phase 0: Preparation | 8 | 4 | 4 | 0 |
| Phase 1: Dead Code | 16 | 16 | 0 | 0 |
| Phase 2: Spec Package | 16 | 7 | 9 | 0 |
| Phase 3: Factory Package | 17 | 7 | 10 | 0 |
| Phase 4: Phase Handlers | 18 | 3 | 9 | 6 |
| Phase 5: Index Builders | 8 | 3 | 5 | 0 |
| Phase 6: Validation | 12 | 8 | 4 | 0 |
| **Total** | **95** | **48** | **41** | **6** |

---

## Quick Reference: Validation Commands

```bash
# After each phase, run these:

# 1. Import check
uv run python -c "import ragicamp"

# 2. Full test suite
uv run pytest tests/ -v

# 3. Application smoke test
uv run ragicamp --help

# 4. Experiment execution test (if infrastructure allows)
uv run ragicamp run conf/study/rag_strategies_test.yaml --dry-run
```

---

## Risk Mitigation Strategies

1. **Always commit after each sub-phase**: Small, atomic commits make rollback easier
2. **Run tests after every change**: Catch regressions immediately
3. **Keep old code temporarily**: Use re-export shims during transition
4. **One package at a time**: Complete spec/ before starting factory/
5. **Phase 4 is highest risk**: Consider extra code review before merging

---

## Dependencies Between Phases

```
Phase 0 (Preparation)
    │
    ▼
Phase 1 (Dead Code) ─────────────────┐
    │                                 │
    ▼                                 │
Phase 2 (Spec Package)                │
    │                                 │
    ▼                                 │
Phase 3 (Factory Package)             │ (All phases depend on Phase 0-1)
    │                                 │
    ▼                                 │
Phase 4 (Phase Handlers) ◄────────────┘
    │
    ▼
Phase 5 (Index Builders) (can run parallel to Phase 4)
    │
    ▼
Phase 6 (Validation & Cleanup)
```

Note: Phase 5 has no dependencies on Phase 4 and could theoretically run in parallel, but sequential execution reduces cognitive load and risk.
