# RAGiCamp Architecture Refactoring Specification

**Version**: 1.0  
**Date**: January 2026  
**Status**: Draft

---

## 1. Current State Analysis

### 1.1 Codebase Metrics

| Category | Files | Lines | Notes |
|----------|-------|-------|-------|
| Core execution | 4 | ~2,700 | experiment.py, runner.py, executor.py, experiment_state.py |
| Factory/Creation | 1 | 632 | factory.py (god class) |
| Index building | 1 | 630 | indexes/builder.py |
| Total source | ~80 | ~12,000 | Excluding tests |

### 1.2 Identified Issues

#### A. Monolithic Orchestrators
The four largest files have overlapping responsibilities:

```
experiment.py (747 lines)
├── ExperimentResult, ExperimentCallbacks (data contracts)
├── Experiment class
│   ├── Path management (5 properties)
│   ├── Health checking
│   ├── Phase orchestration
│   ├── Phase implementations (_phase_init, _phase_generate, etc.)
│   ├── Checkpointing
│   └── Model lifecycle
└── run_experiments() utility

execution/runner.py (650 lines)
├── ExpSpec (experiment specification)
├── Spec building (build_specs, _name_direct, _name_rag)
├── run_metrics_only() - duplicates Experiment._phase_compute_metrics
├── run_generation() - creates Experiment and calls run()
└── run_spec_subprocess() - subprocess spawning with retry

factory.py (632 lines)
├── ComponentFactory class with 5 creation methods
├── Spec parsers (parse_model_spec, parse_dataset_spec)
├── Plugin registries
└── Utility functions (load_retriever, create_query_transformer, etc.)
```

#### B. Blurred Boundaries
- `Experiment` knows too much: configuration, state, execution, persistence, resource management
- `runner.py` contains both spec building (config concern) and execution (runtime concern)
- `factory.py` mixes creation logic with spec parsing and utility functions

#### C. Dead Code (~720 lines)

| File | Lines | Status |
|------|-------|--------|
| `core/protocols.py` | 281 | Completely unused |
| `rag/chunking/semantic.py` | ~200 | Exported but never instantiated |
| `utils/prediction_writer.py` | 238 | Class never used |
| `metrics/ragas_adapter.py` | ~150 | Not in current study, optional |
| `analysis/mlflow_tracker.py` | 304 | Not actively used |

---

## 2. Target Architecture

### 2.1 Design Principles

1. **Single Responsibility**: Each class/module does one thing well
2. **Dependency Inversion**: Depend on abstractions (protocols), not implementations
3. **Separation of Concerns**: 
   - Specification ≠ State ≠ Execution
   - Configuration ≠ Runtime
4. **Explicit Dependencies**: No hidden coupling, inject what you need

### 2.2 Core Domain Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION                            │
├─────────────────────────────────────────────────────────────────┤
│  ExperimentSpec        │  What to run (immutable)               │
│  ├── name              │                                        │
│  ├── model_spec        │                                        │
│  ├── dataset_spec      │                                        │
│  ├── retriever_spec    │  (optional, for RAG)                   │
│  ├── metrics           │                                        │
│  └── options           │  batch_size, top_k, prompts, etc.      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                           STATE                                  │
├─────────────────────────────────────────────────────────────────┤
│  ExperimentState       │  Where we are (mutable, persistent)    │
│  ├── phase             │  INIT → GENERATING → METRICS → DONE   │
│  ├── progress          │  predictions_done, metrics_computed    │
│  └── artifacts         │  paths to outputs                      │
│                        │                                        │
│  ExperimentHealth      │  What's missing (computed from state)  │
│  ├── can_resume        │                                        │
│  ├── missing_work      │                                        │
│  └── resume_phase      │                                        │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         EXECUTION                                │
├─────────────────────────────────────────────────────────────────┤
│  ExperimentRunner      │  Orchestrates execution                │
│  ├── run(spec, state)  │  Main entry point                     │
│  ├── phase_handlers    │  Injected phase implementations        │
│  └── resource_manager  │  GPU/memory lifecycle                  │
│                        │                                        │
│  PhaseHandler (ABC)    │  One handler per phase                 │
│  ├── InitHandler       │  Export questions, save metadata       │
│  ├── GenerationHandler │  Run predictions with checkpointing    │
│  └── MetricsHandler    │  Compute all metrics                   │
│                        │                                        │
│  ResilientExecutor     │  Batch processing with error recovery  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FACTORIES                                 │
├─────────────────────────────────────────────────────────────────┤
│  ModelFactory          │  Create models from specs              │
│  DatasetFactory        │  Create datasets from specs            │
│  RetrieverFactory      │  Create/load retrievers from specs     │
│  AgentFactory          │  Create agents from components         │
│  MetricFactory         │  Create metrics from names             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 New Module Structure

```
ragicamp/
├── spec/                           # CONFIGURATION (immutable)
│   ├── __init__.py
│   ├── experiment.py               # ExperimentSpec dataclass
│   ├── builder.py                  # build_specs() from YAML config
│   └── naming.py                   # Experiment naming conventions
│
├── state/                          # STATE (mutable, persistent)
│   ├── __init__.py
│   ├── experiment_state.py         # ExperimentState, ExperimentPhase
│   ├── health.py                   # ExperimentHealth, check_health()
│   └── artifacts.py                # Artifact path management
│
├── execution/                      # EXECUTION (runtime)
│   ├── __init__.py
│   ├── runner.py                   # ExperimentRunner (slim orchestrator)
│   ├── phases/                     # Phase handler implementations
│   │   ├── __init__.py
│   │   ├── base.py                 # PhaseHandler ABC
│   │   ├── init.py                 # InitHandler
│   │   ├── generation.py           # GenerationHandler
│   │   └── metrics.py              # MetricsHandler
│   ├── executor.py                 # ResilientExecutor (unchanged)
│   └── subprocess.py               # Subprocess spawning logic
│
├── factory/                        # FACTORIES (creation)
│   ├── __init__.py
│   ├── models.py                   # ModelFactory
│   ├── datasets.py                 # DatasetFactory
│   ├── retrievers.py               # RetrieverFactory
│   ├── agents.py                   # AgentFactory
│   └── metrics.py                  # MetricFactory
│
├── indexes/                        # INDEX BUILDING
│   ├── __init__.py
│   ├── base.py                     # Index ABC
│   ├── embedding.py                # EmbeddingIndex
│   ├── hierarchical.py             # HierarchicalIndex
│   └── builders/                   # Split builder logic
│       ├── __init__.py
│       ├── embedding_builder.py
│       └── hierarchical_builder.py
│
├── experiment.py                   # FACADE: Experiment class (simple API)
│                                   # Just delegates to runner
│
└── [other packages unchanged]
```

---

## 3. Refactoring Tasks

### 3.1 Phase 1: Cleanup (Low Risk)

**Task 1.1: Remove dead code**
- [ ] Delete `core/protocols.py` (281 lines)
- [ ] Delete `rag/chunking/semantic.py` (~200 lines)
- [ ] Delete `utils/prediction_writer.py` (238 lines)
- [ ] Delete `metrics/ragas_adapter.py` (~150 lines) - Optional metrics, not in use
- [ ] Delete `analysis/mlflow_tracker.py` (304 lines) - MLflow not in active use

**Task 1.2: Fix imports and exports**
- [ ] Update `__init__.py` files to remove deleted exports
- [ ] Run tests to ensure nothing breaks

**Estimated impact**: ~1,200 lines removed

---

### 3.2 Phase 2: Extract Specifications (Medium Risk)

**Task 2.1: Create `spec/` package**
```python
# spec/experiment.py
@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable experiment configuration."""
    name: str
    exp_type: Literal["direct", "rag"]
    model: str
    dataset: str
    prompt: str
    quant: str = "4bit"
    retriever: Optional[str] = None
    top_k: int = 5
    query_transform: Optional[str] = None
    reranker: Optional[str] = None
    batch_size: int = 8
    min_batch_size: int = 1
    metrics: List[str] = field(default_factory=list)
```

**Task 2.2: Extract spec building from runner.py**
```python
# spec/builder.py
def build_specs(config: Dict[str, Any]) -> List[ExperimentSpec]:
    """Build experiment specs from YAML config."""
    ...

# spec/naming.py  
def name_direct(model: str, prompt: str, dataset: str, quant: str) -> str: ...
def name_rag(...) -> str: ...
```

**Task 2.3: Move state to `state/` package**
- Move `experiment_state.py` → `state/experiment_state.py`
- Extract `ExperimentHealth` and `check_health` → `state/health.py`

---

### 3.3 Phase 3: Split Factory (Medium Risk)

**Task 3.1: Create `factory/` package**
```python
# factory/models.py
class ModelFactory:
    @staticmethod
    def create(spec: str, quantization: str = "none") -> LanguageModel: ...
    
# factory/datasets.py
class DatasetFactory:
    @staticmethod
    def create(name: str, split: str = "validation", limit: Optional[int] = None) -> QADataset: ...

# factory/metrics.py
class MetricFactory:
    @staticmethod
    def create(names: List[str], judge_model: Optional[LanguageModel] = None) -> List[Metric]: ...

# factory/retrievers.py
class RetrieverFactory:
    @staticmethod
    def load(name: str) -> Retriever: ...
    @staticmethod
    def create_query_transformer(type: str, model: LanguageModel) -> Optional[QueryTransformer]: ...
    @staticmethod
    def create_reranker(model: str) -> Optional[Reranker]: ...
```

**Task 3.2: Update all imports**
```python
# factory/__init__.py
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

**Task 3.3: Delete old factory.py and update all imports across codebase**

---

### 3.4 Phase 4: Extract Phase Handlers (Higher Risk)

**Task 4.1: Create phase handler abstraction**
```python
# execution/phases/base.py
from abc import ABC, abstractmethod

class PhaseHandler(ABC):
    """Abstract handler for a single experiment phase."""
    
    @abstractmethod
    def can_handle(self, phase: ExperimentPhase) -> bool:
        """Check if this handler can process the given phase."""
        ...
    
    @abstractmethod
    def execute(
        self, 
        spec: ExperimentSpec, 
        state: ExperimentState,
        context: ExecutionContext,
    ) -> ExperimentState:
        """Execute the phase, returning updated state."""
        ...

@dataclass
class ExecutionContext:
    """Runtime context shared across phases."""
    output_path: Path
    agent: Optional[RAGAgent] = None
    dataset: Optional[QADataset] = None
    metrics: Optional[List[Metric]] = None
    callbacks: Optional[ExperimentCallbacks] = None
```

**Task 4.2: Implement concrete handlers**
```python
# execution/phases/init.py
class InitHandler(PhaseHandler):
    """Export questions and metadata."""
    
    def can_handle(self, phase: ExperimentPhase) -> bool:
        return phase == ExperimentPhase.INIT
    
    def execute(self, spec, state, context) -> ExperimentState:
        # Export questions
        # Save metadata
        # Return updated state
        ...

# execution/phases/generation.py
class GenerationHandler(PhaseHandler):
    """Generate predictions with checkpointing."""
    
    def __init__(self, executor: ResilientExecutor):
        self.executor = executor
    
    def execute(self, spec, state, context) -> ExperimentState:
        # Use executor to generate predictions
        # Handle checkpointing
        # Return updated state
        ...

# execution/phases/metrics.py
class MetricsHandler(PhaseHandler):
    """Compute all requested metrics."""
    
    def execute(self, spec, state, context) -> ExperimentState:
        # Use compute_metrics_batched
        # Return updated state
        ...
```

**Task 4.3: Slim down ExperimentRunner**
```python
# execution/runner.py
class ExperimentRunner:
    """Orchestrates experiment execution through phases."""
    
    def __init__(
        self,
        handlers: List[PhaseHandler],
        resource_manager: ResourceManager,
    ):
        self.handlers = handlers
        self.resource_manager = resource_manager
    
    def run(
        self,
        spec: ExperimentSpec,
        state: Optional[ExperimentState] = None,
        resume: bool = True,
    ) -> ExperimentResult:
        """Run experiment, delegating to appropriate phase handlers."""
        state = state or self._detect_or_create_state(spec)
        context = self._create_context(spec, state)
        
        for phase in self._phases_to_run(state, resume):
            handler = self._get_handler(phase)
            state = handler.execute(spec, state, context)
            state.save(context.output_path / "state.json")
        
        return self._create_result(spec, state, context)
```

---

### 3.5 Phase 5: Split Index Builders (Medium Risk)

**Task 5.1: Extract builder functions**
```python
# indexes/builders/embedding_builder.py
def build_embedding_index(
    index_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    corpus_config: Dict[str, Any],
    doc_batch_size: int = 5000,
) -> str:
    """Build a shared embedding index with batched processing."""
    ...

# indexes/builders/hierarchical_builder.py
def build_hierarchical_index(
    retriever_config: Dict[str, Any],
    corpus_config: Dict[str, Any],
    doc_batch_size: int = 1000,
) -> str:
    """Build a hierarchical index with batched processing."""
    ...
```

**Task 5.2: Keep orchestration in builder.py**
```python
# indexes/builder.py (slimmed)
from .builders.embedding_builder import build_embedding_index
from .builders.hierarchical_builder import build_hierarchical_index

def ensure_indexes_exist(
    retriever_configs: List[Dict[str, Any]],
    corpus_config: Dict[str, Any],
    build_if_missing: bool = True,
) -> List[str]:
    """Ensure all required indexes exist, building missing ones."""
    ...  # Orchestration logic only
```

---

## 4. Migration Strategy

### 4.1 Guiding Principle

**No backward compatibility for code APIs** - we can make breaking changes freely.

**Results must remain compatible** - existing experiment outputs must be loadable, or we provide migration scripts.

### 4.2 Incremental Approach

1. **Phase 1 first**: Clean up dead code (no risk)
2. **Add tests before Phase 2-5**: Ensure test coverage for affected code
3. **One package at a time**: Complete `spec/` before starting `factory/`
4. **Update imports everywhere**: No deprecation warnings, just change

### 4.3 Results Compatibility

**Artifacts that must remain loadable:**
```
outputs/<study>/<experiment>/
├── state.json          # ExperimentState serialization
├── predictions.json    # Predictions with per-item metrics
├── results.json        # Final aggregate results
├── questions.json      # Exported questions
└── metadata.json       # Experiment metadata
```

**If we change any artifact format:**
1. Create migration script in `scripts/migrations/`
2. Document the change in migration script docstring
3. Add to CHANGELOG

**Current migration scripts:**
- `scripts/migrate_to_new_format.py` - Index format migration (already exists)

### 4.4 Breaking Changes Allowed

These changes are fine (no migration needed):
- Renaming modules, classes, functions
- Changing function signatures
- Removing unused exports
- Restructuring packages

These changes need migration scripts:
- Changing `state.json` schema
- Changing `predictions.json` schema  
- Changing `results.json` schema
- Changing index artifact structure

---

## 5. Success Criteria

### 5.1 Quantitative

| Metric | Current | Target |
|--------|---------|--------|
| Largest file (lines) | 747 | < 300 |
| Files > 400 lines | 4 | 0 |
| Dead code lines | ~1,200 | 0 |
| Test coverage | ? | > 80% for new modules |

### 5.2 Qualitative

- [ ] Each class has a single, clear responsibility
- [ ] Dependencies are explicit (injected, not imported)
- [ ] Adding a new phase handler requires no changes to runner
- [ ] Adding a new factory type requires no changes to other factories
- [ ] Experiment can be fully configured without looking at implementation

### 5.3 Results Compatibility

- [ ] All existing `outputs/*/predictions.json` load correctly
- [ ] All existing `outputs/*/results.json` load correctly
- [ ] All existing `outputs/*/state.json` load correctly
- [ ] Analysis notebooks run without modification
- [ ] Resuming interrupted experiments works

---

## 6. Appendix: File Changes Summary

### Files to Delete
```
src/ragicamp/core/protocols.py
src/ragicamp/rag/chunking/semantic.py
src/ragicamp/utils/prediction_writer.py
src/ragicamp/metrics/ragas_adapter.py
src/ragicamp/analysis/mlflow_tracker.py
```

### Files to Create
```
src/ragicamp/spec/__init__.py
src/ragicamp/spec/experiment.py
src/ragicamp/spec/builder.py
src/ragicamp/spec/naming.py

src/ragicamp/state/__init__.py
src/ragicamp/state/experiment_state.py  (moved)
src/ragicamp/state/health.py

src/ragicamp/factory/__init__.py
src/ragicamp/factory/models.py
src/ragicamp/factory/datasets.py
src/ragicamp/factory/metrics.py
src/ragicamp/factory/retrievers.py
src/ragicamp/factory/agents.py

src/ragicamp/execution/phases/__init__.py
src/ragicamp/execution/phases/base.py
src/ragicamp/execution/phases/init.py
src/ragicamp/execution/phases/generation.py
src/ragicamp/execution/phases/metrics.py
src/ragicamp/execution/subprocess.py

src/ragicamp/indexes/builders/__init__.py
src/ragicamp/indexes/builders/embedding_builder.py
src/ragicamp/indexes/builders/hierarchical_builder.py
```

### Files to Modify
```
src/ragicamp/experiment.py          → Slim down to facade
src/ragicamp/execution/runner.py    → Remove spec building, use handlers
src/ragicamp/factory.py             → Deprecate, re-export from factory/
src/ragicamp/indexes/builder.py     → Keep orchestration only
src/ragicamp/experiment_state.py    → Move to state/
```

---

## 7. Migration Scripts

### 7.1 Existing Scripts
```
scripts/migrate_to_new_format.py    # Index format migration
```

### 7.2 Scripts to Create (if needed)

**If we change state.json schema:**
```python
# scripts/migrations/migrate_state_v2.py
"""Migrate state.json from v1 to v2 format.

Changes:
- Renamed 'phase' values: 'generating' -> 'generation'
- Added 'version' field

Usage:
    python scripts/migrations/migrate_state_v2.py outputs/
"""
```

**If we change analysis loader expectations:**
```python
# scripts/migrations/migrate_results_v2.py
"""Migrate results.json for new analysis module.

Changes:
- Normalized metric names
- Added 'schema_version' field

Usage:
    python scripts/migrations/migrate_results_v2.py outputs/
"""
```

### 7.3 Migration Validation

After any migration:
1. Run `uv run pytest tests/` 
2. Run analysis notebook on migrated outputs
3. Verify experiment resume works: `uv run ragicamp run conf/study/rag_strategies_test.yaml`

---

## 8. Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| No backward compatibility | Simplifies refactoring, codebase is internal | Deprecation warnings |
| Results must be compatible | Preserve experiment work, analysis notebooks | Regenerate everything |
| Split factory by component type | Each factory is cohesive | Keep monolithic with methods |
| Phase handlers as separate classes | Easy to add/modify phases | Switch statement in runner |
| Delete ragas_adapter | Not in use, adds complexity | Keep for future |
| Delete mlflow_tracker | Not actively used | Keep for future |
