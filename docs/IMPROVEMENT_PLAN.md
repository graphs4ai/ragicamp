# RAGiCamp Improvement Plan

> **DEPRECATED** (2026-02-13): This document is superseded by:
> - [BACKLOG.md](./BACKLOG.md) — Completed engineering audit (67/68 fixes, Feb 2026)
> - [FUTURE_WORK.md](./FUTURE_WORK.md) — Comprehensive roadmap (bugs, performance, research)
>
> Kept for historical reference only. All tasks below are complete.
>
> **Original purpose**: Track framework improvements needed to run the experiments defined in [EXPERIMENT_CONFIGURATIONS.md](./EXPERIMENT_CONFIGURATIONS.md).
>
> **Last updated**: 2026-01-31 (frozen)

---

## Status Overview

| Capability | Implementation | Config Exposure | Blocking Experiments |
|------------|----------------|-----------------|---------------------|
| Grid Search | ✅ Complete | ✅ Working | - |
| Singleton Experiments | ✅ Complete | ✅ Working | - |
| Agent Types | ✅ 5 types (direct, vanilla_rag, pipeline_rag, iterative_rag, self_rag) | ✅ Working | - |
| Chunking Strategy | ✅ Complete | ✅ Wired | - |
| Fetch-K (rerank pool) | ✅ Complete | ✅ Wired | - |
| Query Transform | ✅ Complete | ✅ Working | - |
| Reranking | ✅ Complete | ✅ Working | - |
| Hybrid/Hierarchical | ✅ Complete | ✅ Working | - |
| **AgentFactory.from_spec()** | ✅ Complete | ✅ Working | - |
| **Experiment.from_spec()** | ✅ Complete | ✅ Working | - |
| **ExperimentIO utility** | ✅ Complete | ✅ Working | - |

### Completed Tasks

- ✅ **Task 1.1**: Wired `chunking_strategy` from retriever config to index builder
- ✅ **Task 1.2**: Wired `fetch_k` from config to ExperimentSpec and agents
- ✅ **Task 2.1**: Added singleton experiment parsing (`experiments` list)
- ✅ **Task 3.1**: Extended AgentType enum with new agent types
- ✅ **Task 3.2**: Created VanillaRAGAgent with @AgentFactory.register
- ✅ **Task 3.3**: Added `pipeline_rag` and `direct` aliases to AgentFactory
- ✅ **Task R.1**: Added `AgentFactory.from_spec()` for spec-based agent creation
- ✅ **Task R.2**: Added `Experiment.from_spec()` for full component creation from spec
- ✅ **Task R.3**: Simplified `run_generation()` to use `Experiment.from_spec()`
- ✅ **Task R.4**: Created `ExperimentIO` utility for centralized file I/O
- ✅ **Task R.5**: Updated `run_metrics_only()` to use `ExperimentIO`

### Remaining Tasks

- ✅ **Task 3.4**: IterativeRAGAgent (multi-turn refinement) - COMPLETED
- ✅ **Task 3.5**: SelfRAGAgent (retrieval decision) - COMPLETED
- ⏳ **Task 4.1**: Cache predictions across execution phases
- ⏳ **Task 4.2**: Parallelize CPU-only metrics

---

## Part 1: Architecture Health Assessment

### 1.1 File Size Distribution

| Lines | Files | Assessment |
|-------|-------|------------|
| 0-200 | Most files | ✅ Good |
| 200-400 | ~15 files | ✅ Acceptable |
| 400-500 | 4 files | ⚠️ Monitor |
| 500+ | 3 files | ⚠️ Review needed |

**Largest files:**
- `experiment.py` (614 lines) - Experiment facade class
- `analysis/visualization.py` (565 lines) - Plotting utilities
- `execution/runner.py` (500 lines) - Experiment execution
- `cli/main.py` (495 lines) - CLI entry points

### 1.2 Identified Concerns

#### Concern 1: Three Entry Points for Running Experiments ✅ RESOLVED

~~There are 3 different ways to run experiments:~~

| Entry Point | File | Purpose | Status |
|-------------|------|---------|--------|
| `run_study()` | `cli/study.py` | CLI orchestration, loads config, builds specs | Orchestration only |
| `run_spec()` | `execution/runner.py` | Dispatch to generation or metrics-only | Thin wrapper |
| `Experiment.from_spec().run()` | `experiment.py` | **Single source of truth** | ✅ Primary API |

**Resolution**: `Experiment.from_spec()` now creates all components. `run_generation()` is now a thin wrapper:
```python
exp = Experiment.from_spec(spec, output_dir, limit, judge_model)
result = exp.run(batch_size=spec.batch_size, ...)
```

**Current flow** (clean):
```
cli/study.py:run_study()
  → spec/builder.py:build_specs()
  → execution/runner.py:run_spec()  # Dispatch only
    → run_generation() uses Experiment.from_spec()
    → OR run_metrics_only() for metrics-only path
```

#### Concern 2: `execution/runner.py` Does Too Much ✅ PARTIALLY RESOLVED

After refactoring, this file now contains:
- `run_spec()` - 80 lines, **dispatch only** (no component creation)
- `run_spec_subprocess()` - 120 lines, subprocess orchestration
- `run_generation()` - **40 lines** (down from 100+), uses `Experiment.from_spec()`
- `run_metrics_only()` - 80 lines, uses `ExperimentIO`

**Status**: Much improved. `run_generation()` reduced by 60%. Component creation logic moved to factories.

#### Concern 3: `spec/builder.py` Growing

After our changes, this file has:
- `build_specs()` - 50 lines
- `_build_direct_specs()` - 40 lines
- `_build_rag_specs()` - 90 lines
- `_build_singleton_specs()` - 100 lines

**Total**: 329 lines. Not critical, but if more builders are added, consider a `builders/` subpackage.

### 1.3 What's Good

1. **Clean layer separation**: `agents/`, `retrievers/`, `models/`, `metrics/` are well-isolated
2. **Factory pattern consistency**: All factories follow `create()` + `@register` pattern
3. **Frozen specs**: `ExperimentSpec` is immutable - prevents mutation bugs
4. **Focused agents**: `VanillaRAGAgent` (180 lines), `DirectLLMAgent` (~80 lines)
5. **No God objects**: Nothing is doing "everything"
6. **Proper abstraction**: `RAGPipeline` composes retriever + transformer + reranker

---

## Part 2: Technical Debt & Refactoring Tasks

### Completed Refactoring (2026-01-31)

These refactoring tasks have been completed to simplify adding new agent types:

#### ✅ Task R.1: Added `AgentFactory.from_spec()`

**Goal**: Enable spec-based agent creation with automatic component wiring.

**What was done**:
- Added `AgentFactory.from_spec(spec, model, retriever)` method
- Automatically creates query transformers, rerankers, and prompt builders from spec
- Handles agent-specific params via `spec.agent_params`
- Extracted `_create_instance()` for code reuse

**Files modified**:
- `factory/agents.py` - added `from_spec()` and refactored internals

**Impact**: Adding new agents now only requires `@AgentFactory.register("name")` decorator.

---

#### ✅ Task R.2: Enhanced `Experiment.from_spec()`

**Goal**: Create fully-configured experiments from ExperimentSpec.

**What was done**:
- Added `Experiment.from_spec(spec, output_dir, limit, judge_model)` 
- Creates model, dataset, agent, and metrics automatically
- Uses `AgentFactory.from_spec()` for agent creation
- Stores spec reference for metadata
- Added `Experiment.from_components()` for manual component creation

**Files modified**:
- `experiment.py` - enhanced `from_spec()`, added `_spec` field

**Impact**: Single line to create a complete experiment from config.

---

#### ✅ Task R.3: Simplified `run_generation()`

**Goal**: Remove duplicated component creation logic.

**What was done**:
- Reduced from 100+ lines to ~40 lines
- Now uses `Experiment.from_spec()` for all component creation
- Uses `ExperimentIO` for metadata saving

**Files modified**:
- `execution/runner.py` - simplified `run_generation()`

**Impact**: No more `if spec.exp_type == "direct" ... else ...` branches in runner.

---

#### ✅ Task R.4: Created `ExperimentIO` Utility

**Goal**: Centralize file I/O with consistent atomic writes.

**What was done**:
- Created `utils/experiment_io.py` with `ExperimentIO` class
- Methods: `save_predictions()`, `load_predictions()`, `save_result()`, etc.
- All writes use atomic pattern (write to temp, then rename)
- Standalone functions for backward compatibility

**Files created**:
- NEW `utils/experiment_io.py`

**Files modified**:
- `utils/__init__.py` - export `ExperimentIO`
- `execution/runner.py` - use `ExperimentIO`
- `execution/phases/generation.py` - use `ExperimentIO`

**Impact**: Consistent I/O patterns, reduced crash corruption risk.

---

### Remaining Refactoring (Optional, Low Priority)

#### Task R.5: Consider `spec/builders/` Subpackage

**Goal**: Split spec builders if more are added.

**When to do this**: If we add more than 4 builder functions.

**Proposed structure**:
```
spec/
  __init__.py
  experiment.py
  naming.py
  builders/
    __init__.py
    direct.py      # _build_direct_specs()
    rag.py         # _build_rag_specs()
    singleton.py   # _build_singleton_specs()
```

**Effort**: Low
**Impact**: Easier navigation, but only if file grows further

---

#### Task R.6: Remove Re-exports from `cli/study.py`

**Goal**: Clean up backward-compatibility re-exports.

**Current state**:
- `cli/study.py` lines 25-54 re-export from other modules
- Creates maintenance burden and import confusion

**Proposed change**:
- Remove re-exports, fix callers to import from correct modules
- Or add deprecation warnings

**Effort**: Low
**Impact**: Cleaner imports, reduced maintenance

---

## Part 3: Remaining Feature Tasks

### Task 3.4: IterativeRAGAgent (Multi-turn Refinement)

**Status**: ✅ COMPLETED (2026-01-31)

**Purpose**: Refine query based on initial retrieval results.

**Flow**:
1. Retrieve with original query
2. LLM evaluates: "Is context sufficient to answer?"
3. If not sufficient: LLM generates refined query → retrieve again
4. Merge documents (deduplicate by content hash)
5. Repeat until max_iterations or sufficient
6. Generate final answer with accumulated context

**Files**: `agents/iterative_rag.py`

**Config format**:
```yaml
experiments:
  - name: iterative_test
    agent_type: iterative_rag
    retriever: dense_bge
    top_k: 5
    max_iterations: 2        # Via agent_params
    stop_on_sufficient: true
```

**Acceptance criteria**:
- [x] Uses `@AgentFactory.register("iterative_rag")` decorator
- [x] Configurable `max_iterations` (default: 2)
- [x] Tracks iterations in response metadata (`context.intermediate_steps`)
- [x] Works in singleton experiments via `AgentFactory.from_spec()`
- [x] Unit tests passing

---

### Task 3.5: SelfRAGAgent (Retrieval Decision)

**Status**: ✅ COMPLETED (2026-01-31)

**Purpose**: Model decides whether to use retrieval based on query.

**Flow**:
1. Assess query: "Do I need external information?"
2. If confident (above threshold): generate directly (no retrieval)
3. If unsure: use RAG path
4. Optionally verify answer is supported by context
5. If not supported and fallback enabled: use direct answer

**Files**: `agents/self_rag.py`

**Config format**:
```yaml
experiments:
  - name: selfrag_test
    agent_type: self_rag
    retriever: dense_bge
    top_k: 5
    retrieval_threshold: 0.5   # Via agent_params
    verify_answer: true
    fallback_to_direct: true
```

**Acceptance criteria**:
- [x] Uses `@AgentFactory.register("self_rag")` decorator
- [x] Tracks retrieval decision in response metadata (`context.metadata.used_retrieval`)
- [x] Configurable threshold and verification
- [x] Works in singleton experiments via `AgentFactory.from_spec()`
- [x] Unit tests passing

---

### Task 4.1: Cache Predictions Across Phases

**Status**: ⏳ Pending (Low Priority)

**Current**: Predictions written to JSON, then re-read by metrics phase.
**Goal**: Keep predictions in `ExecutionContext` to avoid redundant I/O.

---

### Task 4.2: Parallelize CPU-only Metrics

**Status**: ⏳ Pending (Low Priority)

**Current**: Metrics run sequentially.
**Goal**: Run CPU-only metrics (exact_match, f1) in ThreadPoolExecutor while GPU metrics run.

---

## Part 4: Mapping to Experiments

| Experiment Phase | Required Tasks | Status |
|-----------------|----------------|--------|
| A: Baselines | None | ✅ Ready |
| B: Embedding Models | None | ✅ Ready |
| C: Chunk Size & Strategy | 1.1 | ✅ Ready |
| D: Retrieval Strategies | None | ✅ Ready |
| E: Top-K and Reranking | 1.2 | ✅ Ready |
| F: Query Transformation | None | ✅ Ready |
| G: Reranker Comparison | 1.2 | ✅ Ready |
| H: Agent Architectures | 3.4, 3.5 | ✅ Ready (IterativeRAG, SelfRAG implemented) |
| I: Prompt Engineering | None | ✅ Ready |

---

## Part 5: Existing Patterns to Follow

### 5.1 Factory Pattern with Registry

```python
# factory/agents.py - use this pattern for new agents
class AgentFactory:
    _custom_agents: Dict[str, type] = {}
    _aliases: Dict[str, str] = {"pipeline_rag": "fixed_rag", "direct": "direct_llm"}

    @classmethod
    def register(cls, name: str):
        def decorator(agent_class: type) -> type:
            cls._custom_agents[name] = agent_class
            return agent_class
        return decorator
```

### 5.2 Agent Implementation Pattern (Updated)

After the refactoring, adding a new agent is simple:

```python
# agents/iterative_rag.py
from ragicamp.agents.base import RAGAgent, RAGResponse
from ragicamp.factory import AgentFactory

@AgentFactory.register("iterative_rag")
class IterativeRAGAgent(RAGAgent):
    """Multi-turn refinement RAG agent."""
    
    def __init__(
        self,
        name: str,
        model,
        retriever,
        max_iterations: int = 2,
        stop_on_sufficient: bool = True,
        prompt_builder=None,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.stop_on_sufficient = stop_on_sufficient
        self.prompt_builder = prompt_builder
        self.top_k = top_k
    
    def answer(self, query: str, **kwargs) -> RAGResponse:
        # Implementation: retrieve, evaluate, refine, repeat
        pass
    
    def batch_answer(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        # Optimized batch implementation
        return [self.answer(q, **kwargs) for q in queries]
```

**That's it!** No changes needed to:
- `execution/runner.py` (uses `AgentFactory.from_spec()`)
- `Experiment` class (uses `AgentFactory.from_spec()`)
- Any CLI code

Config usage:
```yaml
experiments:
  - name: iterative_test
    agent_type: iterative_rag
    retriever: dense_bge
    top_k: 5
    max_iterations: 2      # Passed via agent_params
    stop_on_sufficient: true
```

### 5.3 AgentType Enum (Current State)

```python
# core/schemas.py - extend when adding new agents
class AgentType(str, Enum):
    DIRECT_LLM = "direct_llm"
    FIXED_RAG = "fixed_rag"
    VANILLA_RAG = "vanilla_rag"
    PIPELINE_RAG = "pipeline_rag"
    ITERATIVE_RAG = "iterative_rag"  # Ready for implementation
    SELF_RAG = "self_rag"            # Ready for implementation
```

---

## Part 6: Design Checklist

Before implementing any task:

### Compatibility
- [ ] Uses `@AgentFactory.register` decorator
- [ ] Extends `AgentType` enum if new agent
- [ ] Follows `to_dict()`/`from_dict()` pattern

### Maintainability
- [ ] Single responsibility (each class does one thing)
- [ ] Dependency injection (pass components, don't create internally)
- [ ] Errors have helpful messages

### Performance
- [ ] Supports batch operations (`batch_answer`, `batch_retrieve`)
- [ ] GPU memory cleaned up after use

### Backwards Compatibility
- [ ] Old config formats still work
- [ ] Default values match current behavior

---

## Appendix: File Reference

| File | Lines | Purpose | Health |
|------|-------|---------|--------|
| `experiment.py` | ~650 | Experiment facade + `from_spec()` | ✅ Well-structured |
| `execution/runner.py` | ~420 | Spec execution (simplified) | ✅ Improved |
| `cli/main.py` | 495 | CLI commands | ✅ Fine for CLI |
| `cli/study.py` | 236 | Study orchestration | ✅ Good |
| `spec/builder.py` | 329 | Spec building | ✅ Monitor growth |
| `agents/fixed_rag.py` | 300 | Pipeline RAG agent | ✅ Good |
| `agents/vanilla_rag.py` | 180 | Simple RAG agent | ✅ Good |
| `factory/agents.py` | ~200 | Agent factory + `from_spec()` | ✅ Good |
| `utils/experiment_io.py` | ~230 | Centralized I/O | ✅ NEW |
