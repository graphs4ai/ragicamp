# RAGiCamp Improvement Plan

> **Purpose**: Track framework improvements needed to run the experiments defined in [EXPERIMENT_CONFIGURATIONS.md](./EXPERIMENT_CONFIGURATIONS.md).
>
> **Last updated**: 2026-01-31

---

## Status Overview

| Capability | Implementation | Config Exposure | Blocking Experiments |
|------------|----------------|-----------------|---------------------|
| Grid Search | ✅ Complete | ✅ Working | - |
| Singleton Experiments | ❌ Missing | ❌ Missing | Phase H (agent comparison) |
| Agent Types | ⚠️ Only 2 types | ⚠️ Hardcoded | Phase H (advanced agents) |
| Chunking Strategy | ✅ **Implemented** | ❌ Not wired | Phase C (chunk strategies) |
| Fetch-K (rerank pool) | ✅ **Implemented** | ❌ Not wired | Phase E, G (reranking) |
| Query Transform | ✅ Complete | ✅ Working | - |
| Reranking | ✅ Complete | ✅ Working | - |
| Hybrid/Hierarchical | ✅ Complete | ✅ Working | Phase D |

**Key Insight**: Several capabilities are already implemented but not exposed in config. These are quick wins.

---

## Part 1: Existing Patterns to Follow

Before implementing anything, understand the patterns already in use.

### 1.1 Factory Pattern with Registry

```python
# factory/agents.py - existing pattern
class AgentFactory:
    _custom_agents: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register custom agents."""
        def decorator(agent_class: type) -> type:
            cls._custom_agents[name] = agent_class
            return agent_class
        return decorator

    @classmethod
    def create(cls, config: Dict, model: LanguageModel, retriever: Optional[Retriever] = None):
        ...
```

**Use this pattern** for new agents. Don't create a separate registry module.

### 1.2 Pydantic Config Schemas

```python
# config/schemas.py - ChunkingConfig already exists!
class ChunkingConfig(BaseModel):
    strategy: str = Field(default="recursive", ...)
    chunk_size: int = Field(default=512, ...)
    chunk_overlap: int = Field(default=50, ...)

class RetrieverConfig(BaseModel):
    chunking: Optional[ChunkingConfig] = Field(default=None, ...)
```

**Use existing schemas**. Extend, don't duplicate.

### 1.3 Core Data Schemas

```python
# core/schemas.py - AgentType enum
class AgentType(str, Enum):
    DIRECT_LLM = "direct_llm"
    FIXED_RAG = "fixed_rag"
    # Add new agent types here
```

**Extend the enum** when adding new agent types.

### 1.4 RAGPipeline Already Has Fetch-K

```python
# rag/pipeline.py - already implemented!
class RAGPipeline:
    def __init__(
        self,
        retriever: "Retriever",
        top_k_retrieve: int = 20,  # ← This is fetch_k!
        top_k_final: int = 5,      # ← This is top_k after reranking
        ...
    ):
```

**Just wire to config**. No new implementation needed.

### 1.5 Chunking Already Implemented

```python
# corpus/chunking.py - all strategies exist!
def get_chunker(config: ChunkConfig) -> ChunkingStrategy:
    strategies = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "recursive": RecursiveChunker,
    }
    return strategies[config.strategy](config)
```

**Just wire to index builder**. The chunking code is complete.

---

## Part 2: Implementation Tasks

### Phase 1: Configuration Wiring (Quick Wins)

These tasks expose existing functionality in config. Minimal new code.

#### Task 1.1: Wire chunking_strategy to index builder ⭐ QUICK WIN

**Current state**: 
- `ChunkingConfig` exists in `config/schemas.py`
- `get_chunker()` factory exists in `corpus/chunking.py`
- Index builder uses hardcoded `RecursiveChunker`

**Goal**: Parse `chunking_strategy` from retriever config and pass to index builder.

**Files to modify**:
- `indexes/builders/embedding_builder.py` - accept `ChunkConfig`
- `indexes/builder.py` - parse chunking from retriever config
- `cli/study.py` - pass chunking config through

**Config format** (already supported by `RetrieverConfig`):
```yaml
retrievers:
  - type: dense
    name: dense_paragraph
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 1024
    chunking_strategy: paragraph  # NEW - now wired!
```

**Acceptance criteria**:
- [ ] `chunking_strategy` parsed from retriever config
- [ ] `ChunkConfig` created and passed to `get_chunker()`
- [ ] Index builder uses configured strategy instead of hardcoded recursive
- [ ] Backwards compatible (default = recursive)

---

#### Task 1.2: Wire fetch_k to ExperimentSpec and agents ⭐ QUICK WIN

**Current state**:
- `RAGPipeline` has `top_k_retrieve` (fetch_k) and `top_k_final` (top_k)
- `FixedRAGAgent` calculates `top_k_retrieve = top_k * 4` if reranker present
- Not configurable from YAML

**Goal**: Allow explicit `fetch_k` in config.

**Files to modify**:
- `spec/experiment.py` - add `fetch_k: Optional[int]` field
- `spec/builder.py` - parse `fetch_k` from config
- `cli/study.py` - pass `fetch_k` to agent creation
- `agents/fixed_rag.py` - use explicit `top_k_retrieve` when provided

**Config format**:
```yaml
# Grid search
rag:
  top_k_values: [3, 5]
  fetch_k_multiplier: 4  # Optional: fetch_k = top_k * multiplier

# Singleton (future)
experiments:
  - name: rerank_test
    top_k: 3
    fetch_k: 20  # Explicit: retrieve 20, rerank to 3
```

**Acceptance criteria**:
- [ ] `fetch_k` field in `ExperimentSpec`
- [ ] Grid search can specify `fetch_k_multiplier` or explicit `fetch_k_values`
- [ ] `FixedRAGAgent` uses explicit `top_k_retrieve` when provided
- [ ] Backwards compatible (default = `top_k * 4` when reranker present)

---

### Phase 2: Singleton Experiments

Enables hypothesis-driven research instead of just grid search.

#### Task 2.1: Add singleton experiment parsing

**Current state**:
- `build_specs()` only handles `direct` and `rag` grid search blocks
- No way to define individual experiments

**Goal**: Support `experiments` list for individual experiment definitions.

**Files to modify**:
- `spec/experiment.py` - add optional fields: `agent_type`, `hypothesis`, `agent_params`
- `spec/builder.py` - add `_build_singleton_specs()` function

**Config format**:
```yaml
# Grid search (existing, still works)
rag:
  enabled: true
  models: [...]
  retrievers: [...]

# Singleton experiments (NEW)
experiments:
  - name: baseline_vanilla
    agent_type: vanilla_rag  # or fixed_rag, iterative_rag, etc.
    model: hf:meta-llama/Llama-3.2-3B-Instruct
    retriever: dense_bge_c512
    dataset: nq
    top_k: 5
    prompt: concise
    hypothesis: "Simple RAG baseline without enhancements"
```

**Changes to ExperimentSpec**:
```python
@dataclass(frozen=True)
class ExperimentSpec:
    # Existing fields...
    
    # NEW fields
    agent_type: Optional[str] = None      # explicit agent type
    fetch_k: Optional[int] = None         # from Task 1.2
    hypothesis: Optional[str] = None      # documentation
    agent_params: Dict[str, Any] = field(default_factory=dict)  # agent-specific config
```

**Acceptance criteria**:
- [ ] `experiments` list parsed if present
- [ ] Each experiment creates one `ExperimentSpec`
- [ ] Can coexist with grid search blocks
- [ ] `hypothesis` field is optional (for documentation)

---

### Phase 3: Agent Hierarchy

Build on existing factory pattern. Start simple, add complexity incrementally.

#### Task 3.1: Extend AgentType enum

**Files**: `core/schemas.py`

```python
class AgentType(str, Enum):
    DIRECT_LLM = "direct_llm"
    FIXED_RAG = "fixed_rag"
    # NEW
    VANILLA_RAG = "vanilla_rag"
    PIPELINE_RAG = "pipeline_rag"  # alias for fixed_rag
    ITERATIVE_RAG = "iterative_rag"
    SELF_RAG = "self_rag"
```

---

#### Task 3.2: Create VanillaRAGAgent (minimal RAG baseline)

**Purpose**: Simplest possible RAG - retrieve and generate. No query transform, no reranking.

**Why it matters**: Clean baseline for ablation studies. Current `FixedRAGAgent` has optional pipeline complexity.

**Files**: NEW `agents/vanilla_rag.py`

**Implementation**:
```python
from ragicamp.factory.agents import AgentFactory

@AgentFactory.register("vanilla_rag")  # Use existing decorator!
class VanillaRAGAgent(RAGAgent):
    """Simplest RAG: retrieve → generate."""
    
    def __init__(self, name, model, retriever, top_k=5, prompt_builder=None, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.prompt_builder = prompt_builder or PromptBuilder()
    
    def answer(self, query: str, **kwargs) -> RAGResponse:
        docs = self.retriever.retrieve(query, top_k=self.top_k)
        context = ContextFormatter.format_numbered(docs)
        prompt = self.prompt_builder.build_rag(query, context)
        answer = self.model.generate(prompt, **kwargs)
        return RAGResponse(answer=answer, context=RAGContext(query=query, retrieved_docs=docs), prompt=prompt)
    
    def batch_answer(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        # Use batch retrieval and batch generation
        all_docs = self.retriever.batch_retrieve(queries, top_k=self.top_k)
        prompts = [self.prompt_builder.build_rag(q, ContextFormatter.format_numbered(d)) 
                   for q, d in zip(queries, all_docs)]
        answers = self.model.generate(prompts, **kwargs)
        return [RAGResponse(...) for ...]
```

**Acceptance criteria**:
- [ ] Uses `@AgentFactory.register("vanilla_rag")` decorator
- [ ] No pipeline, no query transform, no reranking
- [ ] Supports batch processing
- [ ] Works in singleton experiments with `agent_type: vanilla_rag`

---

#### Task 3.3: Add pipeline_rag alias for fixed_rag

**Purpose**: Clearer naming. "Fixed" is misleading since it supports dynamic pipeline features.

**Files**: `factory/agents.py`

```python
@classmethod
def create(cls, config, model, retriever=None):
    agent_type = config["type"]
    
    # Resolve aliases
    ALIASES = {"pipeline_rag": "fixed_rag"}
    agent_type = ALIASES.get(agent_type, agent_type)
    
    # ... rest of existing logic
```

**Acceptance criteria**:
- [ ] `agent_type: pipeline_rag` works in config
- [ ] `agent_type: fixed_rag` still works (backwards compat)
- [ ] Both create the same `FixedRAGAgent` class

---

#### Task 3.4: IterativeRAGAgent (multi-turn refinement)

**Purpose**: Refine query based on initial retrieval results.

**Flow**:
1. Retrieve with original query
2. LLM evaluates: "Is context sufficient to answer?"
3. If not sufficient: LLM generates refined query → retrieve again
4. Merge documents (deduplicate by ID)
5. Repeat until max_iterations or sufficient
6. Generate final answer with accumulated context

**Files**: NEW `agents/iterative_rag.py`

**Config format**:
```yaml
experiments:
  - name: iterative_test
    agent_type: iterative_rag
    retriever: dense_bge
    top_k: 5
    agent_params:
      max_iterations: 2
      stop_on_sufficient: true
```

**Acceptance criteria**:
- [ ] Uses `@AgentFactory.register("iterative_rag")` decorator
- [ ] Configurable `max_iterations` (default: 2)
- [ ] Tracks iterations in response metadata
- [ ] Works in singleton experiments

---

#### Task 3.5: SelfRAGAgent (retrieval decision)

**Purpose**: Model decides whether to use retrieval based on query.

**Flow**:
1. Assess query: "Do I need external information?"
2. If confident (above threshold): generate directly (no retrieval)
3. If unsure: use RAG path
4. Optionally verify answer is supported by context

**Files**: NEW `agents/self_rag.py`

**Config format**:
```yaml
experiments:
  - name: selfrag_test
    agent_type: self_rag
    retriever: dense_bge
    top_k: 5
    agent_params:
      retrieval_threshold: 0.5
      verify_answer: true
      fallback_to_direct: true
```

**Acceptance criteria**:
- [ ] Uses `@AgentFactory.register("self_rag")` decorator
- [ ] Tracks retrieval decision in response metadata
- [ ] Configurable threshold and verification
- [ ] Works in singleton experiments

---

### Phase 4: Optimizations (Low Priority)

#### Task 4.1: Cache predictions across execution phases

**Current**: Predictions written to JSON, then re-read by metrics phase.
**Goal**: Keep predictions in `ExecutionContext` to avoid redundant I/O.

#### Task 4.2: Parallelize CPU-only metrics

**Current**: Metrics run sequentially.
**Goal**: Run CPU-only metrics (exact_match, f1) in ThreadPoolExecutor while GPU metrics run.

---

## Part 3: Implementation Order

```
Phase 1: Configuration Wiring (Quick Wins) ← START HERE
  1.1 Wire chunking_strategy ──┐
  1.2 Wire fetch_k ────────────┴──→ Unlocks: Phase C, E, G experiments

Phase 2: Singleton Experiments
  2.1 Add experiments list parsing ──→ Unlocks: hypothesis-driven research

Phase 3: Agent Hierarchy (depends on Phase 2)
  3.1 Extend AgentType enum ──┐
  3.2 VanillaRAGAgent ────────┤
  3.3 pipeline_rag alias ─────┴──→ 3.4 IterativeRAGAgent ──→ 3.5 SelfRAGAgent
                                   └──→ Unlocks: Phase H experiments

Phase 4: Optimizations (independent, low priority)
  4.1 Cache predictions
  4.2 Parallel metrics
```

**Recommended order**: 1.1 → 1.2 → 2.1 → 3.1 → 3.2 → 3.3 → 3.4 → 3.5

**Rationale**:
1. **Phase 1 first**: Quick wins that unblock experiments with minimal code
2. **Phase 2 second**: Singleton experiments enable testing new agents
3. **Phase 3 third**: Build agents incrementally (vanilla → advanced)
4. **Phase 4 last**: Performance optimization is lower priority than features

---

## Part 4: Mapping to Experiments

| Experiment Phase | Required Tasks | Status |
|-----------------|----------------|--------|
| A: Baselines | None | ✅ Ready |
| B: Embedding Models | None | ✅ Ready |
| C: Chunk Size & Strategy | **1.1** | ⏳ Wire chunking_strategy |
| D: Retrieval Strategies | None | ✅ Ready |
| E: Top-K and Reranking | **1.2** | ⏳ Wire fetch_k |
| F: Query Transformation | None | ✅ Ready |
| G: Reranker Comparison | **1.2** | ⏳ Wire fetch_k |
| H: Agent Architectures | **2.1, 3.1-3.5** | ⏳ Full agent hierarchy |
| I: Prompt Engineering | None | ✅ Ready |

---

## Part 5: Design Checklist

Before implementing, verify each task:

### Compatibility

- [ ] Uses existing factory pattern (`@AgentFactory.register` or `@RetrieverFactory.register`)
- [ ] Extends existing Pydantic schemas (don't create parallel schemas)
- [ ] Follows `to_dict()`/`from_dict()` pattern for serialization
- [ ] Extends `AgentType` enum for new agent types

### Maintainability

- [ ] Single responsibility (each class does one thing)
- [ ] Dependency injection (pass components, don't create internally)
- [ ] Configuration validated in spec layer, not in agent
- [ ] Errors have helpful messages

### Performance

- [ ] Supports batch operations (`batch_answer`, `batch_retrieve`)
- [ ] GPU memory cleaned up after use (`ResourceManager.clear_gpu_memory()`)
- [ ] Avoids redundant I/O (use in-memory caching where appropriate)

### Backwards Compatibility

- [ ] Old config formats still work
- [ ] Aliases for renamed components
- [ ] Default values match current behavior

---

## Appendix: File Reference

| File | Purpose | Patterns |
|------|---------|----------|
| `factory/agents.py` | Agent factory with registry | `@register` decorator, `create()` method |
| `factory/retrievers.py` | Retriever factory | Same pattern as agents |
| `config/schemas.py` | Pydantic validation schemas | `ChunkingConfig`, `RetrieverConfig`, etc. |
| `core/schemas.py` | Core data structures | `AgentType` enum, `RAGResponseMeta` |
| `spec/experiment.py` | Immutable experiment spec | Frozen dataclass with `to_dict()` |
| `spec/builder.py` | Build specs from YAML | `build_specs()`, `_build_rag_specs()` |
| `corpus/chunking.py` | Chunking strategies | `get_chunker()` factory |
| `rag/pipeline.py` | RAG pipeline orchestration | `top_k_retrieve`, `top_k_final` |
| `indexes/builder.py` | Index building orchestration | `ensure_indexes_exist()` |
