# Refactoring Specification: ragicamp Architecture Review

This document identifies architectural issues discovered during the Feb 2026 codebase review, with recommendations for keeping the codebase maintainable as features grow.

## Executive Summary

The ragicamp codebase has grown significantly with new features (vLLM embeddings, sparse indexes, hybrid retrieval, grid/random sampling). While the core architecture remains solid, several modules have accumulated too many responsibilities and should be refactored to maintain the original design principles.

**Critical Modules Requiring Attention:**
1. `spec/builder.py` (522 lines) - Monolithic spec construction
2. `indexes/embedding.py` (715 lines) - Mixed concerns (encoder + FAISS + GPU)
3. `indexes/builder.py` (265 lines) - Complex orchestration with sparse index building

---

## Issue #1: Monolithic Spec Builder

**File:** `src/ragicamp/spec/builder.py` (522 lines)

### Problem

The spec builder has grown to handle too many distinct responsibilities:
- Grid search expansion (direct, RAG, singleton)
- Random/stratified sampling
- Experiment naming conventions
- Config parsing and validation
- Default value resolution

Recent commits added sampling logic (+90 lines) and stratified sampling (+50 lines), pushing this module toward monolith territory.

### Current Structure

```python
# builder.py - Does too much
def build_specs(config, sampling_override, exclude_names): ...
def _apply_sampling(specs, sampling_config, spec_type, exclude_names): ...
def _stratified_sample(specs, n_experiments, stratify_by): ...
def _build_direct_specs(direct_config, datasets, ...): ...
def _build_rag_specs(rag_config, datasets, ...): ...
def _build_singleton_specs(experiments, config, ...): ...
```

### Proposed Refactoring

Split into focused modules under `spec/`:

```
src/ragicamp/spec/
├── __init__.py           # Public API: build_specs()
├── experiment.py         # ExperimentSpec dataclass (existing)
├── naming.py             # Naming conventions (existing)
├── builder.py            # Orchestrator - delegates to submodules (100-150 lines)
├── grid/
│   ├── __init__.py
│   ├── direct.py         # _build_direct_specs()
│   ├── rag.py            # _build_rag_specs()
│   └── singleton.py      # _build_singleton_specs()
└── sampling/
    ├── __init__.py
    ├── random.py         # Random sampling
    └── stratified.py     # Stratified sampling with dimension-aware logic
```

### Refactored API

```python
# spec/builder.py - Thin orchestrator
from ragicamp.spec.grid import build_direct_specs, build_rag_specs, build_singleton_specs
from ragicamp.spec.sampling import apply_sampling

def build_specs(config, sampling_override=None, exclude_names=None):
    """Main entry point - orchestrates submodules."""
    specs = []
    
    if config.get("direct", {}).get("enabled"):
        specs.extend(build_direct_specs(config))
    
    if config.get("rag", {}).get("enabled"):
        rag_specs = build_rag_specs(config)
        if sampling_override or config.get("rag", {}).get("sampling"):
            rag_specs = apply_sampling(rag_specs, sampling_override, exclude_names)
        specs.extend(rag_specs)
    
    if experiments := config.get("experiments"):
        specs.extend(build_singleton_specs(experiments, config))
    
    return specs
```

### Migration Path

1. Create `spec/grid/` directory with `direct.py`, `rag.py`, `singleton.py`
2. Move functions to respective modules
3. Create `spec/sampling/` with `random.py`, `stratified.py`
4. Update imports in `builder.py` to delegate
5. Update tests to import from new locations
6. Deprecate but keep old imports working for one release

---

## Issue #2: EmbeddingIndex Responsibilities

**File:** `src/ragicamp/indexes/embedding.py` (715 lines)

### Problem

`EmbeddingIndex` handles four distinct concerns:
1. **Encoder management** - Loading sentence-transformers or vLLM
2. **FAISS index operations** - Create, train, search (flat/ivf/hnsw)
3. **GPU resource management** - FAISS GPU resources, memory cleanup
4. **Persistence** - Save/load index and documents

Recent commits added vLLM backend support (+80 lines) and index conversion (+60 lines), exacerbating the issue.

### Current Pain Points

```python
class EmbeddingIndex:
    # Encoder concerns
    def _load_vllm_encoder(self): ...
    def _load_sentence_transformers_encoder(self): ...
    
    # FAISS concerns
    def _create_faiss_index(self, num_vectors): ...
    def _set_search_params(self): ...
    
    # GPU concerns
    # Global functions: _check_faiss_gpu_available(), get_faiss_gpu_resources()
    
    # Build/search concerns
    def build(self, documents, batch_size): ...
    def search(self, query_embedding, top_k): ...
    def batch_search(self, query_embeddings, top_k): ...
    
    # Conversion
    def convert_to(self, new_index_type): ...
```

### Proposed Refactoring

Extract into focused classes:

```
src/ragicamp/indexes/
├── __init__.py
├── embedding.py          # EmbeddingIndex - orchestrates components (300 lines)
├── faiss_backend.py      # FAISSBackend - index creation, GPU, search
├── encoders/
│   ├── __init__.py
│   ├── base.py           # EncoderProtocol
│   ├── sentence_transformer.py
│   └── vllm.py
└── gpu_resources.py      # FAISS GPU resource singleton
```

### Proposed Structure

```python
# encoders/base.py
from typing import Protocol

class Encoder(Protocol):
    """Unified interface for embedding models."""
    
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...
    def get_embedding_dimension(self) -> int: ...
    def unload(self) -> None: ...


# encoders/sentence_transformer.py
class SentenceTransformerEncoder:
    """Encoder using sentence-transformers with optimizations."""
    
    def __init__(self, model_name: str, use_fp16: bool = True, use_flash_attn: bool = True):
        ...
    
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._model.encode(texts, batch_size=batch_size, ...)


# encoders/vllm.py
class VLLMEncoder:
    """Encoder using vLLM for high-throughput embedding."""
    # Move from models/vllm_embedder.py - it's really an encoder, not a model


# faiss_backend.py
class FAISSBackend:
    """Handles FAISS index creation, training, and search."""
    
    def __init__(self, index_type: str, use_gpu: bool, nlist: int, nprobe: int):
        ...
    
    def create_index(self, embedding_dim: int, num_vectors: int) -> faiss.Index: ...
    def train(self, vectors: np.ndarray) -> None: ...
    def add(self, vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...
    def to_gpu(self) -> None: ...
    def to_cpu(self) -> None: ...


# embedding.py - Orchestrator
class EmbeddingIndex:
    """Embedding index combining encoder and FAISS backend."""
    
    def __init__(
        self,
        name: str,
        encoder: Encoder | None = None,  # Inject or create
        faiss_backend: FAISSBackend | None = None,
    ):
        self._encoder = encoder
        self._backend = faiss_backend
    
    def build(self, documents: list[Document], batch_size: int = 64):
        texts = [doc.text for doc in documents]
        embeddings = self._encoder.encode(texts, batch_size)
        self._backend.train(embeddings)
        self._backend.add(embeddings)
```

### Benefits

1. **Testability** - Can unit test encoder and FAISS separately
2. **Flexibility** - Easy to add new encoder backends (Cohere, OpenAI embeddings)
3. **Clearer contracts** - Encoder protocol makes expectations explicit
4. **Simpler EmbeddingIndex** - Just orchestration, ~300 lines

---

## Issue #3: Index Building Chain Complexity

**Files:**
- `indexes/builder.py` (265 lines) - Orchestration
- `indexes/builders/embedding_builder.py` (379 lines) - Actual building
- Multiple concerns mixed in both

### Problem

The index building flow is:

```
ensure_indexes_exist()     # builder.py
    → build_embedding_index()  # builders/embedding_builder.py
        → EmbeddingIndex.build()  # embedding.py
    → build_retriever_from_index()  # builder.py (also builds sparse!)
        → SparseIndex.build()  # sparse.py
```

Issues:
1. `builder.py` both orchestrates AND builds sparse indexes
2. `build_retriever_from_index()` creates retrievers AND builds sparse indexes (two jobs)
3. Shared sparse index caching logic is in the wrong layer

### Current Code Smell

```python
# builder.py - Does retriever creation AND sparse index building
def build_retriever_from_index(retriever_config, embedding_index_name, shared_sparse_indexes):
    # Creates retriever config
    retriever_cfg = {...}
    
    # BUT ALSO builds sparse index if hybrid
    if retriever_type == "hybrid":
        if cache_key not in shared_sparse_indexes:
            sparse_index = SparseIndex(...)
            sparse_index.build(documents)  # <-- Building here!
            sparse_index.save()
```

### Proposed Refactoring

```
src/ragicamp/indexes/
├── builder.py              # Pure orchestration (ensure_indexes_exist)
├── builders/
│   ├── embedding_builder.py
│   ├── sparse_builder.py   # New: dedicated sparse index building
│   └── retriever_config.py # New: creates retriever configs (no building)
```

### Proposed Flow

```python
# builder.py - Pure orchestration
def ensure_indexes_exist(retriever_configs, corpus_config):
    # Step 1: Build embedding indexes
    for index_name in unique_embedding_indexes:
        if not exists(index_name):
            build_embedding_index(...)
    
    # Step 2: Build sparse indexes (for hybrid retrievers)
    for config in hybrid_retrievers:
        sparse_name = get_sparse_index_name(...)
        if not exists(sparse_name):
            build_sparse_index(config, embedding_index_name)
    
    # Step 3: Create retriever configs (no building, just config files)
    for config in retriever_configs:
        if not exists(config["name"]):
            create_retriever_config(config, embedding_index_name)
```

---

## Issue #4: Dual Embedder Interface

**Files:**
- `models/vllm_embedder.py` (178 lines)
- Used in `indexes/embedding.py` via conditional loading

### Problem

VLLMEmbedder and SentenceTransformer have similar but not identical interfaces. The `EmbeddingIndex` has two separate loading paths:

```python
def _load_vllm_encoder(self):
    from ragicamp.models.vllm_embedder import VLLMEmbedder
    self._encoder = VLLMEmbedder(...)

def _load_sentence_transformers_encoder(self):
    from sentence_transformers import SentenceTransformer
    self._encoder = SentenceTransformer(...)
```

### Resolution

This is addressed by Issue #2's proposal. The VLLMEmbedder should move to `indexes/encoders/vllm.py` since:
1. It's only used for embeddings, not generation
2. It belongs with other encoders, not in `models/`
3. The unified `Encoder` protocol handles interface differences

---

## Issue #5: CLI Study.py Orchestration

**File:** `src/ragicamp/cli/study.py` (257 lines)

### Assessment

This file is actually well-structured. It:
- Delegates validation to `config.validation`
- Delegates index building to `indexes.builder`
- Delegates experiment execution to `execution.runner`
- Only has thin wrapper logic

### One Minor Issue

The `_get_completed_experiment_names()` function scans for completed experiments. This is used for exclusion-aware sampling and should arguably live in `spec/sampling/` since it's sampling-related logic.

### Recommendation

Move to sampling module:

```python
# spec/sampling/exclusion.py
def get_completed_experiment_names(output_dir: Path) -> set[str]:
    """Scan output directory for completed experiments."""
    ...
```

This keeps study.py focused on CLI orchestration.

---

## Summary: Refactoring Priority

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | Split spec/builder.py | Medium | High - Most actively modified file |
| P1 | Extract encoders from EmbeddingIndex | Medium | High - Enables new encoder backends |
| P2 | Separate FAISS backend | Low | Medium - Cleaner testing |
| P3 | Fix index building chain | Low | Medium - Clearer responsibilities |
| P4 | Move VLLMEmbedder to encoders/ | Low | Low - Just file organization |

---

## What NOT to Refactor

These modules are fine as-is:

1. **experiment.py (677 lines)** - Large but cohesive. The Experiment class genuinely needs to coordinate phases, callbacks, and state.

2. **execution/runner.py and executor.py** - Complex but well-separated. Runner handles single experiments, executor handles batching.

3. **retrievers/hybrid.py (415 lines)** - Good separation. HybridRetriever correctly delegates to EmbeddingIndex and SparseIndex.

4. **analysis/visualization.py (564 lines)** - Visualization code is inherently verbose. The functions are independent and focused.

---

## Migration Strategy

### Phase 1: Spec Builder Split (Non-Breaking)
1. Create new submodules under `spec/`
2. Move functions, keeping old imports as re-exports
3. Add deprecation warnings for direct imports
4. Update internal usages

### Phase 2: Encoder Extraction (Minor Breaking)
1. Create `indexes/encoders/` with protocol
2. Wrap existing implementations
3. Update EmbeddingIndex to use protocol
4. Maintain backward compatibility in `__init__` signature

### Phase 3: FAISS Backend (Internal Only)
1. Extract FAISSBackend class
2. EmbeddingIndex uses it internally
3. No public API changes

---

## Appendix: File Size Thresholds

Recommended limits based on cognitive load research:

| File Size | Assessment | Action |
|-----------|------------|--------|
| < 300 lines | Healthy | - |
| 300-500 lines | Monitor | Consider if still cohesive |
| 500-700 lines | Warning | Plan refactoring |
| > 700 lines | Critical | Refactor soon |

Current status:
- `embedding.py`: 715 lines - CRITICAL
- `experiment.py`: 677 lines - Warning (but cohesive, acceptable)
- `visualization.py`: 564 lines - Warning (acceptable for viz)
- `spec/builder.py`: 522 lines - Warning → needs split
