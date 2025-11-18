# Implementation Checklist

Quick reference for improving RAGiCamp. Check off items as you complete them.

---

## 🔥 Critical (Do First)

### 1. Fix LLM Judge Caching
- [ ] Add `_judgment_cache: Dict[str, tuple]` to `LLMJudgeQAMetric.__init__()`
- [ ] In `compute()`: populate cache with results keyed by `f"{pred}:::{ref}:::{question}"`
- [ ] In `compute_single()`: check cache first, only compute if miss
- [ ] Test with 100 examples (should see only 1 batch of API calls, not 100+)

### 2. Standardize Module Exports

**Fix these `__init__.py` files:**

```python
# src/ragicamp/agents/__init__.py
from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.agents.bandit_rag import BanditRAGAgent
from ragicamp.agents.mdp_rag import MDPRAGAgent

__all__ = [
    "RAGAgent", "RAGContext", "RAGResponse",
    "DirectLLMAgent", "FixedRAGAgent", "BanditRAGAgent", "MDPRAGAgent"
]
```

```python
# src/ragicamp/models/__init__.py
from ragicamp.models.base import LanguageModel
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.models.openai import OpenAIModel

__all__ = ["LanguageModel", "HuggingFaceModel", "OpenAIModel"]
```

```python
# src/ragicamp/datasets/__init__.py
from ragicamp.datasets.base import QADataset, QAExample
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.triviaqa import TriviaQADataset
from ragicamp.datasets.hotpotqa import HotpotQADataset

__all__ = [
    "QADataset", "QAExample",
    "NaturalQuestionsDataset", "TriviaQADataset", "HotpotQADataset"
]
```

```python
# src/ragicamp/retrievers/__init__.py
from ragicamp.retrievers.base import Retriever, Document
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.sparse import SparseRetriever  # When implemented

__all__ = ["Retriever", "Document", "DenseRetriever", "SparseRetriever"]
```

- [ ] Update `src/ragicamp/agents/__init__.py`
- [ ] Update `src/ragicamp/models/__init__.py`
- [ ] Update `src/ragicamp/datasets/__init__.py`
- [ ] Update `src/ragicamp/retrievers/__init__.py`
- [ ] Test imports work: `python -c "from ragicamp.agents import DirectLLMAgent"`
- [ ] Update documentation to use new import style

### 3. Clean Up Scripts

**Delete redundant scripts:**
- [ ] Delete `experiments/scripts/run_gemma2b_baseline.py` (use configs instead)
- [ ] Delete `experiments/scripts/run_fixed_rag_eval.py` (use configs instead)
- [ ] Delete `experiments/scripts/demo_new_architecture.py` (outdated demo)

**Update compare_baselines.py:**
- [ ] Make it config-driven instead of hardcoded
- [ ] Should load agent configs from YAML
- [ ] Should output comparison table

---

## 📦 High Priority (Week 1-2)

### 4. Implement Factory Pattern

Create `src/ragicamp/factory.py`:

```python
"""Factory for creating components from configurations."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from ragicamp.agents import RAGAgent, DirectLLMAgent, FixedRAGAgent
from ragicamp.models import LanguageModel, HuggingFaceModel, OpenAIModel
from ragicamp.datasets import QADataset, NaturalQuestionsDataset, TriviaQADataset, HotpotQADataset
from ragicamp.metrics import Metric
from ragicamp.retrievers import Retriever, DenseRetriever


class ComponentFactory:
    """Factory for creating RAGiCamp components from config dicts."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> LanguageModel:
        """Create a model from configuration."""
        # Implementation
        pass
    
    @staticmethod
    def create_agent(config: Dict[str, Any], model: LanguageModel, retriever: Optional[Retriever] = None) -> RAGAgent:
        """Create an agent from configuration."""
        # Implementation
        pass
    
    @staticmethod
    def create_dataset(config: Dict[str, Any]) -> QADataset:
        """Create a dataset from configuration."""
        # Implementation
        pass
    
    @staticmethod
    def create_metrics(config: List, judge_model: Optional[LanguageModel] = None) -> List[Metric]:
        """Create metrics from configuration."""
        # Implementation
        pass
    
    @staticmethod
    def create_retriever(config: Dict[str, Any]) -> Retriever:
        """Create a retriever from configuration."""
        # Implementation
        pass
```

Checklist:
- [ ] Create `src/ragicamp/factory.py`
- [ ] Implement `create_model()`
- [ ] Implement `create_agent()`
- [ ] Implement `create_dataset()`
- [ ] Implement `create_metrics()`
- [ ] Implement `create_retriever()`
- [ ] Add tests for factory (`tests/test_factory.py`)
- [ ] Refactor `run_experiment.py` to use factory (target: <150 LOC)

### 5. Implement Registry System

Create `src/ragicamp/registry.py`:

```python
"""Component registry for extensibility."""

from typing import Any, Callable, Dict, Type


class ComponentRegistry:
    """Registry for dynamically registering and retrieving components."""
    
    _models: Dict[str, Type] = {}
    _agents: Dict[str, Type] = {}
    _metrics: Dict[str, Type] = {}
    _datasets: Dict[str, Type] = {}
    _retrievers: Dict[str, Type] = {}
    
    @classmethod
    def register_model(cls, name: str) -> Callable:
        """Decorator to register a model class."""
        def decorator(model_class: Type) -> Type:
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str) -> Type:
        """Get a registered model class."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    # Similar methods for agents, metrics, datasets, retrievers
```

Checklist:
- [ ] Create `src/ragicamp/registry.py`
- [ ] Implement model registry
- [ ] Implement agent registry  
- [ ] Implement metric registry
- [ ] Implement dataset registry
- [ ] Implement retriever registry
- [ ] Add `@Registry.register_*` decorators to all components
- [ ] Update factory to use registry
- [ ] Add documentation on how to register custom components

### 6. Implement BM25/Sparse Retriever

Complete `src/ragicamp/retrievers/sparse.py`:

```python
"""Sparse retrieval using BM25."""

from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

from ragicamp.retrievers.base import Retriever, Document


class SparseRetriever(Retriever):
    """BM25-based sparse retrieval."""
    
    def __init__(self, name: str = "sparse", tokenizer_fn=None, **kwargs):
        super().__init__(name, **kwargs)
        self.tokenizer_fn = tokenizer_fn or (lambda x: x.lower().split())
        self.bm25 = None
        self.documents = []
    
    def index(self, documents: List[Document]) -> None:
        """Index documents with BM25."""
        self.documents = documents
        tokenized_corpus = [self.tokenizer_fn(doc.text) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using BM25."""
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")
        
        tokenized_query = self.tokenizer_fn(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.score = float(scores[idx])
            results.append(doc)
        
        return results
```

Checklist:
- [ ] Add `rank-bm25` dependency to `pyproject.toml`
- [ ] Implement `SparseRetriever` as shown above
- [ ] Add tests (`tests/test_retrievers.py`)
- [ ] Add example usage in `examples/`
- [ ] Update documentation

---

## 🧪 Testing & Quality (Week 2-3)

### 7. Add Comprehensive Tests

Create test files:

```
tests/
├── __init__.py
├── test_models.py         # Test HF and OpenAI models
├── test_agents.py         # Test all agent types (expand existing)
├── test_metrics.py        # Test all metrics
├── test_datasets.py       # Test dataset loading and caching
├── test_retrievers.py     # Test dense and sparse retrievers
├── test_factory.py        # Test component factory
├── test_registry.py       # Test registry system
└── integration/
    ├── __init__.py
    ├── test_full_pipeline.py   # End-to-end test
    └── test_configs.py         # Test all config files work
```

Checklist:
- [ ] Create `tests/test_models.py` (mock API calls)
- [ ] Expand `tests/test_agents.py` (test all agent types)
- [ ] Create `tests/test_metrics.py` (test all metrics)
- [ ] Create `tests/test_datasets.py` (test caching, filtering)
- [ ] Create `tests/test_retrievers.py` (test retrieval)
- [ ] Create `tests/test_factory.py` (test factories)
- [ ] Create `tests/test_registry.py` (test registration)
- [ ] Create `tests/integration/test_full_pipeline.py`
- [ ] Create `tests/integration/test_configs.py`
- [ ] Run `pytest --cov=ragicamp` and verify >70% coverage

### 8. Setup CI/CD

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --extra dev
      - name: Run tests
        run: uv run pytest --cov=ragicamp
      - name: Type check
        run: uv run mypy src/ragicamp
      - name: Lint
        run: |
          uv run black --check src/
          uv run isort --check src/
```

Checklist:
- [ ] Create `.github/workflows/ci.yml`
- [ ] Add pytest configuration to `pyproject.toml`
- [ ] Add mypy configuration (strict mode)
- [ ] Setup pre-commit hooks
- [ ] Test CI pipeline passes

### 9. Complete Type Hints

Go through each file and add type hints:

Checklist:
- [ ] Add type hints to `src/ragicamp/models/*.py`
- [ ] Add type hints to `src/ragicamp/agents/*.py`
- [ ] Add type hints to `src/ragicamp/metrics/*.py`
- [ ] Add type hints to `src/ragicamp/datasets/*.py`
- [ ] Add type hints to `src/ragicamp/retrievers/*.py`
- [ ] Add type hints to `src/ragicamp/evaluation/*.py`
- [ ] Add type hints to `src/ragicamp/utils/*.py`
- [ ] Run `mypy src/ragicamp --strict` and fix all errors

---

## 🚀 New Features (Week 3-4)

### 10. Add Model Providers

**Anthropic (Claude):**

Create `src/ragicamp/models/anthropic.py`:
- [ ] Implement `AnthropicModel`
- [ ] Add API key handling
- [ ] Add tests
- [ ] Update factory
- [ ] Update docs

**Cohere:**

Create `src/ragicamp/models/cohere.py`:
- [ ] Implement `CohereModel`
- [ ] Add API key handling
- [ ] Add tests
- [ ] Update factory
- [ ] Update docs

### 11. Implement Hybrid Retriever

Create `src/ragicamp/retrievers/hybrid.py`:

```python
"""Hybrid retrieval combining dense and sparse methods."""

from typing import List
from ragicamp.retrievers.base import Retriever, Document


class HybridRetriever(Retriever):
    """Combine dense and sparse retrieval with fusion."""
    
    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        alpha: float = 0.5,  # Weight for dense (1-alpha for sparse)
        **kwargs
    ):
        super().__init__(name="hybrid", **kwargs)
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve using hybrid fusion."""
        # Get results from both
        dense_results = self.dense.retrieve(query, top_k=top_k*2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k*2)
        
        # Reciprocal rank fusion
        scores = {}
        for rank, doc in enumerate(dense_results):
            scores[doc.id] = scores.get(doc.id, 0) + self.alpha / (rank + 1)
        for rank, doc in enumerate(sparse_results):
            scores[doc.id] = scores.get(doc.id, 0) + (1 - self.alpha) / (rank + 1)
        
        # Sort by combined score
        # ... implementation
```

Checklist:
- [ ] Implement `HybridRetriever`
- [ ] Add tests comparing hybrid vs dense vs sparse
- [ ] Add example usage
- [ ] Update documentation

### 12. Add Experiment Tracking

**Weights & Biases Integration:**

Create `src/ragicamp/tracking/wandb.py`:
- [ ] Implement W&B logging
- [ ] Log metrics, configs, artifacts
- [ ] Add to `run_experiment.py`
- [ ] Add example usage

**Cost Tracking:**

Add to metrics:
- [ ] Track API calls (OpenAI, Anthropic, etc.)
- [ ] Calculate costs per run
- [ ] Add to evaluation output

---

## 📚 Documentation

### 13. API Reference

- [ ] Setup Sphinx or mkdocs
- [ ] Auto-generate API docs from docstrings
- [ ] Host on GitHub Pages
- [ ] Add link to README

### 14. Tutorial Notebooks

Create notebooks:
- [ ] `notebooks/01_getting_started.ipynb`
- [ ] `notebooks/02_custom_agent.ipynb`
- [ ] `notebooks/03_custom_metric.ipynb`
- [ ] `notebooks/04_custom_dataset.ipynb`
- [ ] `notebooks/05_experiment_tracking.ipynb`

### 15. Contributing Guide

- [ ] Create `CONTRIBUTING.md`
- [ ] Document code style
- [ ] Document PR process
- [ ] Add examples of good PRs

---

## ✅ Progress Tracking

Update this section as you complete items:

- [x] Dataset cache fix
- [ ] LLM judge caching
- [ ] Module export standardization
- [ ] Script cleanup
- [ ] Factory pattern
- [ ] Registry system
- [ ] BM25 retriever
- [ ] Comprehensive tests
- [ ] CI/CD setup
- [ ] Type hints
- [ ] New model providers
- [ ] Hybrid retriever
- [ ] Experiment tracking
- [ ] API documentation
- [ ] Tutorial notebooks
- [ ] Contributing guide

**Target Completion:** End of December 2025
**Current Progress:** 1/16 (6%)

---

## 📊 Success Criteria

**When all items are complete:**

✅ Code Quality
- All public APIs have type hints
- Test coverage >70%
- MyPy passes in strict mode
- All scripts <150 LOC

✅ Usability
- Import any component from module root
- Add new component with single decorator
- Run any experiment with single config
- Clear error messages

✅ Performance
- LLM judge: O(N) API calls, not O(N²)
- Dataset loading: <0.1s with cache
- Batch processing: 2-5x speedup

✅ Extensibility
- Add new model in <50 LOC
- Add new agent in <100 LOC
- Add new metric in <80 LOC
- Add new dataset in <120 LOC

