# RAGiCamp Architecture Review & Improvement Roadmap

**Date:** November 18, 2025  
**Codebase Size:** ~5,600 LOC (core) + ~1,700 LOC (scripts)  
**Status:** Solid foundation, needs refinement

---

## 📊 Current State Assessment

### ✅ Strengths

#### 1. **Clean Abstractions**
- Well-defined base classes (`RAGAgent`, `LanguageModel`, `Retriever`, `Metric`, `QADataset`)
- Clear separation of concerns across modules
- Template Method pattern used appropriately (datasets)

#### 2. **Good Module Organization**
```
ragicamp/
├── agents/          ✅ Well-structured
├── models/          ✅ Clean interface
├── retrievers/      ✅ Good abstraction
├── datasets/        ✅ Nice caching system
├── metrics/         ✅ Extensible
├── evaluation/      ✅ Comprehensive
└── utils/           ✅ Helpful utilities
```

#### 3. **Excellent Usability Features**
- Config-driven experiments (YAML)
- Comprehensive Makefile (40+ commands)
- Batch processing support
- Good documentation structure
- Path utilities for consistency

#### 4. **Strong Testing Infrastructure**
- Dataset caching (2-3 orders of magnitude speedup)
- Multiple evaluation modes (quick/full/cpu)
- Example scripts and notebooks

---

## 🔴 Critical Issues

### 1. **Inconsistent `__init__.py` Exports**

**Problem:** Some modules don't export their implementations

```python
# agents/__init__.py - Only exports base classes ❌
from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
# Missing: DirectLLMAgent, FixedRAGAgent, etc.

# models/__init__.py - Only exports base class ❌
from ragicamp.models.base import LanguageModel
# Missing: HuggingFaceModel, OpenAIModel

# metrics/__init__.py - Exports everything with guards ✅ GOOD!
```

**Impact:** Users have to know internal module structure
```python
# Current (bad UX):
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.models.huggingface import HuggingFaceModel

# Should be:
from ragicamp.agents import DirectLLMAgent
from ragicamp.models import HuggingFaceModel
```

### 2. **LLM Judge Architecture Bug (CRITICAL)**

**Problem:** `compute_single()` was being called 100+ times after batch `compute()`

**Status:** ⚠️ User reverted the fix - **NEEDS RE-IMPLEMENTATION**

**Solution:** Implement caching to prevent redundant API calls

### 3. **Redundant/Outdated Scripts**

**Problem:** Several scripts that don't leverage core tools well
```
experiments/scripts/
├── run_experiment.py          ✅ Good - uses everything
├── download_datasets.py        ✅ Good - clean wrapper
├── compare_baselines.py        ❌ Incomplete (hardcoded agents)
├── run_gemma2b_baseline.py     ❌ Duplicate (use configs)
├── run_fixed_rag_eval.py       ❌ Duplicate (use configs)
├── demo_new_architecture.py    ❌ Old demo?
└── index_corpus.py             ✅ Good utility
```

### 4. **Missing Type Hints**

**Problem:** Inconsistent typing across codebase
```python
# Some functions have full hints ✅
def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
    ...

# Others are missing them ❌
def process_results(self, results):  # What type is results?
    ...
```

---

## 🟡 Areas for Improvement

### 1. **Factory Pattern Needed**

**Problem:** Config-to-object instantiation is scattered in `run_experiment.py` (~400 LOC)

**Solution:** Create factory classes

```python
# Proposed: src/ragicamp/factory.py
class ComponentFactory:
    """Central factory for creating components from configs."""
    
    @staticmethod
    def create_model(config: Dict) -> LanguageModel:
        model_type = config.get("type", "huggingface")
        if model_type == "huggingface":
            return HuggingFaceModel(**config)
        elif model_type == "openai":
            return OpenAIModel(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_agent(config: Dict, model: LanguageModel, **kwargs) -> RAGAgent:
        agent_type = config["type"]
        if agent_type == "direct_llm":
            return DirectLLMAgent(model=model, **config)
        elif agent_type == "fixed_rag":
            return FixedRAGAgent(model=model, **kwargs, **config)
        # ...
    
    @staticmethod
    def create_dataset(config: Dict) -> QADataset:
        # ...
    
    @staticmethod
    def create_metrics(config: List, judge_model: Optional[LanguageModel] = None) -> List[Metric]:
        # ...
```

**Benefits:**
- Reduce `run_experiment.py` from 400+ LOC to ~100 LOC
- Easier to test
- Reusable in other scripts
- Single source of truth for instantiation logic

### 2. **Dataset Cache Bug**

**Status:** ✅ FIXED (cache paths now include subset/distractor params)

### 3. **Sparse Retriever Not Implemented**

```python
# src/ragicamp/retrievers/sparse.py
class SparseRetriever(Retriever):
    def index(self, documents: List[Document]) -> None:
        raise NotImplementedError  # ❌ No BM25 implementation
```

### 4. **No Registry System**

**Problem:** Adding new components requires modifying multiple files

**Solution:** Implement a registry pattern
```python
# Proposed: src/ragicamp/registry.py
class Registry:
    _models = {}
    _agents = {}
    _metrics = {}
    
    @classmethod
    def register_model(cls, name: str):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    # Similar for agents, metrics, datasets...

# Usage:
@Registry.register_model("my_model")
class MyCustomModel(LanguageModel):
    ...

# Then in factory:
model_class = Registry.get_model(config["type"])
return model_class(**config)
```

### 5. **Test Coverage**

**Problem:** Only 1 test file exists (`test_agents.py`)

**Missing:**
- Model tests
- Metric tests
- Dataset tests
- Retriever tests
- Integration tests

### 6. **Documentation Gaps**

**Missing:**
- API reference (auto-generated from docstrings)
- Tutorial notebooks (only 1 exists)
- Migration guides (for version upgrades)
- Contributing guide

---

## 📋 Not Yet Implemented

### Core Features (High Priority)

1. **Retrievers:**
   - ❌ BM25/sparse retriever implementation
   - ❌ Hybrid retriever (dense + sparse)
   - ❌ Reranking support
   - ❌ Query reformulation

2. **Models:**
   - ❌ Anthropic (Claude) support
   - ❌ Cohere support
   - ❌ vLLM integration (faster inference)
   - ❌ Local model optimization

3. **Datasets:**
   - ❌ MS MARCO
   - ❌ SQuAD 2.0
   - ❌ Custom corpus loader

4. **Metrics:**
   - ❌ ROUGE metrics
   - ❌ Calibration metrics
   - ❌ Cost tracking (API calls)
   - ❌ Latency metrics

5. **Agents:**
   - ❌ Multi-hop reasoning
   - ❌ Self-RAG (critique & refine)
   - ❌ Agentic RAG (tool use)

### Infrastructure (Medium Priority)

6. **Experiment Tracking:**
   - ❌ Weights & Biases integration
   - ❌ MLflow integration
   - ❌ Result comparison UI

7. **Production:**
   - ❌ API server
   - ❌ Docker containers
   - ❌ Model serving optimization
   - ❌ Caching for retrieval results

8. **Development:**
   - ❌ Pre-commit hooks
   - ❌ CI/CD pipeline
   - ❌ Auto-formatting on save
   - ❌ Type checking in CI

---

## 🎯 Recommended Action Plan

### Phase 1: Fix Critical Issues (1-2 weeks)

**Priority: Fix broken/inefficient code**

1. ✅ **Fix dataset cache paths** (DONE)
2. **Re-implement LLM judge caching**
   - Add cache to prevent redundant API calls
   - Test with 100+ examples
3. **Standardize `__init__.py` exports**
   - Make all implementations importable from module root
   - Update docs with new import style
4. **Remove redundant scripts**
   - Delete `run_gemma2b_baseline.py`, `run_fixed_rag_eval.py`
   - Update `compare_baselines.py` to use configs

### Phase 2: Core Refactoring (2-3 weeks)

**Priority: Improve architecture quality**

5. **Implement Factory Pattern**
   - Create `src/ragicamp/factory.py`
   - Refactor `run_experiment.py` to use it
   - Target: <150 LOC for main script

6. **Add Registry System**
   - Create `src/ragicamp/registry.py`
   - Register all components
   - Update factory to use registry

7. **Complete Type Hints**
   - Add hints to all public APIs
   - Enable strict mypy checking
   - Add to CI pipeline

8. **Implement BM25 Retriever**
   - Complete `sparse.py` implementation
   - Add benchmarks vs dense retriever

### Phase 3: Testing & Quality (2 weeks)

**Priority: Ensure reliability**

9. **Add Comprehensive Tests**
   ```
   tests/
   ├── test_models.py
   ├── test_agents.py
   ├── test_metrics.py
   ├── test_datasets.py
   ├── test_retrievers.py
   ├── test_factory.py
   └── integration/
       ├── test_full_pipeline.py
       └── test_configs.py
   ```
   - Target: 70%+ coverage

10. **Setup CI/CD**
    - GitHub Actions for tests
    - Auto-formatting check
    - Type checking
    - Lint checks

### Phase 4: New Features (Ongoing)

**Priority: Expand capabilities**

11. **Add Missing Providers**
    - Anthropic (Claude)
    - Cohere
    - Together AI

12. **Implement Hybrid Retriever**
    - Combine dense + sparse
    - Add reranking

13. **Add Experiment Tracking**
    - W&B integration
    - Cost tracking
    - Latency metrics

---

## 📐 Architecture Principles to Maintain

### ✅ DO

1. **Keep abstractions simple**
   - Base classes should have <5 abstract methods
   - Clear single responsibility

2. **Config-driven everything**
   - No hardcoded values
   - All hyperparameters in YAML

3. **Short scripts, powerful core**
   - Scripts should be <150 LOC
   - Delegate to core library

4. **Fail gracefully**
   - Optional dependencies with try/except
   - Clear error messages

5. **Document as you go**
   - Docstrings on all public APIs
   - Examples in docs for new features

### ❌ DON'T

1. **Don't over-engineer**
   - No abstract factories for 2 classes
   - Avoid premature optimization

2. **Don't scatter configuration**
   - One source of truth (YAML)
   - No config in multiple places

3. **Don't duplicate code**
   - Use utilities (formatting, paths, prompts)
   - Extract common patterns

4. **Don't skip testing**
   - Write tests for new features
   - Maintain >70% coverage

5. **Don't create `.md` files in root**
   - Use `docs/guides/` for documentation
   - Keep root clean

---

## 🔧 Concrete Next Steps

### Immediate (This Week)

```bash
# 1. Fix LLM judge caching
#    - Add cache dictionary to LLMJudgeQAMetric
#    - Populate in compute(), use in compute_single()

# 2. Standardize imports
#    - Update all __init__.py files
#    - Test with: python -c "from ragicamp.agents import DirectLLMAgent"

# 3. Clean up scripts
rm experiments/scripts/{run_gemma2b_baseline,run_fixed_rag_eval,demo_new_architecture}.py
```

### Short-term (Next 2 Weeks)

```python
# 4. Create factory.py
# 5. Refactor run_experiment.py
# 6. Add registry.py
# 7. Complete sparse retriever
```

### Medium-term (Next Month)

```python
# 8. Add comprehensive tests
# 9. Setup CI/CD
# 10. Add type hints everywhere
# 11. Implement hybrid retriever
```

---

## 📊 Success Metrics

### Code Quality
- ✅ All scripts <150 LOC
- ✅ Test coverage >70%
- ✅ MyPy passing with strict mode
- ✅ All public APIs documented

### Usability
- ✅ Import any component from module root
- ✅ Add new component in <20 LOC (with registry)
- ✅ Run experiment with single config file
- ✅ Clear error messages for common issues

### Performance
- ✅ LLM judge: 1 API call per batch (not N+1)
- ✅ Dataset loading: <0.1s with cache
- ✅ Batch processing: 2-5x speedup

---

## 🎓 Conclusion

**Overall Assessment:** 7.5/10

**Strengths:**
- Solid abstractions and module organization
- Excellent config system and Makefile
- Good documentation structure
- Smart caching for datasets

**Key Improvements Needed:**
1. Fix LLM judge caching (critical performance bug)
2. Standardize `__init__.py` exports (usability)
3. Add factory pattern (reduce duplication)
4. Implement registry (extensibility)
5. Add comprehensive tests (reliability)

**Philosophy:**
The codebase is on the right track. Focus on:
- **Simplicity** over cleverness
- **Consistency** in patterns
- **Utilities** over duplication
- **Tests** for confidence
- **Docs** for users

With the recommended changes, RAGiCamp will be a best-in-class RAG experimentation framework.

