# RAGiCamp Development Guidelines

This document describes the repository philosophy, patterns, and guidelines for both contributors and AI agents.

## Repository Goals

RAGiCamp is a **research framework** for experimenting with RAG approaches. It prioritizes:

1. **Simplicity** - Easy to understand, easy to modify
2. **Modularity** - Components can be swapped independently
3. **Reproducibility** - Experiments should be reproducible via configs
4. **Reliability** - Phased execution with checkpointing for long experiments

This is NOT:
- A production RAG system
- A comprehensive evaluation suite
- An MLOps platform

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli/study.py)                      │
│                    Thin orchestration layer                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Experiment (experiment.py)                    │
│           Phases: INIT → GENERATING → COMPUTING_METRICS          │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │    Agent     │    │   Dataset    │    │   Metrics    │
    │  (DirectLLM  │    │ (NQ, HotpotQA│    │ (F1, EM,     │
    │   FixedRAG)  │    │  TriviaQA)   │    │  BERTScore)  │
    └──────────────┘    └──────────────┘    └──────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌────────┐    ┌───────────┐
│ Model  │    │ Retriever │
│ (HF,   │    │ (Dense,   │
│ OpenAI)│    │  Sparse)  │
└────────┘    └───────────┘
```

### Key Principles

1. **Data Contracts** - Use typed dataclasses from `core/schemas.py`
2. **Single Source of Truth** - Each concept lives in ONE place
3. **Required Dependencies** - Don't make critical things optional
4. **Separation of Concerns** - Each layer has one job

---

## 📦 Data Contracts (CRITICAL)

All data flowing through the system MUST use these schemas from `core/schemas.py`:

### PredictionRecord
```python
from ragicamp.core.schemas import PredictionRecord, RetrievedDoc

# This is what predictions.json contains
record = PredictionRecord(
    idx=0,
    question="What is the capital of France?",
    prediction="Paris",
    expected=["Paris"],
    prompt="Answer the question: What is the capital of France?",
    retrieved_docs=[  # Only for RAG
        RetrievedDoc(rank=1, content="Paris is the capital...", score=0.95)
    ],
    metrics={"f1": 1.0, "exact_match": 1.0},
)
```

### RetrievedDoc
```python
# Structured format for retrieved documents
doc = RetrievedDoc(
    rank=1,           # Position (1-indexed)
    content="...",    # Document text
    score=0.95,       # Retrieval score (optional)
    source="wiki",    # Source identifier (optional)
)
```

### ⚠️ Don't Use Dict[str, Any]
```python
# BAD: No contract, easy to make mistakes
metadata = {"retrieved_context": "...", "num_docs": 5}

# GOOD: Typed, validated at compile time
from ragicamp.core.schemas import RAGResponseMeta
metadata = RAGResponseMeta(
    agent_type=AgentType.FIXED_RAG,
    num_docs_used=5,
    retrieved_docs=[...]
)
```

---

## 🎯 Prompt Building

### Single Source of Truth: PromptBuilder

ALL prompts MUST be built using `PromptBuilder` from `utils/prompts.py`:

```python
from ragicamp.utils.prompts import PromptBuilder

# Create builder for a prompt style + dataset
builder = PromptBuilder.from_config("fewshot", dataset="hotpotqa")

# Build prompts
direct_prompt = builder.build_direct(query="What is AI?")
rag_prompt = builder.build_rag(query="What is AI?", context="...")
```

### Prompt Structure

**Direct (no retrieval):**
```
Answer the question using your knowledge. [style instruction] If you don't know, answer 'Unknown'.

[Examples: (if fewshot)
Q: ...
A: ...]

Question: {query}
Answer:
```

**RAG (with context):**
```
Answer the question using the context below and your own knowledge. [style instruction] If you don't know, answer 'Unknown'.

[Examples: (if fewshot)]

Context:
{retrieved documents}

Question: {query}
Answer:
```

### ⚠️ Don't Build Prompts in Agents
```python
# BAD: Prompt logic scattered
class MyAgent:
    def answer(self, query):
        prompt = f"You are helpful. Answer: {query}"  # NO!

# GOOD: Use PromptBuilder
class MyAgent:
    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder  # REQUIRED
    
    def answer(self, query):
        prompt = self.prompt_builder.build_direct(query)
```

---

## 🤖 Agent Guidelines

### Required Constructor Arguments

Agents MUST accept these as **required** (not optional):

```python
class DirectLLMAgent(RAGAgent):
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        prompt_builder: PromptBuilder,  # REQUIRED
    ):
        ...

class FixedRAGAgent(RAGAgent):
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        prompt_builder: PromptBuilder,  # REQUIRED
        top_k: int = 5,
    ):
        ...
```

### RAGResponse Must Be Complete

```python
# Agents MUST return complete responses
return RAGResponse(
    answer=answer,
    prompt=prompt,  # REQUIRED: Full prompt for debugging
    context=context,
    metadata=RAGResponseMeta(
        agent_type=AgentType.FIXED_RAG,
        num_docs_used=len(docs),
        retrieved_docs=[  # REQUIRED for RAG
            RetrievedDoc(rank=i+1, content=d.text, score=d.score)
            for i, d in enumerate(docs)
        ],
    ),
)
```

---

## 📁 File Organization

```
src/ragicamp/
├── core/
│   ├── schemas.py      # ⭐ Data contracts (PredictionRecord, RetrievedDoc)
│   ├── protocols.py    # Interface definitions
│   ├── exceptions.py   # Exception hierarchy
│   ├── constants.py    # Enums and constants
│   └── logging.py      # Logging utilities
│
├── agents/
│   ├── base.py         # RAGAgent, RAGResponse, RAGContext
│   ├── direct_llm.py   # DirectLLMAgent (requires PromptBuilder)
│   └── fixed_rag.py    # FixedRAGAgent (requires PromptBuilder)
│
├── utils/
│   └── prompts.py      # ⭐ PromptBuilder (single source of truth)
│
├── execution/
│   └── executor.py     # ResilientExecutor (batch + retry logic)
│
├── experiment.py       # Experiment runner
├── experiment_state.py # Phase tracking
└── factory.py          # ComponentFactory
```

---

## ✅ Do's

### Use Data Contracts
```python
# Always use PredictionRecord for predictions
from ragicamp.core.schemas import PredictionRecord
pred = PredictionRecord(idx=0, question=q, prediction=p, ...)
```

### Use PromptBuilder
```python
# Always use PromptBuilder for prompts
builder = PromptBuilder.from_config("fewshot", dataset="nq")
prompt = builder.build_direct(query)
```

### Type Everything
```python
# Use type hints for all public APIs
def run(self, batch_size: int = 8) -> ExperimentResult:
```

### Fail Fast
```python
# Validate early
if prompt_builder is None:
    raise ValueError("prompt_builder is required")
```

---

## ❌ Don'ts

### Don't Use Dict[str, Any] for Structured Data
```python
# BAD
metadata = {"stuff": "things"}

# GOOD
metadata = RAGResponseMeta(agent_type=AgentType.DIRECT_LLM)
```

### Don't Build Prompts Inline
```python
# BAD
prompt = f"Answer: {query}"

# GOOD
prompt = self.prompt_builder.build_direct(query)
```

### Don't Make Critical Parameters Optional
```python
# BAD
def __init__(self, model=None, prompt_builder=None):

# GOOD
def __init__(self, model: LanguageModel, prompt_builder: PromptBuilder):
```

### Don't Duplicate Logic
```python
# BAD: Same prompt logic in multiple places
# study.py: get_prompt()
# agents/direct_llm.py: build_prompt()
# utils/prompts.py: PromptBuilder

# GOOD: Single source in PromptBuilder
```

### Don't Ignore Return Values
```python
# BAD: Agent ignores prompt_template parameter (what we fixed!)
def __init__(self, prompt_template=None):
    pass  # Ignored!

# GOOD: Actually use it
def __init__(self, prompt_builder: PromptBuilder):
    self.prompt_builder = prompt_builder
```

---

## 🔧 Adding New Components

### New Agent

```python
from ragicamp.agents.base import RAGAgent, RAGResponse
from ragicamp.utils.prompts import PromptBuilder
from ragicamp.core.schemas import RAGResponseMeta, AgentType

class MyAgent(RAGAgent):
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        prompt_builder: PromptBuilder,  # REQUIRED
    ):
        super().__init__(name)
        self.model = model
        self.prompt_builder = prompt_builder
    
    def answer(self, query: str, **kwargs) -> RAGResponse:
        prompt = self.prompt_builder.build_direct(query)
        answer = self.model.generate(prompt)
        
        return RAGResponse(
            answer=answer,
            prompt=prompt,  # REQUIRED
            context=RAGContext(query=query),
            metadata=RAGResponseMeta(agent_type=AgentType.DIRECT_LLM),
        )
```

### New Prompt Style

Add to `conf/prompts/fewshot_examples.yaml`:
```yaml
my_dataset:
  style: "Give a concise answer."
  stop_instruction: "Answer with just the fact."
  examples:
    - question: "Example Q1"
      answer: "Example A1"
```

Then use:
```python
builder = PromptBuilder.from_config("fewshot", dataset="my_dataset")
```

---

## 🧪 Testing

### Mock PromptBuilder
```python
from unittest.mock import Mock
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

# Create a simple mock
mock_builder = PromptBuilder(PromptConfig())
```

### Test Agents Return Complete Data
```python
def test_agent_returns_complete_response():
    response = agent.answer("What is AI?")
    
    assert response.answer is not None
    assert response.prompt is not None  # Check prompt is saved
    assert response.metadata is not None
```

---

## 📊 Experiment Output Format

### predictions.json Schema
```json
{
  "experiment": "direct_hf_gemma_default_nq",
  "predictions": [
    {
      "idx": 0,
      "question": "What is the capital of France?",
      "prediction": "Paris",
      "expected": ["Paris"],
      "prompt": "Answer the question using your knowledge...",
      "retrieved_docs": [
        {"rank": 1, "content": "Paris is...", "score": 0.95}
      ],
      "metrics": {"f1": 1.0, "exact_match": 1.0}
    }
  ],
  "aggregate_metrics": {"f1": 0.85, "exact_match": 0.80}
}
```

---

## 🚨 Common Mistakes to Avoid

1. **Ignoring constructor parameters** - If you accept a param, USE it
2. **Building prompts in multiple places** - Use PromptBuilder only
3. **Using Dict for structured data** - Use dataclasses from schemas.py
4. **Making required things optional** - If it's needed, require it
5. **Not saving the prompt** - Always include prompt in RAGResponse
6. **Saving context as string** - Use RetrievedDoc list instead
