# RAGiCamp Architecture

This document describes the technical architecture and design decisions of RAGiCamp.

## Overview

RAGiCamp follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                   │
│                    (cli/study.py, cli/main.py)                          │
│              Thin wiring layer - no business logic                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Orchestration Layer                             │
│                         (experiment.py)                                  │
│              Manages experiment lifecycle and phases                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│    Execution Layer   │ │   Data Layer     │ │   Metrics Layer      │
│  (execution/)        │ │  (datasets/)     │ │   (metrics/)         │
│  ResilientExecutor   │ │  QADataset       │ │   F1, EM, BERTScore  │
└──────────────────────┘ └──────────────────┘ └──────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent Layer                                    │
│                    (agents/direct_llm.py, agents/fixed_rag.py)          │
│              Combines Model + Retriever + PromptBuilder                  │
└─────────────────────────────────────────────────────────────────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌────────────┐ ┌────────────┐ ┌────────────────┐
│   Model    │ │ Retriever  │ │ PromptBuilder  │
│  (models/) │ │(retrievers)│ │ (utils/prompts)│
└────────────┘ └────────────┘ └────────────────┘
```

---

## Core Design Principles

### 1. Data Contracts First

All data flowing through the system uses typed dataclasses:

```python
# core/schemas.py - The single source of truth for data structures
@dataclass
class PredictionRecord:
    idx: int
    question: str
    prediction: str
    expected: List[str]
    prompt: str
    retrieved_docs: Optional[List[RetrievedDoc]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
```

**Why?**
- Prevents the "what fields does this dict have?" problem
- Catches errors at development time, not runtime
- Self-documenting code

### 2. Single Source of Truth

Each concept lives in exactly ONE place:

| Concept | Location | NOT here |
|---------|----------|----------|
| Prompt building | `utils/prompts.py` | ~~agents~~, ~~study.py~~ |
| Data schemas | `core/schemas.py` | ~~scattered dataclasses~~ |
| Component creation | `factory.py` | ~~cli scripts~~ |
| Experiment phases | `experiment_state.py` | ~~experiment.py~~ |

### 3. Composition Over Inheritance

Agents compose their functionality from injected dependencies:

```python
class FixedRAGAgent:
    def __init__(
        self,
        model: LanguageModel,      # Injected
        retriever: Retriever,       # Injected
        prompt_builder: PromptBuilder,  # Injected
    ):
        # Agent combines these, doesn't inherit from them
```

### 4. Required Dependencies Are Required

If something is needed, require it in the constructor:

```python
# BAD: Optional parameter that's actually required
def __init__(self, prompt_builder=None):
    self.prompt_builder = prompt_builder or DefaultBuilder()

# GOOD: Required parameter
def __init__(self, prompt_builder: PromptBuilder):
    self.prompt_builder = prompt_builder
```

---

## Data Flow

### Generation Phase

```
┌─────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────┐
│ Dataset │ ──▶ │ Executor  │ ──▶ │    Agent    │ ──▶ │ Model    │
│(questions)│   │ (batches) │     │ (prompts)   │     │(generate)│
└─────────┘     └───────────┘     └─────────────┘     └──────────┘
                      │                   │
                      │           ┌───────┴───────┐
                      │           ▼               ▼
                      │     ┌──────────┐   ┌─────────────┐
                      │     │Retriever │   │PromptBuilder│
                      │     └──────────┘   └─────────────┘
                      │
                      ▼
              ┌──────────────┐
              │PredictionRecord│
              │ (typed data) │
              └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │predictions.json│
              └──────────────┘
```

### Data Transformation

```
Question (str)
    │
    ▼ [Agent.answer()]
RAGResponse
    ├── answer: str
    ├── prompt: str
    └── metadata: RAGResponseMeta
           └── retrieved_docs: List[RetrievedDoc]
    │
    ▼ [Executor.execute()]
Dict (executor result)
    │
    ▼ [Experiment._save_predictions()]
PredictionRecord
    │
    ▼ [json.dump()]
predictions.json
```

---

## Key Components

### PromptBuilder (utils/prompts.py)

The single source of truth for prompt construction.

```python
builder = PromptBuilder.from_config("fewshot", dataset="hotpotqa")

# For direct (no retrieval)
prompt = builder.build_direct(query)

# For RAG (with context)
prompt = builder.build_rag(query, context)
```

**Design decisions:**
- Factory method `from_config()` for common configurations
- Separate methods for direct vs RAG
- Loads few-shot examples from YAML (cached)

### ResilientExecutor (execution/executor.py)

Handles batch processing with automatic error recovery.

```python
executor = ResilientExecutor(agent, batch_size=32, min_batch_size=1)
results = executor.execute(queries)
```

**Design decisions:**
- Automatically reduces batch size on OOM errors
- Provides checkpointing callbacks
- Returns structured results with errors marked

### Experiment (experiment.py)

Manages the experiment lifecycle through phases.

```
INIT → GENERATING → GENERATED → COMPUTING_METRICS → COMPLETE
```

**Design decisions:**
- Phase transitions are explicit
- State is persisted after each phase
- Supports resume from any phase

### ComponentFactory (factory.py)

Creates components from configuration dictionaries.

```python
model = ComponentFactory.create_model({"type": "huggingface", "model_name": "..."})
agent = ComponentFactory.create_agent({"type": "direct_llm"}, model)
```

**Design decisions:**
- Supports plugin registration for custom components
- Parses spec strings (`hf:model/name`, `openai:gpt-4`)
- Centralizes all component creation

---

## File Structure Rationale

```
src/ragicamp/
│
├── core/                 # Foundational types and utilities
│   ├── schemas.py        # ⭐ Data contracts (ALWAYS check here first)
│   ├── protocols.py      # Interface definitions
│   ├── exceptions.py     # Custom exceptions
│   ├── constants.py      # Enums, magic values
│   └── logging.py        # Logging setup
│
├── agents/               # RAG agent implementations
│   ├── base.py           # RAGAgent, RAGResponse, RAGContext
│   ├── direct_llm.py     # No retrieval baseline
│   └── fixed_rag.py      # Standard RAG pipeline
│
├── models/               # Language model backends
│   ├── base.py           # LanguageModel interface
│   ├── huggingface.py    # HuggingFace transformers
│   └── openai.py         # OpenAI API
│
├── retrievers/           # Document retrieval
│   ├── base.py           # Retriever interface
│   ├── dense.py          # FAISS-based dense retrieval
│   └── sparse.py         # BM25 sparse retrieval
│
├── datasets/             # QA datasets
│   ├── base.py           # QADataset interface
│   ├── nq.py             # Natural Questions
│   ├── triviaqa.py       # TriviaQA
│   └── hotpotqa.py       # HotpotQA
│
├── metrics/              # Evaluation metrics
│   ├── base.py           # Metric interface
│   ├── exact_match.py    # Exact match
│   ├── bertscore.py      # BERTScore
│   └── llm_judge_qa.py   # LLM-as-judge
│
├── execution/            # Batch execution
│   └── executor.py       # ResilientExecutor
│
├── utils/                # Utilities
│   ├── prompts.py        # ⭐ PromptBuilder (single source)
│   ├── formatting.py     # Context formatting
│   └── resource_manager.py # GPU memory management
│
├── experiment.py         # Experiment runner
├── experiment_state.py   # Phase tracking
└── factory.py            # Component creation
```

---

## Common Patterns

### Creating an Agent

```python
from ragicamp.utils.prompts import PromptBuilder
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.agents.direct_llm import DirectLLMAgent

# 1. Create dependencies
model = HuggingFaceModel("google/gemma-2b-it", load_in_4bit=True)
prompt_builder = PromptBuilder.from_config("fewshot", dataset="nq")

# 2. Inject into agent
agent = DirectLLMAgent(
    name="my_agent",
    model=model,
    prompt_builder=prompt_builder,  # Required!
)

# 3. Use agent
response = agent.answer("What is AI?")
```

### Saving Predictions

```python
from ragicamp.core.schemas import PredictionRecord, RetrievedDoc

# Create structured prediction
pred = PredictionRecord(
    idx=0,
    question="What is AI?",
    prediction="Artificial Intelligence",
    expected=["AI", "Artificial Intelligence"],
    prompt="Answer: What is AI?",
    retrieved_docs=[
        RetrievedDoc(rank=1, content="AI is...", score=0.9)
    ],
)

# Save to JSON
with open("predictions.json", "w") as f:
    json.dump({"predictions": [pred.to_dict()]}, f)
```

### Running an Experiment

```python
from ragicamp import Experiment

exp = Experiment(
    name="my_experiment",
    agent=agent,
    dataset=dataset,
    metrics=metrics,
    output_dir="outputs/",
)

result = exp.run(batch_size=8)
```

---

## Error Handling Strategy

### Exception Hierarchy

```
RAGiCampError (base)
├── ConfigError        # Configuration issues
├── ModelError         # Model loading/inference
├── RetrieverError     # Retrieval failures
├── EvaluationError    # Metric computation
└── RecoverableError   # Can retry with smaller batch
```

### Recovery Strategy

```python
# ResilientExecutor handles CUDA errors by reducing batch size
try:
    results = model.generate(batch)
except CUDAError:
    batch_size //= 2
    retry()
```

---

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock dependencies (models, retrievers)
- Fast, run on every commit

### Integration Tests
- Test component interactions
- Use small test datasets
- Skip GPU tests in CI

### E2E Tests
- Full experiment runs
- Manual, before releases
- Verify output formats

---

## Performance Considerations

### Batch Processing
- Use `batch_answer()` for GPU efficiency
- ResilientExecutor handles batch size tuning
- Checkpoint every N items for long runs

### Memory Management
- Models implement `unload()` method
- `ResourceManager.clear_gpu_memory()` between experiments
- Use quantization (4-bit, 8-bit) for large models

### Disk I/O
- Atomic writes (write to temp, then rename)
- JSON for human readability
- Checkpoint frequently for resumability
