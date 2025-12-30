# RAGiCamp Architecture

This document describes the architecture and design principles of RAGiCamp.

## Core Design Principles

1. **Modularity**: Each component (agents, models, retrievers, metrics) is independent and interchangeable
2. **Phased Execution**: Experiments run in phases with checkpoints for reliability
3. **Extensibility**: Easy to add new datasets, models, agents, or metrics via factory pattern
4. **Separation of Concerns**: Generation, evaluation, and analysis are cleanly separated

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI & Study Runner                            │
│  ragicamp run | health | resume | metrics | compare | evaluate      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────────┐
│                        Experiment Layer                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Experiment (phased execution)                                 │   │
│  │   ├── Phase: INIT (save metadata, export questions)          │   │
│  │   ├── Phase: GENERATING (with checkpointing)                 │   │
│  │   ├── Phase: GENERATED (cleanup model)                       │   │
│  │   ├── Phase: COMPUTING_METRICS (per-item + aggregate)        │   │
│  │   └── Phase: COMPLETE (save results)                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │ ExperimentState  │  │ ExperimentHealth │                         │
│  │ (persistence)    │  │ (health checks)  │                         │
│  └──────────────────┘  └──────────────────┘                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────────┐
│                          Agent Layer                                 │
│  ┌────────────────┐              ┌─────────────────┐                │
│  │  DirectLLMAgent│              │  FixedRAGAgent  │                │
│  │  (no retrieval)│              │  (with context) │                │
│  └───────┬────────┘              └────────┬────────┘                │
│          │                                │                          │
│          ▼                                ▼                          │
│  ┌───────────────┐               ┌────────────────┐                 │
│  │     Model     │               │    Model +     │                 │
│  │               │               │    Retriever   │                 │
│  └───────────────┘               └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘

┌───────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│   Model Layer     │  │ Retriever Layer │  │    Metrics Layer     │
│  ┌─────────────┐  │  │  ┌───────────┐  │  │  ┌───────────────┐   │
│  │ HuggingFace │  │  │  │   Dense   │  │  │  │ F1, EM        │   │
│  │   OpenAI    │  │  │  │  (FAISS)  │  │  │  │ BERTScore     │   │
│  └─────────────┘  │  │  └───────────┘  │  │  │ BLEURT        │   │
└───────────────────┘  └─────────────────┘  │  │ LLM-as-Judge  │   │
                                            │  └───────────────────┘   │
                                            └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        Support Layers                                │
│  ┌────────────┐  ┌───────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  Datasets  │  │  Analysis     │  │  Factory   │  │   Utils    │ │
│  │  - NQ      │  │  - Loader     │  │  - Create  │  │  - Paths   │ │
│  │  - HotpotQA│  │  - Compare    │  │  - Register│  │  - Memory  │ │
│  │  - TriviaQA│  │  - Visualize  │  │            │  │  - Prompts │ │
│  └────────────┘  └───────────────┘  └────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Experiment (`ragicamp.experiment`)

The central orchestrator for running evaluations with phased execution.

- **Experiment**: Main class that coordinates agents, datasets, and metrics
- **ExperimentState**: Persistent state tracking (phase, progress, timestamps)
- **ExperimentHealth**: Health check result (missing predictions, metrics)
- **ExperimentCallbacks**: Hooks for monitoring (on_phase_start, on_batch_end, etc.)
- **ExperimentResult**: Final result container with metrics and metadata

**Key features:**
- Phased execution: INIT → GENERATING → GENERATED → COMPUTING_METRICS → COMPLETE
- Automatic checkpointing during generation
- Resume from any phase after crash
- Health checks to detect incomplete experiments

### Agents (`ragicamp.agents`)

Agents are the core decision-making components that answer questions. All agents inherit from `RAGAgent` base class.

- **DirectLLMAgent**: Baseline that directly queries LLM without retrieval
- **FixedRAGAgent**: Standard RAG that retrieves context before answering

**Key abstractions:**
- `RAGAgent.answer(query)` → `RAGResponse`
- `RAGAgent.batch_answer(queries)` → `List[RAGResponse]`
- `RAGResponse`: Contains answer, prompt, and metadata

### Models (`ragicamp.models`)

Unified interface for different LLM providers.

- **Base class**: `LanguageModel`
  - `generate(prompt)`: Text generation (single or batch)
  - `unload()`: Clean up GPU memory
  - `count_tokens(text)`: Token counting

- **Implementations**:
  - `HuggingFaceModel`: Local HF transformers models (with quantization)
  - `OpenAIModel`: OpenAI API models (with async support)

### Retrievers (`ragicamp.retrievers`)

Document retrieval systems.

- **Base class**: `Retriever`
  - `retrieve(query, top_k)`: Find relevant documents
  - `index_documents(docs)`: Index document corpus
  - `save(path)` / `load_index(path)`: Persistence

- **Implementations**:
  - `DenseRetriever`: Neural embeddings + FAISS

### Datasets (`ragicamp.datasets`)

Loaders for QA datasets.

- **Base class**: `QADataset`
  - Standard interface for all datasets
  - `QAExample` dataclass for examples

- **Implementations**: NaturalQuestions, HotpotQA, TriviaQA

### Metrics (`ragicamp.metrics`)

Evaluation metrics.

- **Base class**: `Metric`
  - `compute(predictions, references)`: Compute aggregate score
  - `get_per_item_scores()`: Per-item scores for detailed analysis

- **Implementations**:
  - `ExactMatchMetric`, `F1Metric`: Token-level metrics
  - `BertScoreMetric`: Semantic similarity
  - `BLEURTMetric`: Learned metric
  - `LLMJudgeQAMetric`: LLM-as-judge (OpenAI-based)

### Analysis (`ragicamp.analysis`)

Tools for loading and comparing experiment results.

- `ResultsLoader`: Load results from output directories
- `compare_results()`: Group and compare by dimension
- `best_by()`: Get top N results by metric
- `pivot_results()`: Create pivot tables
- `MLflowTracker`: Log results to MLflow

### Factory (`ragicamp.factory`)

Component creation with plugin support.

```python
# Create from config
model = ComponentFactory.create_model({
    "type": "huggingface",
    "model_name": "google/gemma-2b-it",
    "load_in_4bit": True,
})

# Register custom component
@ComponentFactory.register_model("anthropic")
class AnthropicModel(LanguageModel):
    ...
```

## Data Flow

### Experiment Execution Flow

```
Study Config (YAML)
       │
       ▼
┌─────────────────┐
│  Study Runner   │──▶ Builds experiment specs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Experiment    │
│                 │
│  Phase: INIT    │──▶ Save metadata, export questions
│       │         │
│       ▼         │
│  Phase: GEN     │──▶ Generate predictions (with checkpoints)
│       │         │
│       ▼         │
│  Phase: METRICS │──▶ Compute all metrics
│       │         │
│       ▼         │
│  Phase: DONE    │──▶ Save final results
└─────────────────┘
         │
         ▼
   Output Files:
   - state.json (phase tracking)
   - questions.json (exported questions)
   - predictions.json (answers + per-item metrics)
   - results.json (aggregate metrics)
   - metadata.json (experiment config)
```

### Answer Generation Flow

```
Question
    │
    ▼
Agent.answer(query)
    │
    ├──▶ [FixedRAG] Retriever.retrieve(query) ──▶ Context
    │
    ▼
Prompt Template + Question + Context
    │
    ▼
Model.generate(prompt)
    │
    ▼
RAGResponse (answer, prompt, metadata)
```

## Extension Points

Adding new components is straightforward:

1. **New Agent**: Inherit from `RAGAgent`, implement `answer()` and optionally `batch_answer()`
2. **New Model**: Inherit from `LanguageModel`, implement `generate()` and `unload()`
3. **New Retriever**: Inherit from `Retriever`, implement `retrieve()`, `index_documents()`, `save()`
4. **New Dataset**: Inherit from `QADataset`, implement `load()`
5. **New Metric**: Inherit from `Metric`, implement `compute()` and optionally `get_per_item_scores()`

Use `@ComponentFactory.register_*` decorators for automatic registration.

## Configuration

Experiments are defined via YAML configs:

```yaml
name: my_study
datasets: [nq, hotpotqa]

direct:
  enabled: true
  models: [hf:google/gemma-2b-it]
  prompts: [default, fewshot]
  quantization: [4bit]

rag:
  enabled: true
  models: [hf:google/gemma-2b-it]
  retrievers: [simple_minilm_recursive_512]
  top_k_values: [5, 10]
  quantization: [4bit]

metrics: [f1, exact_match, llm_judge]
output_dir: outputs/my_study
```

## File Organization

```
src/ragicamp/
├── experiment.py        # Experiment + Callbacks + Result
├── experiment_state.py  # ExperimentPhase, State, Health
├── factory.py           # ComponentFactory
├── agents/              # DirectLLM, FixedRAG
├── models/              # HuggingFace, OpenAI
├── retrievers/          # Dense (FAISS)
├── datasets/            # NQ, TriviaQA, HotpotQA
├── metrics/             # F1, EM, BERTScore, LLM-judge
├── analysis/            # ResultsLoader, comparison, visualization
├── corpus/              # Document chunking
├── cli/                 # CLI commands
├── core/                # Logging, exceptions, protocols
├── config/              # Pydantic schemas
├── evaluation/          # compute_metrics_from_file
└── utils/               # ResourceManager, paths, prompts
```
