# RAGiCamp Usage Guide

This guide shows how to use RAGiCamp for your RAG experiments.

## Installation

```bash
# Navigate to the repository
cd ragicamp

# Install with uv (recommended)
uv sync

# Install with dev dependencies
uv sync --extra dev
```

## Quick Start

### 1. Run a Study (Recommended)

The easiest way to run experiments is with study configs:

```bash
# Preview what will run
uv run ragicamp run conf/study/simple_hf.yaml --dry-run

# Run experiments
uv run ragicamp run conf/study/simple_hf.yaml --skip-existing
```

### 2. Check Experiment Status

```bash
# See health of all experiments
uv run ragicamp health outputs/simple_hf
```

### 3. Compare Results

```bash
# Compare by model
uv run ragicamp compare outputs/simple_hf --metric f1
```

## Python API

### Using ComponentFactory (Recommended)

```python
from ragicamp import Experiment, ComponentFactory
from ragicamp.metrics import F1Metric, ExactMatchMetric

# Create model
model = ComponentFactory.create_model({
    "type": "huggingface",
    "model_name": "google/gemma-2b-it",
    "load_in_4bit": True,
})

# Create agent
agent = ComponentFactory.create_agent(
    {"type": "direct_llm", "name": "my_baseline"},
    model=model,
)

# Create dataset
dataset = ComponentFactory.create_dataset({
    "name": "natural_questions",
    "split": "validation",
    "num_examples": 100,
})

# Run experiment
exp = Experiment(
    name="my_experiment",
    agent=agent,
    dataset=dataset,
    metrics=[F1Metric(), ExactMatchMetric()],
)

result = exp.run(batch_size=8)
print(f"F1: {result.f1:.3f}, EM: {result.exact_match:.3f}")
```

### Direct Component Creation

```python
from ragicamp.agents import DirectLLMAgent, FixedRAGAgent
from ragicamp.models import HuggingFaceModel, OpenAIModel
from ragicamp.retrievers import DenseRetriever
from ragicamp.datasets import NaturalQuestionsDataset

# Create model directly
model = HuggingFaceModel(
    model_name="google/gemma-2b-it",
    load_in_4bit=True,
)

# Create agent
agent = DirectLLMAgent(
    name="my_agent",
    model=model,
)

# Ask a question
response = agent.answer("What is the capital of France?")
print(response.answer)
```

### Using RAG Agent

```python
from ragicamp.agents import FixedRAGAgent
from ragicamp.retrievers import DenseRetriever

# Load pre-built index
retriever = DenseRetriever.load_index("simple_minilm_recursive_512")

# Create RAG agent
agent = FixedRAGAgent(
    name="my_rag_agent",
    model=model,
    retriever=retriever,
    top_k=5,
)

response = agent.answer("When did World War 2 end?")
print(response.answer)
print(f"Retrieved {len(response.context)} documents")
```

## Study Configuration

Create a study config to define multiple experiments:

```yaml
# conf/study/my_study.yaml
name: my_study
description: "My experiments"
num_questions: 100
datasets: [nq, triviaqa]

direct:
  enabled: true
  models:
    - hf:google/gemma-2b-it
    - openai:gpt-4o-mini
  prompts: [default, concise, fewshot]
  quantization: [4bit]

rag:
  enabled: true
  models:
    - hf:google/gemma-2b-it
  retrievers:
    - simple_minilm_recursive_512
  top_k_values: [3, 5, 10]
  prompts: [default, fewshot]
  quantization: [4bit]

metrics: [f1, exact_match, bertscore, llm_judge]

llm_judge:
  model: openai:gpt-4o-mini
  type: binary

output_dir: outputs/my_study
```

Run it:

```bash
uv run ragicamp run conf/study/my_study.yaml --skip-existing
```

## CLI Commands

### Run Study

```bash
ragicamp run <config.yaml> [OPTIONS]
  --dry-run        Preview experiments and status
  --skip-existing  Skip completed experiments
  --validate       Validate config only
```

### Check Health

```bash
ragicamp health <output_dir> [OPTIONS]
  --metrics        Comma-separated metrics to check
```

### Recompute Metrics

```bash
ragicamp metrics <exp_dir> -m <metrics>
  -m, --metrics    Required. Comma-separated metrics (f1,llm_judge)
  --judge-model    Model for LLM judge (default: gpt-4o-mini)
```

### Compare Results

```bash
ragicamp compare <output_dir> [OPTIONS]
  --metric, -m     Metric to compare (default: f1)
  --group-by, -g   Dimension to group by (model, dataset, retriever)
  --pivot A B      Create pivot table
  --top N          Show top N results
  --mlflow         Log to MLflow
```

### Build Index

```bash
ragicamp index [OPTIONS]
  --corpus         Corpus: simple, en (default: simple)
  --embedding      Embedding: minilm, e5, mpnet (default: minilm)
  --chunk-size     Chunk size in chars (default: 512)
  --max-docs       Max documents to index
```

### Evaluate Predictions

```bash
ragicamp evaluate <predictions.json> [OPTIONS]
  --metrics        Metrics to compute (f1, exact_match, llm_judge_qa)
  --output         Output file path
  --judge-model    Model for LLM judge
```

## Experiment Lifecycle

Experiments run in phases with automatic checkpointing:

```
INIT → GENERATING → GENERATED → COMPUTING_METRICS → COMPLETE
```

### Artifacts Created

| Phase | Artifacts |
|-------|-----------|
| INIT | `state.json`, `questions.json`, `metadata.json` |
| GENERATING | `predictions.json` (partial, checkpointed) |
| COMPLETE | `predictions.json` (full), `results.json` |

### Resuming Experiments

Experiments automatically resume from the last checkpoint:

```bash
# Just run again - will pick up where it left off
uv run ragicamp run conf/study/my_study.yaml

# Check what needs to be done
uv run ragicamp run conf/study/my_study.yaml --dry-run
```

## Analysis

### Load and Compare Results

```python
from ragicamp.analysis import ResultsLoader, compare_results, best_by, pivot_results

# Load results
loader = ResultsLoader("outputs/my_study")
results = loader.load_all()

# Compare by model
stats = compare_results(results, group_by="model", metric="f1")
print(stats)

# Get top 10 by F1
for r in best_by(results, metric="f1", n=10):
    print(f"{r.name}: {r.f1:.3f}")

# Create pivot table
pivot = pivot_results(results, rows="model", cols="dataset", metric="f1")
```

### MLflow Tracking

```python
from ragicamp.analysis import MLflowTracker

# Log results to MLflow
tracker = MLflowTracker("my_study")
tracker.backfill_from_results(results)

# View in MLflow UI
# $ mlflow ui
# Open http://localhost:5000
```

## Best Practices

### Memory Management

```python
from ragicamp.utils.resource_manager import ResourceManager

# Clear GPU memory between experiments
ResourceManager.clear_gpu_memory()

# Use 4-bit quantization for large models
model = HuggingFaceModel(model_name="...", load_in_4bit=True)
```

### Batch Processing

```python
# Use batch processing for faster inference
result = exp.run(batch_size=8)  # Process 8 questions at a time
```

### Checkpointing

```python
# Checkpoints are automatic, but you can control frequency
result = exp.run(checkpoint_every=50)  # Checkpoint every 50 predictions
```

### Health Checks

```python
from ragicamp import check_health

# Check before running
health = check_health("outputs/my_study/exp_name")
if health.is_complete:
    print("Already done!")
elif health.can_resume:
    print(f"Resume from: {health.resume_phase.value}")
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Use quantization
model = HuggingFaceModel(model_name="...", load_in_4bit=True)

# Reduce batch size
result = exp.run(batch_size=4)

# Clear memory between experiments
ResourceManager.clear_gpu_memory()
```

### Slow Inference

```bash
# Use batch processing
result = exp.run(batch_size=16)

# Skip completed experiments
ragicamp run config.yaml --skip-existing
```

### Experiment Crashed

```bash
# Check health
ragicamp health outputs/my_study

# Just run again - will resume automatically
ragicamp run config.yaml
```

See [Troubleshooting](TROUBLESHOOTING.md) for more solutions.
