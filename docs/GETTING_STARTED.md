# Getting Started with RAGiCamp

Welcome to RAGiCamp! This guide will help you get started quickly.

## What is RAGiCamp?

RAGiCamp is a modular framework for experimenting with Retrieval-Augmented Generation (RAG) approaches. It provides:

- **Phased experiment execution** with automatic checkpointing and resume
- **Multiple agents**: DirectLLM (no retrieval) and FixedRAG baselines
- **Multiple models**: HuggingFace and OpenAI support
- **Comprehensive metrics**: F1, Exact Match, BERTScore, BLEURT, LLM-as-judge
- **Health monitoring**: Detect incomplete experiments and resume from failures

## Installation

```bash
# Navigate to the repository
cd ragicamp

# Install dependencies with uv (recommended)
uv sync

# Optional: Install with additional dependencies
uv sync --extra dev
```

## 5-Minute Quickstart

### 1. Run Your First Study (Dry Run)

```bash
# Preview what will run (no actual execution)
uv run ragicamp run conf/study/simple_hf.yaml --dry-run
```

This shows the experiment status:
- ✓ Complete experiments
- ○ Incomplete/pending experiments

### 2. Run the Study

```bash
# Run experiments (skips completed ones)
uv run ragicamp run conf/study/simple_hf.yaml --skip-existing
```

### 3. Check Experiment Health

```bash
# See status of all experiments
uv run ragicamp health outputs/simple_hf
```

### 4. Compare Results

```bash
# Compare by model
uv run ragicamp compare outputs/simple_hf --metric f1
```

## Python API Example

```python
from ragicamp import Experiment, ComponentFactory
from ragicamp.metrics import F1Metric, ExactMatchMetric

# Create components using factory
model = ComponentFactory.create_model({
    "type": "huggingface",
    "model_name": "google/gemma-2b-it",
    "load_in_4bit": True,
})

agent = ComponentFactory.create_agent(
    {"type": "direct_llm", "name": "my_baseline"},
    model=model,
)

dataset = ComponentFactory.create_dataset({
    "name": "natural_questions",
    "split": "validation",
    "num_examples": 100,
})

# Create and run experiment
exp = Experiment(
    name="my_first_experiment",
    agent=agent,
    dataset=dataset,
    metrics=[F1Metric(), ExactMatchMetric()],
)

result = exp.run(batch_size=8)
print(f"F1: {result.f1:.3f}, EM: {result.exact_match:.3f}")
```

## Core Concepts

### Experiments

Experiments run in phases, with checkpoints at each step:

```
INIT → GENERATING → GENERATED → COMPUTING_METRICS → COMPLETE
```

If an experiment crashes, it automatically resumes from the last checkpoint.

### Agents

- **DirectLLMAgent**: No retrieval, directly queries the LLM
- **FixedRAGAgent**: Retrieves context before answering

### Models

- **HuggingFaceModel**: Local models with optional quantization (4bit, 8bit)
- **OpenAIModel**: OpenAI API models

### Metrics

- **F1Metric**, **ExactMatchMetric**: Token-level metrics
- **BertScoreMetric**: Semantic similarity
- **BLEURTMetric**: Learned metric
- **LLMJudgeQAMetric**: LLM-as-judge evaluation

### Datasets

- Natural Questions (NQ)
- TriviaQA
- HotpotQA

## Study Configuration

Define experiments in YAML:

```yaml
# conf/study/my_study.yaml
name: my_study
description: "My first study"
num_questions: 100
datasets: [nq]

direct:
  enabled: true
  models:
    - hf:google/gemma-2b-it
  prompts: [default, concise]
  quantization: [4bit]

metrics: [f1, exact_match]
output_dir: outputs/my_study
```

Run it:
```bash
uv run ragicamp run conf/study/my_study.yaml
```

## Project Structure

```
ragicamp/
├── src/ragicamp/          # Core library
│   ├── experiment.py      # Phased Experiment class
│   ├── experiment_state.py # State management
│   ├── agents/            # DirectLLM, FixedRAG
│   ├── models/            # HuggingFace, OpenAI
│   ├── retrievers/        # Dense retrieval
│   ├── datasets/          # NQ, TriviaQA, HotpotQA
│   ├── metrics/           # F1, EM, BERTScore, LLM-judge
│   └── cli/               # Command-line interface
├── conf/                  # Configuration files
│   ├── study/             # Study configs
│   └── prompts/           # Few-shot examples
├── scripts/               # Utility scripts
├── artifacts/             # Saved indexes
├── outputs/               # Experiment results
└── notebooks/             # Analysis notebooks
```

## CLI Commands

```bash
# Run study
ragicamp run <config.yaml> [--dry-run] [--skip-existing]

# Check health
ragicamp health <output_dir>

# Recompute metrics
ragicamp metrics <exp_dir> -m f1,llm_judge

# Compare results
ragicamp compare <output_dir> --metric f1

# Build index
ragicamp index --corpus simple --embedding minilm
```

## Next Steps

1. **Read the docs**:
   - [Architecture](ARCHITECTURE.md) - System design
   - [Cheatsheet](../CHEATSHEET.md) - Quick reference
   - [Metrics Guide](guides/METRICS.md) - Evaluation details

2. **Run the comprehensive baseline**:
   ```bash
   uv run ragicamp run conf/study/comprehensive_baseline.yaml --dry-run
   ```

3. **Analyze results**:
   - Open `notebooks/experiment_analysis.ipynb`
   - Use `ragicamp compare` CLI

4. **Customize**:
   - Create your own study config
   - Add custom metrics
   - Extend with new models

## Common Issues

### Out of Memory (OOM)
- Use `load_in_4bit: true` for quantization
- Reduce `batch_size`

### Experiment Crashed
- Just re-run - experiments auto-resume from checkpoints
- Check health: `ragicamp health <output_dir>`

### Slow Inference
- Increase `batch_size` (e.g., 8 or 16)
- Use `--skip-existing` to avoid re-running completed experiments

Happy experimenting! 🏕️
