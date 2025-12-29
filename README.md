# RAGiCamp

A modular framework for experimenting with RAG (Retrieval-Augmented Generation) approaches.

## Features

- **Unified Experiment API** - Single `Experiment` class for all evaluations
- **Multiple Agents** - DirectLLM (no retrieval) and FixedRAG baselines
- **Multiple Models** - HuggingFace and OpenAI model support
- **Batch Processing** - Parallel answer generation for faster experiments
- **Checkpointing** - Automatic save/resume for long experiments
- **Comprehensive Metrics** - F1, Exact Match, BERTScore, BLEURT, LLM-as-judge

## Quick Start

```bash
# Install
uv sync

# Run a study
uv run python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml

# Or use the CLI
uv run ragicamp run conf/study/comprehensive_baseline.yaml --skip-existing
```

## Project Structure

```
ragicamp/
├── src/ragicamp/          # Core library
│   ├── experiment.py      # Unified Experiment class
│   ├── agents/            # RAG agents (DirectLLM, FixedRAG)
│   ├── models/            # LLM backends (HuggingFace, OpenAI)
│   ├── retrievers/        # Dense/Sparse retrieval
│   ├── datasets/          # QA datasets (NQ, TriviaQA, HotpotQA)
│   ├── metrics/           # Evaluation metrics
│   ├── evaluation/        # Evaluator class
│   └── cli/               # Command-line interface
├── conf/                  # Configuration files
│   ├── study/             # Study configs
│   ├── model/             # Model configs
│   └── retriever/         # Retriever configs
├── scripts/               # Utility scripts
│   └── experiments/       # Experiment runners
├── artifacts/             # Saved indexes and models
└── outputs/               # Experiment results
```

## Running Experiments

### Using Python API

```python
from ragicamp import Experiment, ComponentFactory
from ragicamp.metrics import F1Metric, ExactMatchMetric

# Create components
model = ComponentFactory.create_model({
    "type": "huggingface",
    "model_name": "google/gemma-2b-it",
    "load_in_4bit": True,
})

agent = ComponentFactory.create_agent(
    {"type": "direct_llm", "name": "baseline"},
    model=model,
)

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

result = exp.run(batch_size=8, checkpoint_every=50)
print(f"F1: {result.f1:.3f}, EM: {result.exact_match:.3f}")
```

### Using Study Config

```yaml
# conf/study/my_study.yaml
name: my_study
description: "My experiment"
num_questions: 100
datasets: [nq, hotpotqa]

direct:
  enabled: true
  models:
    - hf:google/gemma-2b-it
  prompts: [default, concise]
  quantization: [4bit]

metrics: [f1, exact_match]
output_dir: outputs/my_study
```

```bash
uv run ragicamp run conf/study/my_study.yaml
```

## CLI Commands

```bash
# Run study
ragicamp run <config.yaml> [--dry-run] [--skip-existing]

# Build index
ragicamp index --corpus simple --embedding minilm --chunk-size 512

# Compare results
ragicamp compare outputs/my_study/

# Compute metrics on predictions
ragicamp evaluate predictions.json --metrics f1 exact_match
```

## Configuration

### Study Config

| Field | Description | Default |
|-------|-------------|---------|
| `name` | Study name | required |
| `num_questions` | Limit per dataset | null (all) |
| `datasets` | List of datasets | `[nq]` |
| `batch_size` | Batch size for inference | 8 |
| `metrics` | Metrics to compute | `[f1, exact_match]` |

### Model Spec Format

- HuggingFace: `hf:google/gemma-2b-it`
- OpenAI: `openai:gpt-4o-mini`

### Quantization

- `4bit` - 4-bit quantization (faster, less VRAM)
- `8bit` - 8-bit quantization
- `none` - Full precision

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Agents Guide](docs/guides/AGENTS.md)
- [Metrics Guide](docs/guides/METRICS.md)

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Format code
uv run black src/ tests/ scripts/
uv run isort src/ tests/ scripts/

# Run tests
uv run pytest

# Type check
uv run mypy src/
```

## License

MIT
