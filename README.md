# RAGiCamp

A modular framework for experimenting with RAG (Retrieval-Augmented Generation) approaches.

## Features

- **Phased Experiment Execution** - INIT → GENERATING → GENERATED → COMPUTING_METRICS → COMPLETE
- **Automatic Resume** - Experiments resume from last checkpoint after crashes
- **Health Monitoring** - Check experiment status, detect incomplete runs
- **Multiple Agents** - DirectLLM (no retrieval) and FixedRAG baselines
- **Multiple Models** - HuggingFace and OpenAI model support
- **Batch Processing** - Parallel answer generation for faster experiments
- **Comprehensive Metrics** - F1, Exact Match, BERTScore, BLEURT, LLM-as-judge

## Quick Start

```bash
# Install
uv sync

# Run a study (dry-run to check status)
uv run ragicamp run conf/study/comprehensive_baseline.yaml --dry-run

# Run for real (skips completed experiments)
uv run ragicamp run conf/study/comprehensive_baseline.yaml --skip-existing

# Check health of experiments
uv run ragicamp health outputs/comprehensive_baseline
```

## Project Structure

```
ragicamp/
├── src/ragicamp/          # Core library
│   ├── experiment.py      # Phased Experiment class
│   ├── experiment_state.py # State management (phases, health)
│   ├── factory.py         # ComponentFactory
│   ├── agents/            # RAG agents (DirectLLM, FixedRAG)
│   ├── models/            # LLM backends (HuggingFace, OpenAI)
│   ├── retrievers/        # Dense retrieval (FAISS)
│   ├── datasets/          # QA datasets (NQ, TriviaQA, HotpotQA)
│   ├── metrics/           # Evaluation metrics
│   ├── analysis/          # Results loading and comparison
│   └── cli/               # Command-line interface
├── conf/                  # Configuration files
│   ├── study/             # Study configs (YAML)
│   └── prompts/           # Few-shot examples
├── scripts/               # Utility scripts
├── artifacts/             # Saved indexes
├── outputs/               # Experiment results
└── notebooks/             # Analysis notebooks
```

## Experiment Lifecycle

Each experiment goes through phases, with artifacts saved at each step:

| Phase | Artifacts | Description |
|-------|-----------|-------------|
| `INIT` | `state.json`, `questions.json`, `metadata.json` | Config saved, questions exported |
| `GENERATING` | `predictions.json` (partial) | Answers being generated |
| `GENERATED` | `predictions.json` (complete) | All predictions done |
| `COMPUTING_METRICS` | `predictions.json` + per-item metrics | Metrics computed |
| `COMPLETE` | `results.json` | Final summary |

If an experiment crashes, it resumes from the last saved state.

## CLI Commands

```bash
# Run study from config
ragicamp run <config.yaml> [--dry-run] [--skip-existing]

# Check experiment health
ragicamp health <output_dir> [--metrics f1,exact_match]

# Resume incomplete experiments
ragicamp resume <output_dir> [--dry-run]

# Recompute metrics for an experiment
ragicamp metrics <exp_dir> -m f1,llm_judge

# Build retrieval index
ragicamp index --corpus simple --embedding minilm --chunk-size 512

# Compare results
ragicamp compare <output_dir> --metric f1 --group-by model

# Compute metrics on predictions file
ragicamp evaluate predictions.json --metrics f1 exact_match llm_judge_qa
```

## Python API

```python
from ragicamp import Experiment, ComponentFactory, check_health
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

# Check health first
health = exp.check_health()
if health.is_complete:
    print("Already done!")
else:
    result = exp.run(batch_size=8)
    print(f"F1: {result.f1:.3f}, EM: {result.exact_match:.3f}")
```

## Study Config

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
  prompts: [default, concise, fewshot]
  quantization: [4bit, 8bit]

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

## Model Spec Format

- HuggingFace: `hf:google/gemma-2b-it`
- OpenAI: `openai:gpt-4o-mini`

## Quantization

- `4bit` - 4-bit quantization (faster, less VRAM)
- `8bit` - 8-bit quantization
- `none` - Full precision

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Agents Guide](docs/guides/AGENTS.md)
- [Metrics Guide](docs/guides/METRICS.md)
- [Cheatsheet](CHEATSHEET.md)

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Format code
uv run ruff format src/ tests/ scripts/

# Run tests
uv run pytest

# Type check
uv run mypy src/
```

## License

MIT
