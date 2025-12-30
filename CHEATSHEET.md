# RAGiCamp Cheatsheet

Quick reference for common tasks. For detailed docs, see `docs/`.

---

## 🚀 Quick Start (30 seconds)

```bash
# Install
uv sync

# Run quick test (dry-run first)
uv run ragicamp run conf/study/simple_hf.yaml --dry-run

# Run for real
uv run ragicamp run conf/study/simple_hf.yaml

# View results
ls outputs/
```

---

## 📋 Common Workflows

### 1. Run a Study
```bash
# Dry-run to check status
uv run ragicamp run conf/study/comprehensive_baseline.yaml --dry-run

# Run (skips completed experiments)
uv run ragicamp run conf/study/comprehensive_baseline.yaml --skip-existing

# Or via script
uv run python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml
```

### 2. Check Experiment Health
```bash
# Check all experiments in a directory
uv run ragicamp health outputs/comprehensive_baseline

# Shows: ✓ Complete, ○ Incomplete, ✗ Failed
```

### 3. Recompute Metrics Only
```bash
# Recompute specific metrics for one experiment
uv run ragicamp metrics outputs/comprehensive_baseline/my_exp -m f1,llm_judge
```

### 4. Compare Results
```bash
# Basic comparison by model
uv run ragicamp compare outputs/comprehensive_baseline

# Compare by retriever
uv run ragicamp compare outputs/comprehensive_baseline --group-by retriever

# Pivot table
uv run ragicamp compare outputs/comprehensive_baseline --pivot model dataset
```

### 5. Build Retrieval Index
```bash
# Quick (simple Wikipedia, small)
uv run ragicamp index --corpus simple --embedding minilm --chunk-size 512

# Full
uv run ragicamp index --corpus simple --embedding minilm --chunk-size 1024 --max-docs 50000
```

---

## 🎛️ CLI Commands

| Command | Description |
|---------|-------------|
| `ragicamp run <config>` | Run study from YAML config |
| `ragicamp health <dir>` | Check experiment health |
| `ragicamp resume <dir>` | Resume incomplete experiments |
| `ragicamp metrics <dir>` | Recompute metrics |
| `ragicamp compare <dir>` | Compare results |
| `ragicamp evaluate <file>` | Compute metrics on predictions file |
| `ragicamp index` | Build retrieval index |

### Run Options
```bash
ragicamp run <config.yaml> [OPTIONS]
  --dry-run        Preview experiments and their status
  --skip-existing  Skip completed experiments
  --validate       Validate config only
```

### Compare Options
```bash
ragicamp compare <output_dir> [OPTIONS]
  --metric, -m     Metric to compare (default: f1)
  --group-by, -g   Dimension to group by (model, dataset, retriever, etc.)
  --pivot A B      Create pivot table (rows=A, cols=B)
  --top N          Show top N results (default: 10)
  --mlflow         Log to MLflow
```

---

## 📁 Study Config Structure

```yaml
name: my_study
description: "Description"
num_questions: 100  # null = all
datasets: [nq, triviaqa, hotpotqa]
batch_size: 8

direct:
  enabled: true
  models:
    - hf:google/gemma-2b-it
    - openai:gpt-4o-mini
  prompts: [default, concise, fewshot]
  quantization: [4bit, 8bit]

rag:
  enabled: true
  models:
    - hf:google/gemma-2b-it
  retrievers:
    - simple_minilm_recursive_512
    - simple_minilm_recursive_1024
  top_k_values: [3, 5, 10]
  prompts: [default, fewshot]
  quantization: [4bit]

metrics: [f1, exact_match, bertscore, bleurt, llm_judge]

llm_judge:
  model: openai:gpt-4o-mini
  type: binary

output_dir: outputs/my_study
```

---

## 📊 Model Spec Format

| Provider | Format | Example |
|----------|--------|---------|
| HuggingFace | `hf:model/name` | `hf:google/gemma-2b-it` |
| OpenAI | `openai:model-name` | `openai:gpt-4o-mini` |

---

## 🔧 Quantization

| Option | Description | VRAM |
|--------|-------------|------|
| `4bit` | 4-bit quantization | Lowest |
| `8bit` | 8-bit quantization | Medium |
| `none` | Full precision | Highest |

---

## 📁 Output Structure

Each experiment creates:

```
outputs/my_study/experiment_name/
├── state.json        # Phase tracking
├── questions.json    # Exported questions
├── metadata.json     # Experiment config
├── predictions.json  # Answers + per-item metrics
└── results.json      # Final aggregate metrics
```

---

## 🐍 Python API

### Run Experiment
```python
from ragicamp import Experiment, ComponentFactory
from ragicamp.metrics import F1Metric

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
    "num_examples": 100,
})

exp = Experiment(
    name="my_exp",
    agent=agent,
    dataset=dataset,
    metrics=[F1Metric()],
)

result = exp.run(batch_size=8)
print(f"F1: {result.f1:.3f}")
```

### Check Health
```python
from ragicamp import check_health

health = check_health("outputs/my_study/exp_name")
print(health.summary())
# ✓ Complete (100 predictions, 5 metrics)
# ○ generating - predictions: 45/100
```

### Load and Compare Results
```python
from ragicamp.analysis import ResultsLoader, compare_results, best_by

loader = ResultsLoader("outputs/my_study")
results = loader.load_all()

# Top 5 by F1
for r in best_by(results, metric="f1", n=5):
    print(f"{r.name}: {r.f1:.3f}")
```

---

## 💡 Tips

### Speed Up
- Use `--dry-run` to preview
- Use `batch_size=16` for faster inference
- Use `4bit` quantization
- Reduce `num_questions` for testing

### Resume After Crash
Experiments auto-resume! Just re-run the same command.

### Recompute Only Metrics
```bash
uv run ragicamp metrics <exp_dir> -m f1,llm_judge
```

### Clear GPU Memory
```python
from ragicamp.utils.resource_manager import ResourceManager
ResourceManager.clear_gpu_memory()
```

---

## 🧪 Development

```bash
# Format
uv run ruff format src/ tests/ scripts/

# Lint
uv run ruff check src/ tests/ scripts/

# Test
uv run pytest

# Test with coverage
uv run pytest --cov=ragicamp
```

---

## 🔗 Quick Links

| Resource | Location |
|----------|----------|
| Full docs | `docs/README.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Metrics guide | `docs/guides/METRICS.md` |
| Troubleshooting | `docs/TROUBLESHOOTING.md` |

---

**That's it! Now go run some experiments.** 🚀
