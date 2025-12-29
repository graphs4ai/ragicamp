# Quick Start Guide - v0.2

**Get started with MLflow tracking, Ragas metrics, and state management in 5 minutes.**

---

## Prerequisites

```bash
# Ensure dependencies are installed
uv sync
```

## 30-Second Demo

```bash
# 1. Run example with all v0.2 features
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/example_mlflow_ragas.yaml

# 2. View results in MLflow UI
mlflow ui

# 3. Open browser to http://localhost:5000
```

**That's it!** You now have experiment tracking, Ragas metrics, and state management.

---

## What's New in v0.2

### MLflow Tracking (Auto-enabled)

**What it does:** Automatically tracks all experiments in a beautiful UI

**How to use:**
```yaml
# In your config (already enabled by default)
mlflow:
  enabled: true
  experiment_name: "my_experiments"
```

**View results:**
```bash
mlflow ui
# Open http://localhost:5000
```

### Ragas Metrics (Just add metric names)

**What it does:** State-of-the-art RAG evaluation metrics

**How to use:**
```yaml
# In your config
metrics:
  - exact_match        # Traditional
  - f1
  - faithfulness       # Ragas (NEW!)
  - answer_relevancy   # Ragas (NEW!)
  - context_precision  # Ragas (NEW!)
  - bertscore          # Still using custom
```

### State Management (Auto-enabled)

**What it does:** Resume from failures at phase level

**How to use:**
```bash
# Run experiment
make eval-rag-wiki-simple

# If it crashes, just rerun - it resumes automatically!
make eval-rag-wiki-simple
```

**Force rerun specific phase:**
```yaml
evaluation:
  force_rerun_phases: ["metrics"]  # Rerun metrics, keep predictions
```

---

## Common Tasks

### Task 1: Track Your First Experiment

```bash
# Run any experiment - tracking is automatic!
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_quick.yaml

# View in MLflow
mlflow ui
```

### Task 2: Use Ragas Metrics

```yaml
# Edit your config file
metrics:
  - exact_match
  - faithfulness      # Add this
  - answer_relevancy  # And this

# Run
uv run python experiments/scripts/run_experiment.py --config your_config.yaml
```

### Task 3: Compare Experiments

```bash
# Run baseline
uv run python run_experiment.py --config configs/baseline.yaml

# Run with RAG
uv run python run_experiment.py --config configs/rag.yaml

# Compare in MLflow UI
mlflow ui
# Click "Compare" to see side-by-side comparison
```

### Task 4: Recover from Failure

```bash
# Run long experiment
make eval-rag-wiki-simple

# It crashes? No problem!
# Just rerun - state management handles it
make eval-rag-wiki-simple
```

### Task 5: Iterate on Metrics

```yaml
# 1. First run - generates predictions
evaluation:
  mode: both
  save_state: true

# Run it
uv run python run_experiment.py --config config.yaml

# 2. Try different metrics - reuses predictions!
evaluation:
  force_rerun_phases: ["metrics"]

metrics:
  - exact_match
  - f1
  - faithfulness
  - bertscore

# Run again - skips generation, only computes new metrics
uv run python run_experiment.py --config config.yaml
```

---

## Key Commands

### MLflow
```bash
mlflow ui                        # Launch UI
mlflow experiments list          # List all experiments
mlflow runs list --experiment-id 0  # List runs in experiment
```

### State Management
```bash
# Check state
cat outputs/*_state.json | jq '.'

# Clean up
rm outputs/*_state.json
```

### Ragas
```bash
# Test Ragas is working
uv run python -c "
from ragicamp.metrics import create_ragas_metric
m = create_ragas_metric('faithfulness')
print('✓ Ragas working!')
"
```

---

## Troubleshooting

### MLflow not tracking?

```yaml
# Check config
mlflow:
  enabled: true  # Make sure it's true
```

### Ragas metric fails?

```bash
# Install Ragas
uv sync
# or
pip install ragas
```

### State file corrupted?

```bash
# Delete and restart
rm outputs/*_state.json
```

---

## Next Steps

- **Learn more:** [MLflow & Ragas Guide](MLFLOW_RAGAS_GUIDE.md)
- **See examples:** `experiments/configs/example_mlflow_ragas.yaml`
- **Read docs:** [Documentation Index](../README.md)

---

**Questions?** Check the [complete guide](MLFLOW_RAGAS_GUIDE.md) or open an issue.
