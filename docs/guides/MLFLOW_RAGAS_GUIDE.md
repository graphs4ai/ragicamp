# Complete Guide: MLflow, Ragas & State Management

> **Version:** v0.2  
> **Status:** Production Ready  
> **Quick Start:** [5-minute guide](QUICKSTART_V02.md)

**Integrated experiment tracking, state-of-the-art RAG metrics, and resumable experiments.**

---

## Table of Contents

- [Overview](#overview)
- [MLflow Tracking](#mlflow-tracking)
- [Ragas Metrics](#ragas-metrics)
- [State Management](#state-management)
- [Configuration](#configuration)
- [Workflows](#workflows)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What's New in v0.2

| Feature | What It Does | Setup Required |
|---------|-------------|----------------|
| **MLflow Tracking** | Auto-track experiments, visual comparison | None (auto-enabled) |
| **Ragas Metrics** | State-of-the-art RAG evaluation | None (just add names) |
| **State Management** | Resume from failures, rerun phases | None (auto-enabled) |
| **Optuna Ready** | Hyperparameter optimization | Coming soon |

### Key Benefits

1. **Zero Configuration** - Everything works out of the box
2. **Better Insights** - Ragas provides superior RAG metrics
3. **Never Lose Work** - Phase-level resumption
4. **Professional Workflow** - MLflow UI for all experiments

---

## 🚀 Quick Start

### Enable MLflow Tracking

Add to your config:

```yaml
mlflow:
  enabled: true
  experiment_name: "rag_evaluation"
  run_name: "gemma_2b_baseline"
  tracking_uri: null  # Defaults to ./mlruns
  log_artifacts: true
```

Run your experiment:

```bash
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/my_config.yaml
```

View results in MLflow UI:

```bash
mlflow ui
# Open http://localhost:5000
```

### Use Ragas Metrics

In your config, just add Ragas metric names:

```yaml
metrics:
  - exact_match
  - f1
  - faithfulness          # Ragas metric!
  - answer_relevancy      # Ragas metric!
  - context_precision     # Ragas metric!
  - bertscore            # Custom metric (Ragas doesn't have this)
```

### Enable State Management

```yaml
evaluation:
  mode: both
  save_state: true  # Default: true
  state_file: null  # Auto-generated
  
  # Force rerun specific phases
  force_rerun_phases: []  # e.g., ["metrics"] to rerun only metrics
```

---

## 📊 MLflow Features

### 1. Automatic Experiment Tracking

Every experiment run logs:
- **Parameters**: Model name, top_k, temperature, dataset size, etc.
- **Metrics**: All evaluation metrics (EM, F1, Faithfulness, etc.)
- **Artifacts**: Predictions, configs, checkpoints
- **Tags**: Agent type, dataset, model family

### 2. Compare Experiments

MLflow UI provides:
- Side-by-side comparison
- Metric plots over time
- Parameter correlation analysis
- Best run identification

### 3. Model Registry

```python
# Models automatically versioned in MLflow
# Access via UI or API:
import mlflow

model = mlflow.pytorch.load_model("runs:/<run_id>/model")
```

### 4. Programmatic Access

```python
from ragicamp.utils import MLflowTracker

tracker = MLflowTracker(
    enabled=True,
    experiment_name="rag_experiments"
)

with tracker.start_run(run_name="test_run"):
    tracker.log_params({"model": "gemma-2b", "top_k": 5})
    tracker.log_metrics({"f1": 0.85, "exact_match": 0.72})
    tracker.log_artifact("outputs/predictions.json")
```

---

## 🎯 Ragas Metrics

### Available Metrics

| Metric | What it Measures | Best For |
|--------|------------------|----------|
| **faithfulness** | Answer grounded in context | Preventing hallucinations |
| **answer_relevancy** | Answer relevance to question | Answer quality |
| **context_precision** | Relevant context retrieved | Retrieval quality |
| **context_recall** | All needed context retrieved | Retrieval completeness |
| **answer_similarity** | Semantic similarity to reference | Semantic correctness |
| **answer_correctness** | Overall correctness | General quality |

### Usage

**In Config (Easy):**
```yaml
metrics:
  - faithfulness
  - answer_relevancy
  - exact_match  # Mix with custom metrics!
```

**Programmatic (Advanced):**
```python
from ragicamp.metrics import create_ragas_metric

# Create Ragas metric
faithfulness = create_ragas_metric("faithfulness")

# Or use adapter for custom Ragas metrics
from ragicamp.metrics import RagasMetricAdapter
from ragas.metrics import my_custom_metric

metric = RagasMetricAdapter(my_custom_metric, name="custom")
```

### Why Ragas?

✅ **Unified Interface**: Works with your existing metric code  
✅ **Better Quality**: State-of-the-art RAG metrics  
✅ **Less Code**: Delete custom faithfulness/hallucination (300+ lines)  
✅ **Maintained**: Active development by Ragas team  

### Ragas vs Custom

| Feature | Custom | Ragas |
|---------|--------|-------|
| **Faithfulness** | ❌ Use Ragas | ✅ State-of-the-art |
| **Context metrics** | ❌ Use Ragas | ✅ Comprehensive |
| **BERTScore** | ✅ Keep custom | ❌ Not available |
| **BLEURT** | ✅ Keep custom | ❌ Not available |
| **F1 Score** | ✅ Keep custom | ❌ Not available |

---

## 🔄 State Management

### Phase-Level Resumption

Your experiments now have **phases**:
1. **Generation**: Create predictions
2. **Metrics**: Compute evaluation metrics

Each phase is tracked independently!

### Benefits

**Before:**
```
Run experiment → OOM at metric computation → Lost everything → Start over 😢
```

**After:**
```
Run experiment → OOM at metrics → State saved!
Rerun with force_rerun_phases: ["metrics"] → Reuses predictions → Completes! ✅
```

### Usage

**Auto-resumption (default):**
```bash
# Run experiment
make eval-rag-wiki-simple

# If it fails, just rerun!
make eval-rag-wiki-simple
# ✓ Skips completed phases automatically
```

**Force rerun specific phase:**
```yaml
evaluation:
  force_rerun_phases: ["metrics"]  # Rerun only metrics, reuse predictions
```

**Check experiment state:**
```python
from ragicamp.utils import ExperimentState

state = ExperimentState.load("outputs/my_experiment_state.json")
print(state.summary())

# Output:
# Experiment: rag_eval
# Created: 2024-12-10T10:30:00
# 
# Phases:
#   ✅ generation: completed
#      Output: outputs/predictions.json
#   ✅ metrics: completed
#      Output: outputs/results.json
```

### State File Format

```json
{
  "name": "rag_eval",
  "phases": {
    "generation": {
      "status": "completed",
      "output_path": "outputs/predictions.json",
      "started_at": "2024-12-10T10:30:00",
      "completed_at": "2024-12-10T10:35:00",
      "metadata": {"num_examples": 100}
    },
    "metrics": {
      "status": "completed",
      "output_path": "outputs/results.json",
      "started_at": "2024-12-10T10:35:00",
      "completed_at": "2024-12-10T10:38:00",
      "metadata": {"metrics_computed": 5}
    }
  },
  "mlflow_run_id": "abc123...",
  "config": {...}
}
```

---

## 🔧 Configuration Examples

### Minimal (MLflow + Ragas)

```yaml
agent:
  type: fixed_rag
  name: "rag_baseline"

model:
  model_name: "google/gemma-2-2b-it"
  load_in_4bit: true

dataset:
  name: natural_questions
  num_examples: 100

retriever:
  artifact_path: "wikipedia_simple_chunked"

metrics:
  - exact_match
  - f1
  - faithfulness
  - answer_relevancy

# MLflow enabled by default
mlflow:
  enabled: true
  experiment_name: "rag_baseline"
```

### Advanced (Full Features)

```yaml
agent:
  type: fixed_rag
  name: "rag_optimized"
  top_k: 5

model:
  model_name: "google/gemma-2-2b-it"
  load_in_4bit: true
  max_tokens: 150
  temperature: 0.7

dataset:
  name: natural_questions
  split: validation
  num_examples: 500
  filter_no_answer: true

retriever:
  type: dense
  artifact_path: "wikipedia_large_chunked"
  embedding_model: "all-MiniLM-L6-v2"

metrics:
  - exact_match
  - f1
  - bertscore
  - faithfulness
  - answer_relevancy
  - context_precision

evaluation:
  mode: both
  batch_size: 8
  checkpoint_every: 20  # Per-question checkpointing
  resume_from_checkpoint: true
  save_state: true  # Phase-level state
  force_rerun_phases: []  # Empty = resume all

output:
  save_predictions: true
  output_path: "outputs/rag_optimized.json"
  output_dir: "outputs"

mlflow:
  enabled: true
  experiment_name: "rag_optimization"
  run_name: "gemma_2b_top5"
  tags:
    model_family: "gemma"
    dataset: "natural_questions"
  log_artifacts: true
  log_models: false  # Set true for model versioning

# Optuna (coming soon)
optuna:
  enabled: false
  n_trials: 50
  metric_to_optimize: "f1"
  direction: "maximize"
  search_params:
    top_k: [1, 20]
    temperature: [0.1, 2.0]
```

### Two-Phase with State Management

```yaml
# Phase 1: Generate predictions
evaluation:
  mode: generate
  checkpoint_every: 10
  save_state: true

output:
  output_path: "outputs/predictions.json"

mlflow:
  enabled: true
  run_name: "generation_phase"
```

Run generation:
```bash
uv run python experiments/scripts/run_experiment.py \
  --config phase1_generate.yaml
```

Then update config for phase 2:

```yaml
# Phase 2: Compute metrics
evaluation:
  mode: evaluate
  predictions_file: "outputs/predictions.json"
  save_state: true

mlflow:
  enabled: true
  run_name: "metrics_phase"
```

Run metrics:
```bash
uv run python experiments/scripts/run_experiment.py \
  --config phase2_metrics.yaml
```

**Or just force rerun:**
```yaml
evaluation:
  mode: both
  save_state: true
  force_rerun_phases: ["metrics"]  # Reuse generation, recompute metrics
```

---

## 📈 Workflows

### Workflow 1: Standard Experiment with Tracking

```bash
# 1. Create config with MLflow enabled
cat > my_exp.yaml <<EOF
agent: {type: fixed_rag, name: "test"}
model: {model_name: "google/gemma-2-2b-it", load_in_4bit: true}
dataset: {name: natural_questions, num_examples: 100}
retriever: {artifact_path: "wikipedia_simple"}
metrics: [exact_match, f1, faithfulness]
mlflow: {enabled: true, experiment_name: "my_experiments"}
EOF

# 2. Run experiment
uv run python experiments/scripts/run_experiment.py --config my_exp.yaml

# 3. View in MLflow UI
mlflow ui
# Open http://localhost:5000
```

### Workflow 2: Recover from Failure

```bash
# Run experiment
make eval-rag-wiki-simple

# It fails at metrics computation (OOM)
# State is saved automatically!

# Option 1: Just rerun (auto-resumes)
make eval-rag-wiki-simple

# Option 2: Force rerun only metrics
# Edit config: force_rerun_phases: ["metrics"]
make eval-rag-wiki-simple
```

### Workflow 3: Iterative Metric Development

```bash
# 1. Generate predictions once
uv run python run_experiment.py --config generate_only.yaml

# 2. Try different metrics (reuses predictions!)
# Update metrics list in config
uv run python run_experiment.py --config evaluate_metrics_v1.yaml

# 3. Add more metrics
# Update metrics list again
uv run python run_experiment.py --config evaluate_metrics_v2.yaml

# All runs tracked in MLflow!
```

---

## 🎓 Best Practices

### MLflow

1. **Name experiments meaningfully**
   ```yaml
   mlflow:
     experiment_name: "rag_baselines"  # Groups related runs
     run_name: "gemma_2b_top5"  # Specific run
   ```

2. **Use tags for organization**
   ```yaml
   mlflow:
     tags:
       model_family: "gemma"
       experiment_type: "baseline"
       dataset: "nq"
   ```

3. **Don't log models for quick experiments**
   ```yaml
   mlflow:
     log_models: false  # Saves time and space
   ```

### Ragas

1. **Use for RAG-specific metrics**
   - faithfulness, answer_relevancy, context_precision ✅
   
2. **Keep custom for non-RAG metrics**
   - BERTScore, BLEURT, F1 ✅

3. **Mix freely**
   ```yaml
   metrics:
     - exact_match     # Custom
     - faithfulness    # Ragas
     - bertscore       # Custom
     - answer_relevancy # Ragas
   ```

### State Management

1. **Always enable (it's default)**
   ```yaml
   evaluation:
     save_state: true  # Default, but be explicit
   ```

2. **Use force_rerun for iteration**
   ```yaml
   evaluation:
     force_rerun_phases: ["metrics"]  # Try different metrics
   ```

3. **Clean up state files after success**
   ```bash
   rm outputs/*_state.json  # After experiment completes
   ```

---

## ❓ FAQ

### Q: Does MLflow slow down experiments?
**A:** Minimal overhead (~1-2%). Logging happens in background.

### Q: Can I use Ragas without MLflow?
**A:** Yes! They're independent. Disable MLflow:
```yaml
mlflow:
  enabled: false
```

### Q: What if I don't have Ragas installed?
**A:** Graceful fallback. Ragas metrics skipped with warning.

### Q: How do I delete an MLflow experiment?
```bash
mlflow experiments delete --experiment-id <id>
```

### Q: Can I use remote MLflow server?
**A:** Yes:
```yaml
mlflow:
  tracking_uri: "http://mlflow-server:5000"
```

### Q: What happens to old checkpoints?
**A:** State files are separate from checkpoints. Both are kept for safety.

---

## 🎉 Summary

| Feature | Benefit | Setup Time |
|---------|---------|------------|
| **MLflow** | Never lose results, compare runs easily | 0 min (enabled by default) |
| **Ragas** | Better RAG metrics, less code | 0 min (just add metric names) |
| **State Management** | Resume from failures, rerun phases | 0 min (enabled by default) |
| **Optuna** | Auto-optimize parameters | Coming soon! |

**Total setup time: 0 minutes!** Everything works out of the box! ✨

---

## 🚀 Next Steps

1. **Try MLflow UI**: `mlflow ui` → http://localhost:5000
2. **Use Ragas metrics**: Add `faithfulness` to your config
3. **Test state management**: Kill a run mid-execution, restart it
4. **Read Optuna guide**: Coming soon for hyperparameter tuning

**Questions?** Open an issue or check the examples!
