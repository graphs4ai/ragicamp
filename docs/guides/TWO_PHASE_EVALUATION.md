# Two-Phase Evaluation Guide

## Overview

RAGiCamp uses a **two-phase evaluation approach** for robustness and flexibility:

1. **Phase 1: Generate Predictions** - Generate answers and save them
2. **Phase 2: Compute Metrics** - Calculate metrics on saved predictions

This design ensures you **never lose progress** if metrics computation fails (API errors, crashes, timeouts).

---

## Why Two Phases?

### The Problem

In traditional single-phase evaluation:
```
Generate predictions → Compute metrics → Save results
         ↓                      ❌
    50 minutes            API fails at minute 100
                          
Result: Lost all work! 😢
```

### The Solution

With two-phase evaluation:
```
Phase 1: Generate predictions → Save immediately ✓
Phase 2: Compute metrics → Can retry if it fails ✓

Result: Never lose predictions! 😊
```

### Real-World Benefits

- **API failures**: LLM judge APIs can fail (403, 429, timeout) - just retry metrics
- **Cost savings**: Recompute metrics without regenerating expensive predictions
- **Experimentation**: Try different metric combinations on same predictions
- **Checkpointing**: LLM judge saves progress every 5 batches

---

## Three Modes of Operation

### 1. Generate Mode (Predictions Only)

**When to use**: Generate predictions first, compute metrics later

**Config**:
```yaml
evaluation:
  mode: generate  # Only generate predictions
  batch_size: 8
  num_examples: 100
```

**Command**:
```bash
uv run python experiments/scripts/run_experiment.py \
  --config configs/my_config.yaml
```

**Output**:
- `{agent}_predictions_raw.json` - Predictions without metrics

**Next step**:
```bash
python scripts/compute_metrics.py \
  --predictions outputs/{agent}_predictions_raw.json \
  --config configs/my_config.yaml
```

---

### 2. Evaluate Mode (Metrics Only)

**When to use**: Compute metrics on existing predictions

**Config**:
```yaml
evaluation:
  mode: evaluate  # Only compute metrics
  predictions_file: "outputs/agent_predictions_raw.json"

metrics:
  - exact_match
  - f1
  - llm_judge_qa
```

**Command**:
```bash
uv run python experiments/scripts/run_experiment.py \
  --config configs/evaluate_config.yaml
```

**Use cases**:
- Retry after API failure
- Add new metrics to existing predictions
- Compare different metrics on same predictions

---

### 3. Both Mode (Generate + Evaluate)

**When to use**: Do everything in one run (traditional approach)

**Config**:
```yaml
evaluation:
  mode: both  # Generate predictions then compute metrics
  batch_size: 8
  num_examples: 100
```

**Command**:
```bash
uv run python experiments/scripts/run_experiment.py \
  --config configs/my_config.yaml
```

**Note**: Still saves predictions first, so you can retry metrics if they fail!

---

## Common Workflows

### Workflow 1: Robust Production (Recommended)

```bash
# Step 1: Generate predictions (mode: generate)
uv run python experiments/scripts/run_experiment.py \
  --config configs/generate_only.yaml
# → Takes 50 minutes, saves predictions

# Step 2: Compute fast metrics
python scripts/compute_metrics.py \
  --predictions outputs/predictions_raw.json \
  --metrics exact_match f1 bertscore
# → Takes 5 minutes

# Step 3: Add LLM judge (can fail safely)
python scripts/compute_metrics.py \
  --predictions outputs/predictions_raw.json \
  --config configs/with_llm_judge.yaml
# → If it fails at batch 35/57, no problem!
# → Just run again, resumes from checkpoint
```

### Workflow 2: Large-Scale Evaluation

```bash
# Generate predictions for full dataset
evaluation:
  mode: generate
  num_examples: 3610  # Full validation set

# Takes hours, but only need to do once!
# Then experiment with different metrics:

# Try different judge models
python scripts/compute_metrics.py --predictions preds.json --config judge_gpt4.yaml
python scripts/compute_metrics.py --predictions preds.json --config judge_gpt4o_mini.yaml

# Try different metric combinations
python scripts/compute_metrics.py --predictions preds.json --metrics exact_match f1
python scripts/compute_metrics.py --predictions preds.json --metrics bertscore bleurt
```

### Workflow 3: Quick Iteration

```bash
# For quick tests, use both mode
evaluation:
  mode: both
  num_examples: 10

# Everything in one command
uv run python experiments/scripts/run_experiment.py --config quick_test.yaml
```

---

## LLM Judge Checkpointing

The LLM judge automatically saves progress every 5 batches.

### If Evaluation Fails

```bash
# Original run (fails at batch 35/57)
python scripts/compute_metrics.py \
  --predictions outputs/predictions.json \
  --config llm_judge_config.yaml

# Error: openai.PermissionDeniedError: Error code: 403
# Checkpoint saved at: outputs/checkpoints/agent_llm_judge_checkpoint.json

# Just run again - it resumes automatically!
python scripts/compute_metrics.py \
  --predictions outputs/predictions.json \
  --config llm_judge_config.yaml

# ✓ Resumed from batch 35, continues from there
```

### Manual Checkpoint Management

```bash
# View checkpoint
cat outputs/checkpoints/agent_llm_judge_checkpoint.json

# Delete checkpoint to start fresh
rm outputs/checkpoints/agent_llm_judge_checkpoint.json
```

---

## Configuration Examples

### Example 1: Generate Only

```yaml
# config/generate_3610.yaml
agent:
  type: direct_llm
  name: "gemma_2b_full_dataset"

model:
  type: huggingface
  model_name: "google/gemma-2-2b-it"
  load_in_8bit: true

dataset:
  name: natural_questions
  split: validation
  num_examples: 3610  # Full validation set

evaluation:
  mode: generate
  batch_size: 32

output:
  save_predictions: true
  output_path: "outputs/nq_full_predictions.json"
```

### Example 2: Evaluate Only

```yaml
# config/compute_llm_judge.yaml
evaluation:
  mode: evaluate
  predictions_file: "outputs/gemma_2b_full_dataset_predictions_raw.json"

judge_model:
  type: openai
  model_name: "gpt-4o"

metrics:
  - llm_judge_qa

output:
  save_predictions: true
  output_path: "outputs/nq_with_llm_judge.json"
```

### Example 3: Both (Classic Mode)

```yaml
# config/classic_evaluation.yaml
agent:
  type: direct_llm
  name: "gemma_2b_classic"

model:
  type: huggingface
  model_name: "google/gemma-2-2b-it"
  load_in_8bit: true

dataset:
  name: natural_questions
  split: validation
  num_examples: 100

evaluation:
  mode: both  # Do everything
  batch_size: 8

metrics:
  - exact_match
  - f1
  - bertscore
  - bleurt

output:
  save_predictions: true
  output_path: "outputs/nq_complete.json"
```

---

## File Outputs

### After Generate Mode

```
outputs/
├── natural_questions_questions.json  # Dataset (reusable)
└── agent_predictions_raw.json        # Predictions without metrics
```

### After Evaluate Mode (or Both Mode)

```
outputs/
├── natural_questions_questions.json  # Dataset (reusable)
├── agent_predictions_raw.json        # Raw predictions
├── agent_predictions.json            # Predictions with per-question metrics
└── agent_summary.json                # Overall metrics + statistics
```

### Checkpoint Files

```
outputs/checkpoints/
└── agent_llm_judge_checkpoint.json   # LLM judge progress (auto-cleanup on success)
```

---

## Best Practices

### 1. Always Use Generate Mode for Large Evaluations

```yaml
evaluation:
  mode: generate  # Safest for large datasets
  num_examples: 3610
```

### 2. Use Both Mode for Quick Tests Only

```yaml
evaluation:
  mode: both  # OK for small tests
  num_examples: 10
```

### 3. Compute Expensive Metrics Separately

```bash
# Fast metrics first
python scripts/compute_metrics.py \
  --predictions preds.json \
  --metrics exact_match f1

# Expensive metrics later
python scripts/compute_metrics.py \
  --predictions preds.json \
  --metrics bertscore bleurt llm_judge_qa
```

### 4. Use Checkpointing for LLM Judge

The system automatically checkpoints every 5 batches. No action needed!

---

## Troubleshooting

### Q: Evaluation failed halfway through

**A**: Check if predictions were saved:
```bash
ls outputs/*_predictions_raw.json
```

If yes, just compute metrics:
```bash
python scripts/compute_metrics.py --predictions outputs/predictions_raw.json --config config.yaml
```

### Q: LLM judge failed with 403 error

**A**: Checkpoint was automatically saved. Just run again:
```bash
python scripts/compute_metrics.py --predictions outputs/predictions_raw.json --config config.yaml
# Automatically resumes from checkpoint
```

### Q: Want to try different metrics

**A**: Use evaluate mode or compute_metrics.py:
```bash
python scripts/compute_metrics.py \
  --predictions outputs/predictions_raw.json \
  --metrics exact_match f1 bertscore  # Try different combinations
```

### Q: Checkpoint not resuming

**A**: Check checkpoint location:
```bash
ls outputs/checkpoints/
cat outputs/checkpoints/*_llm_judge_checkpoint.json
```

If stuck, delete and restart:
```bash
rm outputs/checkpoints/*_llm_judge_checkpoint.json
```

---

## Summary

✅ **Use `generate` mode** for large evaluations (robust to failures)  
✅ **Use `evaluate` mode** to compute metrics on existing predictions  
✅ **Use `both` mode** for quick tests only  
✅ **LLM judge checkpoints** automatically (every 5 batches)  
✅ **Never lose predictions** - always saved before metrics  

**Recommended workflow**:
```bash
# 1. Generate predictions (robust)
mode: generate → predictions_raw.json

# 2. Compute metrics (can retry)
python scripts/compute_metrics.py → predictions.json + summary.json
```

This approach is **production-ready** and **failure-resistant**! 🎉

