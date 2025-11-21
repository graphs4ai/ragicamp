# Robust Two-Phase Evaluation - Update Summary

**Date**: November 21, 2025  
**Status**: ✅ Complete

---

## 🎯 Problem Solved

**Before**: Evaluation runs for 2 hours generating predictions, then fails at LLM judge batch 35/57 due to API 403 error. All progress lost! 😢

**After**: Two-phase evaluation ensures predictions are saved immediately. If metrics fail, just recompute them. Never lose progress! 😊

---

## 🛡️ What Changed

### 1. Core Architecture

**New Two-Phase Design**:
```
Phase 1: Generate Predictions → Save immediately ✓
Phase 2: Compute Metrics → Can retry if it fails ✓
```

### 2. Three Evaluation Modes

```yaml
evaluation:
  mode: generate  # Generate predictions only (safest!)
  mode: evaluate  # Compute metrics only (on existing predictions)
  mode: both      # Do both (but still saves predictions first)
```

### 3. Automatic Checkpointing

- LLM judge saves progress every 5 batches
- Automatically resumes from checkpoint on failure
- Checkpoint cleaned up on success

---

## 📁 Files Changed

### Core System Files

1. **`src/ragicamp/evaluation/evaluator.py`**
   - Added `generate_predictions()` method (Phase 1)
   - Predictions always saved before metrics
   - Better error handling

2. **`src/ragicamp/metrics/llm_judge_qa.py`**
   - Added checkpointing support
   - Saves progress every 5 batches
   - Auto-resume on failure

3. **`src/ragicamp/config/schemas.py`**
   - Added `mode` field to `EvaluationConfig`
   - Added `predictions_file` field for evaluate mode
   - Added validation for modes

4. **`experiments/scripts/run_experiment.py`**
   - Split into three functions: `run_generate()`, `run_evaluate()`, `run_both()`
   - Automatic checkpoint path setup
   - Better progress reporting

### New Files

5. **`scripts/compute_metrics.py`** ⭐ **NEW**
   - Standalone script to compute metrics on saved predictions
   - Resume from checkpoint support
   - Can specify custom metrics

6. **`docs/guides/TWO_PHASE_EVALUATION.md`** ⭐ **NEW**
   - Comprehensive guide to two-phase evaluation
   - Real-world examples
   - Troubleshooting guide

### Config Examples

7. **`experiments/configs/example_generate_only.yaml`** ⭐ **NEW**
8. **`experiments/configs/example_evaluate_only.yaml`** ⭐ **NEW**
9. **`experiments/configs/example_both.yaml`** ⭐ **NEW**
10. **`experiments/configs/nq_baseline_gemma2b_quick.yaml`** (updated)

### Documentation Updates

11. **`README.md`** - Added two-phase evaluation section
12. **`WHATS_NEW.md`** - Announced new feature (#1 priority)
13. **`QUICK_REFERENCE.md`** - Added three-mode examples
14. **`DOCS_INDEX.md`** - Added two-phase guide to index

---

## 🚀 How to Use

### Recommended Workflow (Generate → Evaluate)

```bash
# Step 1: Create config with mode: generate
cat > config/my_eval.yaml << EOF
evaluation:
  mode: generate
  batch_size: 32
  num_examples: 3610
EOF

# Step 2: Generate predictions (saved automatically)
uv run python experiments/scripts/run_experiment.py \
  --config config/my_eval.yaml

# Output: outputs/agent_predictions_raw.json

# Step 3: Compute metrics (can retry if it fails!)
python scripts/compute_metrics.py \
  --predictions outputs/agent_predictions_raw.json \
  --config config/my_eval.yaml

# If LLM judge fails at batch 35? No problem!
# Just run Step 3 again - it resumes from checkpoint
```

### Quick Test (Both Mode)

```bash
# For small tests, use both mode
evaluation:
  mode: both
  num_examples: 10

# Run everything in one command
uv run python experiments/scripts/run_experiment.py --config quick_test.yaml
```

---

## 💡 Real-World Example

Your exact scenario from the error:

```bash
# You had this happen:
# - Generated 3610 predictions (50 minutes)
# - LLM judge failed at batch 35/57 with 403 error
# - Lost all progress! 😢

# Now with two-phase:

# Step 1: Generate (done once)
mode: generate
→ Takes 50 minutes
→ Saves outputs/predictions_raw.json ✓

# Step 2: Compute metrics (can retry!)
python scripts/compute_metrics.py --predictions predictions_raw.json
→ Fails at batch 35/57 with 403 error
→ Checkpoint saved! ✓

# Step 3: Just run again
python scripts/compute_metrics.py --predictions predictions_raw.json
→ Resumes from batch 35 ✓
→ Success! 😊
```

---

## 📊 Benefits

### 1. Robustness
- ✅ Never lose predictions to API failures
- ✅ Auto-checkpoint every 5 batches
- ✅ Resume from where you left off

### 2. Cost Savings
- ✅ Generate predictions once (expensive)
- ✅ Compute metrics many times (cheaper)
- ✅ Experiment with different metric combinations

### 3. Flexibility
- ✅ Try different judge models on same predictions
- ✅ Add new metrics without regenerating
- ✅ Compare metric combinations easily

### 4. Developer Experience
- ✅ Clear error messages
- ✅ Progress indicators
- ✅ Helpful recovery instructions

---

## 🔧 Technical Details

### Checkpoint Format

```json
{
  "last_batch": 35,
  "cache": {
    "prediction:::reference:::question": ["correct", 1.0],
    ...
  },
  "judgment_type": "binary",
  "batch_size": 64
}
```

Saved at: `outputs/checkpoints/{agent}_llm_judge_checkpoint.json`

### File Outputs

#### After Generate Mode:
```
outputs/
├── natural_questions_questions.json  # Reusable dataset
└── agent_predictions_raw.json        # Predictions without metrics
```

#### After Evaluate Mode:
```
outputs/
├── agent_predictions.json     # Predictions + per-question metrics
└── agent_summary.json         # Overall metrics + statistics
```

---

## 🧪 Testing

### Test the New Workflow

```bash
# 1. Test generate mode
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/example_generate_only.yaml

# Should create: outputs/gemma_2b_baseline_generate_predictions_raw.json

# 2. Test evaluate mode
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/example_evaluate_only.yaml

# Should compute metrics on the predictions

# 3. Test both mode
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/example_both.yaml

# Should do both in sequence
```

---

## 📚 Documentation

### New Documentation

1. **[docs/guides/TWO_PHASE_EVALUATION.md](docs/guides/TWO_PHASE_EVALUATION.md)**
   - Complete guide to two-phase evaluation
   - Three modes explained
   - Common workflows
   - Troubleshooting

### Updated Documentation

2. **[README.md](README.md)** - Updated with two-phase overview
3. **[WHATS_NEW.md](WHATS_NEW.md)** - Feature #1
4. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Added mode examples
5. **[DOCS_INDEX.md](DOCS_INDEX.md)** - Added to navigation

---

## ⚡ Quick Commands

```bash
# Generate predictions
uv run python experiments/scripts/run_experiment.py \
  --config config_with_mode_generate.yaml

# Compute metrics on existing predictions
python scripts/compute_metrics.py \
  --predictions outputs/predictions_raw.json \
  --config config_with_metrics.yaml

# Or specify metrics directly
python scripts/compute_metrics.py \
  --predictions outputs/predictions_raw.json \
  --metrics exact_match f1 bertscore llm_judge_qa

# Check checkpoint
ls outputs/checkpoints/
cat outputs/checkpoints/*_llm_judge_checkpoint.json

# Delete checkpoint to start fresh
rm outputs/checkpoints/*_llm_judge_checkpoint.json
```

---

## 🎓 Best Practices

### For Large Evaluations (100+ examples)
```yaml
evaluation:
  mode: generate  # Safest approach
```

### For Quick Tests (< 20 examples)
```yaml
evaluation:
  mode: both  # Convenience
```

### For Retrying/Experimenting
```yaml
evaluation:
  mode: evaluate  # Use existing predictions
  predictions_file: "outputs/predictions_raw.json"
```

---

## ✅ Backward Compatibility

**All existing code still works!**

- Old configs default to `mode: both`
- Old API still supported (but deprecated)
- No breaking changes

---

## 🎉 Summary

This update makes RAGiCamp **production-ready** and **failure-resistant**!

**Key Takeaway**: Generate predictions once (expensive), compute metrics many times (cheap). Never lose progress to API failures.

**Recommended**: Always use `mode: generate` for large evaluations.

---

**Questions?** See [docs/guides/TWO_PHASE_EVALUATION.md](docs/guides/TWO_PHASE_EVALUATION.md)

**Happy evaluating!** 🚀

