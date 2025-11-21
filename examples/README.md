# RAGiCamp Examples

**Note**: For running experiments, we recommend using the **config-based approach** in `experiments/configs/` instead of these programmatic examples.

---

## 🎯 Recommended: Config-Based Approach

The easiest way to run experiments is with YAML configs:

```bash
# Quick test
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_quick.yaml

# With LLM judge
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_llm_judge.yaml
```

See `experiments/configs/` for example configs.

---

## 📚 Utility Examples

These examples demonstrate specific utilities:

### `dataset_download_example.py` ⭐
Shows how to download and cache datasets.

```bash
python examples/dataset_download_example.py
```

**What it does:**
- Downloads Natural Questions dataset
- Shows caching mechanism
- Demonstrates dataset inspection

### `filter_dataset_example.py`
Shows how to filter datasets.

```bash
python examples/filter_dataset_example.py
```

### `path_utilities_example.py`
Demonstrates path utility functions.

```bash
python examples/path_utilities_example.py
```

### `analyze_per_question_metrics.py`
Analyzes per-question metrics from evaluation results.

```bash
python examples/analyze_per_question_metrics.py
```

**Use with your results:**
```bash
python examples/analyze_per_question_metrics.py \
  outputs/your_agent_predictions.json
```

---

## 🎓 Learning Path

1. **Start here**: Read `docs/guides/TWO_PHASE_EVALUATION.md`
2. **Try examples**: Run configs from `experiments/configs/`
3. **Understand utilities**: Check utility examples in this directory
4. **Advanced**: Read `docs/ARCHITECTURE.md`

---

## 📖 Documentation

- **Quick Start**: `README.md`
- **Guides**: `docs/guides/`
- **Config Reference**: `QUICK_REFERENCE.md`
- **What's New**: `WHATS_NEW.md`

---

## 💡 Tips

### For Quick Tests

```bash
# Generate predictions (Phase 1)
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/example_generate_only.yaml

# Compute metrics (Phase 2)
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/example_evaluate_only.yaml
```

### For Analysis

```bash
# After running evaluation, analyze results
python examples/analyze_per_question_metrics.py \
  outputs/gemma_2b_baseline_predictions.json
```

---

## 🚀 Next Steps

1. Try a quick evaluation: `make eval-baseline-quick`
2. Read the two-phase evaluation guide
3. Create your own config based on examples
4. Run your experiments!

Happy evaluating! 🎉

