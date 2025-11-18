# RAGiCamp Quick Reference - Config-Based Evaluation

## 🚀 TL;DR - One Command to Rule Them All

```bash
cd /home/gabriel_frontera_cloudwalk_io/ragicamp

# Quick test (2-3 min)
make eval-baseline-quick

# Quick test with batching (1-2 min - FASTER!)
make eval-baseline-quick-batch

# Full evaluation (20-25 min)
make eval-baseline-full

# Full with batching (10-15 min - 2X FASTER!)
make eval-baseline-full-batch

# Compare with RAG
make index-wiki-small    # Once
make eval-rag            # Then evaluate
```

---

## 📋 Available Commands

### Quick Commands

| Command | What it does | Time | Metrics |
|---------|--------------|------|---------|
| `make eval-baseline-quick` | Test on 10 examples | 2-3 min | EM, F1 |
| `make eval-baseline-full` | Full eval on 100 examples | 20-25 min | EM, F1, BERTScore, BLEURT |
| `make eval-rag` | RAG evaluation | 25-30 min | All metrics |

### Setup Commands

```bash
make install         # Install dependencies
make setup          # Full setup with BLEURT
make index-wiki-small  # Index corpus (for RAG)
```

---

## 🎛️ Switching Approaches

All controlled by **config files** in `experiments/configs/`:

### 1. Edit Config File

```yaml
# experiments/configs/my_experiment.yaml
agent:
  type: direct_llm  # Change to: fixed_rag, bandit_rag, mdp_rag

model:
  model_name: "google/gemma-2-2b-it"  # Change model
  load_in_8bit: true

dataset:
  num_examples: 100  # Change number

metrics:
  - exact_match
  - f1
  - bertscore  # Add/remove metrics
  - bleurt
```

### 2. Run Experiment

```bash
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/my_experiment.yaml \
  --mode eval
```

---

## 📊 Config Files Ready to Use

| Config | Agent | Examples | Metrics |
|--------|-------|----------|---------|
| `nq_baseline_gemma2b_quick.yaml` | DirectLLM | 10 | EM, F1 |
| `nq_baseline_gemma2b_full.yaml` | DirectLLM | 100 | All |
| `nq_fixed_rag_gemma2b.yaml` | FixedRAG | 100 | All |

---

## 📁 Output Files

Every evaluation creates 3 JSON files in `outputs/`:

```
outputs/
├── natural_questions_questions.json      # Dataset (reusable)
├── {agent_name}_predictions.json         # Predictions + per-question metrics
└── {agent_name}_summary.json             # Overall metrics + stats
```

---

## 🔄 Complete Workflow

```bash
# 1. Setup (once)
cd /home/gabriel_frontera_cloudwalk_io/ragicamp
make setup

# 2. Quick test
make eval-baseline-quick

# 3. Full baseline
make eval-baseline-full

# 4. Index corpus for RAG (once)
make index-wiki-small

# 5. RAG evaluation
make eval-rag

# 6. Compare results
ls outputs/
# - gemma_2b_baseline_quick_summary.json
# - gemma_2b_baseline_summary.json
# - gemma_2b_fixed_rag_summary.json
```

---

## 💡 Common Tasks

### Change Number of Examples
Edit config file:
```yaml
dataset:
  num_examples: 50  # Change this
```

### Use Different Model
```yaml
model:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
```

### Enable Batch Processing (2x faster!)
```yaml
evaluation:
  batch_size: 8  # Process 8 questions at once
  # Adjust based on GPU memory: 4 (8GB), 8 (16GB), 16 (24GB+)
```

### Select Metrics
```yaml
metrics:
  - exact_match
  - f1
  # Remove bertscore/bleurt for faster eval
```

### Run on CPU
```yaml
model:
  device: "cpu"
  load_in_8bit: false
```

---

## 🛠️ Utilities

### Path Utilities (Avoid FileNotFoundError)

```python
from ragicamp.utils import ensure_dir, safe_write_json

# Ensure directory exists before writing
ensure_dir("outputs/results.json")

# Or use safe_write_json (recommended)
safe_write_json(data, "outputs/results.json", indent=2)

# Setup all standard directories
from ragicamp.utils import ensure_output_dirs
ensure_output_dirs()  # Creates outputs/, artifacts/, data/, etc.
```

See: `examples/path_utilities_example.py`

---

## 📖 Documentation

- **CONFIG_BASED_EVALUATION.md** - Complete config guide
- **BASELINE_EVALUATION_GUIDE.md** - Detailed evaluation guide
- **docs/** - Full framework documentation

---

## ⚡ Makefile Commands Summary

```bash
# Setup
make install              # Install dependencies
make setup               # Full setup with BLEURT

# Evaluation (Config-Based - RECOMMENDED)
make eval-baseline-quick # Quick test
make eval-baseline-full  # Full evaluation
make eval-rag           # RAG evaluation

# Configuration Management (NEW!)
make validate-config CONFIG=experiments/configs/my.yaml  # Validate config
make validate-all-configs                                # Validate all configs
make create-config OUTPUT=my.yaml TYPE=baseline          # Create template

# Indexing
make index-wiki-small   # Index 10k articles (for testing)
make index-wiki-simple  # Index 200k articles (for production)

# Utilities
make list-artifacts     # List saved indices
make clean-outputs      # Clean output files
make help              # Show all commands
```

---

## ✅ What You Need to Know

1. **Everything is config-driven** - Change config, not code
2. **One script handles all approaches** - `run_experiment.py`
3. **Reuse configs** - Easy to compare approaches
4. **Version control friendly** - Commit configs to git
5. **Reproducible** - Same config = same results

**Key File**: `experiments/scripts/run_experiment.py`  
**Key Folder**: `experiments/configs/`

---

## 🎯 Answer to Your Original Question

> "If I want to run inference on Natural Questions using a direct question (baseline approach, no retrieval), what do I need to do?"

**Answer**: Just run this:

```bash
make eval-baseline-quick  # Test (10 examples)
# or
make eval-baseline-full   # Full (100 examples)
```

> "Do we need to implement something else?"

**Answer**: NO! Everything is implemented.

> "I need to compute BLEURT, BERTScore, and other useful metrics."

**Answer**: They're all configured in the config files and will run automatically! ✅

---

**That's it! Use configs to switch between approaches.** 🚀

