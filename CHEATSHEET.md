# RAGiCamp Cheatsheet

Quick reference for common tasks. For detailed docs, see `docs/`.

---

## 🚀 Quick Start (30 seconds)

```bash
# Install
make setup

# Run quick test
make quick-test

# View results
ls outputs/
```

---

## 📋 Common Workflows

### 1. Quick Smoke Test
```bash
make quick-test
# or
python -m ragicamp.cli.run experiment=quick_test
```

### 2. Baseline Evaluation (DirectLLM)
```bash
make baseline
# or
python -m ragicamp.cli.run experiment=baseline
```

### 3. RAG Evaluation
```bash
# First time: index corpus
make index

# Then run
make rag
# or
python -m ragicamp.cli.run experiment=rag
```

### 4. Compare Models
```bash
python -m ragicamp.cli.run --multirun \
  model=gemma_2b_4bit,phi3 \
  experiment=baseline
```

### 5. Parameter Sweep
```bash
python -m ragicamp.cli.run --multirun \
  experiment=rag \
  agent.top_k=1,3,5,10
```

---

## 🎛️ Override Any Parameter

```bash
# Change model
python -m ragicamp.cli.run model=phi3

# Change dataset size  
python -m ragicamp.cli.run dataset.num_examples=50

# Change multiple things
python -m ragicamp.cli.run \
  model=phi3 \
  dataset=triviaqa \
  evaluation=quick \
  metrics=fast
```

---

## 📁 Available Configs

| Category | Options |
|----------|---------|
| **model** | `gemma_2b`, `gemma_2b_4bit`, `gemma_2b_8bit`, `phi3`, `llama3_8b`, `openai_gpt4`, `cpu` |
| **dataset** | `nq`, `triviaqa`, `hotpotqa` |
| **agent** | `direct_llm`, `fixed_rag`, `bandit_rag` |
| **metrics** | `fast`, `standard`, `full`, `rag` |
| **evaluation** | `quick`, `standard`, `full`, `generate_only`, `evaluate_only` |
| **experiment** | `baseline`, `rag`, `quick_test`, `model_comparison` |

---

## 📊 Metrics Presets

| Preset | Metrics | Speed |
|--------|---------|-------|
| `fast` | EM, F1 | ~1 sec |
| `standard` | EM, F1, LLM Judge | ~1 min |
| `full` | EM, F1, BERTScore, LLM Judge | ~5 min |
| `rag` | EM, F1, Faithfulness, Context Precision | ~2 min |

---

## 🔧 Data Preparation

```bash
# Download datasets
make download-nq
make download-triviaqa
make download-all

# Index corpus for RAG
make index                    # Small (10k docs, fast)
make index-wiki-simple        # Full (200k docs, slow)

# List what you have
make list-datasets
make list-artifacts
```

---

## 🧪 Testing

```bash
make test           # All tests
make test-fast      # Skip slow tests
make test-coverage  # With coverage report
```

---

## 📁 Project Structure

```
ragicamp/
├── conf/               # Hydra configs (composable)
├── scripts/            # Utility scripts
├── src/ragicamp/       # Main library
├── tests/              # Test suite
├── outputs/            # Evaluation results
├── artifacts/          # Saved indexes
└── data/               # Downloaded datasets
```

---

## 💡 Pro Tips

### Speed Up Evaluation
```bash
# Use fast metrics
python -m ragicamp.cli.run metrics=fast

# Reduce examples
python -m ragicamp.cli.run dataset.num_examples=10

# Use 4-bit quantization
python -m ragicamp.cli.run model=gemma_2b_4bit
```

### Resume Failed Runs
Checkpointing is automatic! Just rerun the same command.

### Disable MLflow
```bash
python -m ragicamp.cli.run mlflow=disabled
```

### View Full Config
```bash
python -m ragicamp.cli.run --cfg job
```

---

## 🔗 Quick Links

| Resource | Location |
|----------|----------|
| Full docs | `docs/README.md` |
| Config reference | `conf/README.md` |
| Hydra guide | `docs/guides/MLFLOW_RAGAS_GUIDE.md` |
| Troubleshooting | `docs/TROUBLESHOOTING.md` |

---

## ⚡ All Make Commands

```bash
make help              # Show all commands

# Experiments
make quick-test        # Fast smoke test
make baseline          # DirectLLM baseline
make rag              # RAG evaluation
make compare-models    # Multi-model comparison

# Data
make download-all      # Download all datasets
make index            # Index small corpus
make index-wiki-simple # Index full corpus

# Development
make test             # Run tests
make lint             # Run linters
make format           # Format code
make clean            # Clean up
```

---

**That's it! Now go run some experiments.** 🚀
