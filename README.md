# RAGiCamp 🏕️

A modular, production-ready framework for experimenting with Retrieval-Augmented Generation (RAG). Build, evaluate, and compare QA systems - from simple baselines to adaptive RL-based agents.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

>**🎉 NEW!** Automatic checkpointing, memory-efficient evaluation, and RL training support! See **[WHATS_NEW.md](WHATS_NEW.md)**

## ✨ Key Features

- 🛡️ **Robust Evaluation** - Automatic checkpointing, resume from failures, never lose progress
- 💾 **Memory Efficient** - Automatic GPU memory management between generation and metrics
- 🎯 **Multiple RAG Strategies** - DirectLLM baseline, FixedRAG, adaptive BanditRAG, and MDP-based agents
- 📊 **Comprehensive Metrics** - Standard (EM, F1), semantic (BERTScore, BLEURT), and LLM-as-a-judge
- ⚙️ **Config-Driven** - Run experiments by editing YAML configs, no code changes needed
- 🔬 **Research-Ready** - Built-in RL training, policy optimization, experiment tracking

## 🚀 Quick Start

```bash
# Install dependencies
make install

# Quick evaluation (10 examples)
make eval-baseline-quick

# Full evaluation (100 examples)
make eval-baseline-full

# RAG with Wikipedia
make index-wiki-small-chunked  # Index once
make eval-rag-wiki-simple      # Then evaluate

# See all commands
make help
```

## 💡 Evaluation Workflow

### Simple (One Command)

```bash
# Everything in one go - with automatic checkpointing
make eval-baseline-quick
```

### Advanced (Two-Phase for Large Runs)

```yaml
# Phase 1: Generate predictions (saved immediately)
evaluation:
  mode: generate
  checkpoint_every: 10  # Save every 10 questions

# Phase 2: Compute metrics separately (can retry if fails)
evaluation:
  mode: evaluate
  predictions_file: "outputs/predictions_raw.json"
```

**Why two-phase:**
- ✅ **Never lose progress** - Predictions saved before metrics
- ✅ **Retry on failure** - If BERTScore/LLM judge fails, just run again
- ✅ **Experiment freely** - Try different metrics on same predictions

See **[Evaluation Guide](docs/guides/CONFIG_BASED_EVALUATION.md)** for details.

## 🎯 Config-Based Workflow ⭐

```bash
# Create from template
make create-config OUTPUT=my_exp.yaml TYPE=baseline

# Validate
make validate-config CONFIG=my_exp.yaml

# Run
uv run python experiments/scripts/run_experiment.py \
  --config my_exp.yaml
```

**Edit config to change everything:**
```yaml
agent:
  type: fixed_rag  # or: direct_llm, bandit_rag, mdp_rag

model:
  model_name: "google/gemma-2-2b-it"
  load_in_4bit: true

retriever:
  artifact_path: "wikipedia_simple_chunked_1024_overlap_128"

dataset:
  num_examples: 100

metrics:
  - exact_match
  - f1
  - bertscore
```

No code changes needed! ✅

## 🏗️ Architecture

```
ragicamp/
├── src/ragicamp/           # Core framework
│   ├── agents/             # RAG strategies (DirectLLM, FixedRAG, BanditRAG, MDPRAG)
│   ├── models/             # LLM interfaces (HuggingFace, OpenAI)
│   ├── retrievers/         # Retrieval systems (Dense, Sparse)
│   ├── datasets/           # QA datasets (NQ, HotpotQA, TriviaQA)
│   ├── metrics/            # Evaluation metrics
│   ├── policies/           # Decision policies (Bandits, MDP)
│   ├── training/           # Training utilities
│   ├── evaluation/         # Evaluation utilities
│   ├── config/             # Pydantic schemas & validation (NEW!)
│   ├── factory.py          # Component instantiation (NEW!)
│   ├── registry.py         # Component registration (NEW!)
│   └── utils/              # Formatting, prompts, artifacts
├── experiments/            # Configs and scripts
├── scripts/                # CLI tools (validate, create configs)
├── docs/                   # Documentation
├── artifacts/              # Saved models and indices
└── outputs/                # Evaluation results
```

### ✨ New in Latest Version

- **Type-Safe Configs**: Pydantic schemas with validation
- **Component Factory**: Centralized component creation
- **Registry System**: Easy extensibility for custom components
- **Config Validation**: Catch errors before running experiments
- **Better Imports**: Clean `from ragicamp.agents import DirectLLMAgent`

## 🎯 Typical Workflow

### 1. Choose Your Approach

**Baseline (No RAG):**
```bash
make eval-baseline-quick  # DirectLLM agent
```

**With Retrieval:**
```bash
make index-wiki-small  # Index corpus (once)
make eval-rag          # Evaluate with retrieval
```

### 2. Select Metrics

- **Fast & Free**: Exact Match, F1
- **Semantic**: BERTScore, BLEURT
- **High-Quality**: LLM-as-a-judge (requires OpenAI API key)

### 3. Compare Results

All evaluations save 3 JSON files:
- `{dataset}_questions.json` - Questions (reusable)
- `{agent}_predictions.json` - Per-question predictions & metrics
- `{agent}_summary.json` - Overall metrics & statistics

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[What's New](WHATS_NEW.md)** ⭐ | Latest features and improvements |
| **[Changelog](CHANGELOG.md)** | Detailed version history |
| **[Quick Reference](QUICK_REFERENCE.md)** | One-page command cheat sheet |
| **[Documentation Index](docs/README.md)** | Complete documentation catalog |
| **[Config Guide](docs/guides/CONFIG_BASED_EVALUATION.md)** | How to use config files |
| **[Metrics Guide](docs/guides/METRICS.md)** | Choosing the right metrics |
| **[LLM Judge Guide](docs/guides/LLM_JUDGE.md)** | Using GPT-4 for evaluation |
| **[Architecture](docs/ARCHITECTURE.md)** | System design & components |
| **[Agents Guide](docs/AGENTS.md)** | Understanding different agents |

**📖 Navigation Tips:**
- **New here?** Start with [Quick Reference](QUICK_REFERENCE.md) for commands, then [docs/](docs/README.md) for full docs
- **Need help?** Check [docs/README.md](docs/README.md) - it's the complete documentation index
- **Specific topic?** Browse [docs/guides/](docs/guides/) for focused guides on configs, metrics, LLM judge, etc.

## 🛠️ Common Commands

```bash
# Setup
make install                    # Install dependencies
make setup                      # Full setup + verification

# Quick Evaluation
make eval-baseline-quick        # 10 examples, fast metrics
make eval-baseline-full         # 100 examples, all metrics
make eval-baseline-cpu          # CPU mode (slower)

# With LLM Judge (requires OPENAI_API_KEY)
make eval-with-llm-judge        # Binary correctness evaluation
make eval-with-llm-judge-mini   # Budget version (GPT-4o-mini)

# RAG Evaluation
make index-wiki-small           # Index corpus (once)
make eval-rag                   # Evaluate with retrieval

# Configuration Management (NEW!)
make validate-config CONFIG=my.yaml   # Validate a config file
make validate-all-configs             # Validate all configs
make create-config OUTPUT=my.yaml     # Create config template

# Utilities
make help                       # Show all commands
make list-artifacts             # List saved models/indices

# See 'make help' for complete list
```

## 🔬 What's Inside

### Agents (Answer Generation)

| Agent | Description | Best For |
|-------|-------------|----------|
| **DirectLLM** | No retrieval, direct LLM queries | Baseline, model capabilities |
| **FixedRAG** | Standard RAG with fixed parameters | Production, most use cases |
| **BanditRAG** | Learns optimal retrieval parameters | Adaptive systems, optimization |
| **MDPRAG** | Multi-step reasoning with state | Complex reasoning, research |

### Metrics (Evaluation)

| Type | Metrics | Speed | Use Case |
|------|---------|-------|----------|
| **Standard** | Exact Match, F1 | ⚡ Fast | Baseline, development |
| **Semantic** | BERTScore, BLEURT | 🐢 Slow | Research, publication |
| **LLM Judge** | GPT-4 evaluation | 💰 Paid | High-quality labels, production monitoring |

### Datasets

- **Natural Questions** - Real Google search queries
- **HotpotQA** - Multi-hop reasoning questions  
- **TriviaQA** - Trivia questions from the web

## 🎓 Use Cases

- **Research**: Experiment with different RAG strategies, publish results
- **Development**: Quickly prototype and evaluate QA systems
- **Production**: Build and deploy RAG applications with saved artifacts
- **Benchmarking**: Compare models and approaches systematically
- **Learning**: Understand RAG, RL, and QA evaluation methods

## 🧪 Testing

RAGiCamp has comprehensive unit tests covering all core functionality:

```bash
# Run all tests
make test

# Run specific test categories
make test-two-phase      # Two-phase evaluation tests
make test-checkpoint     # Checkpointing tests
make test-config         # Config validation tests

# Run with coverage
make test-coverage

# Fast tests only (skip slow ones)
make test-fast
```

**Test Coverage:**
- ✅ Two-phase evaluation system (generate → evaluate)
- ✅ LLM judge checkpointing (resume from failures)
- ✅ Config validation (all three modes)
- ✅ Metrics computation (EM, F1, etc.)
- ✅ Component factory
- ✅ Agent functionality

See **[tests/README.md](tests/README.md)** for detailed testing guide.

## 🤝 Contributing

Contributions welcome! This is a research framework designed for experimentation.

**Before contributing:**
1. Run tests: `make test`
2. Check coverage: `make test-coverage`
3. Format code: `make format`
4. Validate configs: `make validate-all-configs`

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with: [HuggingFace Transformers](https://huggingface.co/transformers) • [FAISS](https://github.com/facebookresearch/faiss) • [Sentence Transformers](https://www.sbert.net/) • [BERTScore](https://github.com/Tiiiger/bert_score) • [OpenAI](https://openai.com)

---

**Ready to start?** → `make help` | **Questions?** → See [docs/](docs/) | **Quick test?** → `make eval-baseline-quick`
