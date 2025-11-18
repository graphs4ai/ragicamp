# RAGiCamp 🏕️

A modular, production-ready framework for experimenting with Retrieval-Augmented Generation (RAG). Build, evaluate, and compare QA systems - from simple baselines to adaptive RL-based agents.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🎉 NEW!** Type-safe configs, validation, 100x faster LLM judge, and more! See **[WHATS_NEW.md](WHATS_NEW.md)** and **[CHANGELOG.md](CHANGELOG.md)**

## ✨ Key Features

- 🎯 **Multiple RAG Strategies** - DirectLLM baseline, FixedRAG, adaptive BanditRAG, and MDP-based agents
- 📊 **Comprehensive Metrics** - Standard (EM, F1), semantic (BERTScore, BLEURT), and LLM-as-a-judge evaluation
- ⚙️ **Config-Driven** - Run experiments by editing YAML configs, no code changes needed
- 💾 **Production-Ready** - Save/load trained models, artifact management, reproducible experiments
- 🔬 **Research-Friendly** - Built-in RL training, policy optimization, experiment tracking

## 🚀 Quick Start

```bash
# Install dependencies
make install

# Quick evaluation (10 examples)
make eval-baseline-quick

# Full evaluation with all metrics (100 examples)
make eval-baseline-full

# See all available commands
make help
```

## 💡 Two Ways to Use

### 1. Config-Based (Recommended) ⭐

```bash
# Create from template
make create-config OUTPUT=my_exp.yaml TYPE=baseline

# Validate config
make validate-config CONFIG=my_exp.yaml

# Run experiment
uv run python experiments/scripts/run_experiment.py \
  --config my_exp.yaml \
  --mode eval
```

**Benefits:**
- ✅ Type-safe with validation
- ✅ Reproducible experiments
- ✅ No code changes needed
- ✅ Easy to share and version

### 2. Programmatic

```python
# Clean imports from module root
from ragicamp.agents import DirectLLMAgent
from ragicamp.models import HuggingFaceModel
from ragicamp.datasets import NaturalQuestionsDataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

# Create agent
model = HuggingFaceModel('google/gemma-2-2b-it')
agent = DirectLLMAgent(name="baseline", model=model)

# Evaluate
dataset = NaturalQuestionsDataset(split="validation")
evaluator = Evaluator(agent, dataset, [ExactMatchMetric(), F1Metric()])
results = evaluator.evaluate(num_examples=100)
```

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

## 🤝 Contributing

Contributions welcome! This is a research framework designed for experimentation.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with: [HuggingFace Transformers](https://huggingface.co/transformers) • [FAISS](https://github.com/facebookresearch/faiss) • [Sentence Transformers](https://www.sbert.net/) • [BERTScore](https://github.com/Tiiiger/bert_score) • [OpenAI](https://openai.com)

---

**Ready to start?** → `make help` | **Questions?** → See [docs/](docs/) | **Quick test?** → `make eval-baseline-quick`
