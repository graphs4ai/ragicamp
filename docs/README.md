# RAGiCamp Documentation

Complete documentation for the RAGiCamp framework.

> **🆕 New in v0.2:** MLflow tracking, Ragas metrics, and state management!  
> [Quick Start →](guides/QUICKSTART_V02.md) | [Release Notes →](V02_RELEASE_NOTES.md)

---

## 🚀 Getting Started

| Document | Description |
|----------|-------------|
| **[Quick Start v0.2](guides/QUICKSTART_V02.md)** ⭐ | 5-minute guide to new features |
| **[Getting Started](GETTING_STARTED.md)** | Installation and first steps |
| **[Quick Reference](../QUICK_REFERENCE.md)** | Command cheat sheet |

## 🆕 v0.2 Features

| Guide | Description |
|-------|-------------|
| **[MLflow & Ragas Guide](guides/MLFLOW_RAGAS_GUIDE.md)** | Experiment tracking, Ragas metrics, state management |
| **[Release Notes](V02_RELEASE_NOTES.md)** | What's new in v0.2 |

## 🏗️ Core Documentation

| Document | Description |
|----------|-------------|
| **[Architecture](ARCHITECTURE.md)** | System design and components |
| **[Agents Guide](AGENTS.md)** | Understanding different agent types |
| **[Usage Guide](USAGE.md)** | Detailed usage patterns |
| **[Troubleshooting](TROUBLESHOOTING.md)** | Common issues and solutions |

## 📖 Feature Guides

### Evaluation

| Guide | Description |
|-------|-------------|
| **[Config-Based Evaluation](guides/CONFIG_BASED_EVALUATION.md)** | Using YAML configs for experiments |
| **[Two-Phase Evaluation](guides/TWO_PHASE_EVALUATION.md)** | Separate generation and metrics |
| **[Baseline Evaluation](guides/BASELINE_EVALUATION.md)** | Evaluating without retrieval |
| **[Batch Processing](guides/BATCH_PROCESSING.md)** | Parallel evaluation |

### Metrics

| Guide | Description |
|-------|-------------|
| **[Metrics Guide](guides/METRICS.md)** | Choosing and using metrics |
| **[LLM Judge](guides/LLM_JUDGE.md)** | Using GPT-4 for evaluation |
| **[Faithfulness Metrics](guides/FAITHFULNESS_METRICS_SUMMARY.md)** | RAG-specific metrics |
| **[Normalization](guides/NORMALIZATION_GUIDE.md)** | Text normalization |

### Data & Output

| Guide | Description |
|-------|-------------|
| **[Dataset Management](guides/DATASET_MANAGEMENT.md)** | Working with datasets |
| **[Output Structure](guides/OUTPUT_STRUCTURE.md)** | Understanding outputs |
| **[Path Utilities](guides/PATH_UTILITIES.md)** | File and path management |

## 🎯 Quick Links

### New Users

- **[Quick Start v0.2](guides/QUICKSTART_V02.md)** - Try new features in 5 minutes
- **[Getting Started](GETTING_STARTED.md)** - Full installation guide
- **[Quick Reference](../QUICK_REFERENCE.md)** - Command cheat sheet

### Common Tasks

- **Tracking experiments?** → [MLflow Guide](guides/MLFLOW_RAGAS_GUIDE.md#mlflow-tracking)
- **Better RAG metrics?** → [Ragas Metrics](guides/MLFLOW_RAGAS_GUIDE.md#ragas-metrics)
- **Resume from failure?** → [State Management](guides/MLFLOW_RAGAS_GUIDE.md#state-management)
- **Using configs?** → [Config Guide](guides/CONFIG_BASED_EVALUATION.md)
- **Want LLM judge?** → [LLM Judge](guides/LLM_JUDGE.md)
- **Choosing metrics?** → [Metrics Guide](guides/METRICS.md)

### Understanding the System

- **How it works?** → [Architecture](ARCHITECTURE.md)
- **Agent types?** → [Agents Guide](AGENTS.md)
- **Having issues?** → [Troubleshooting](TROUBLESHOOTING.md)

## 💡 Documentation Structure

```
docs/
├── README.md                      # This file - documentation index
├── V02_RELEASE_NOTES.md          # What's new in v0.2
├── GETTING_STARTED.md            # Installation and setup
├── ARCHITECTURE.md               # System design
├── USAGE.md                      # Detailed usage
├── TROUBLESHOOTING.md            # Common issues
│
└── guides/                       # Feature guides
    ├── QUICKSTART_V02.md         # ⭐ v0.2 quick start
    ├── MLFLOW_RAGAS_GUIDE.md     # ⭐ MLflow, Ragas, State
    ├── CONFIG_BASED_EVALUATION.md
    ├── TWO_PHASE_EVALUATION.md
    ├── METRICS.md
    ├── LLM_JUDGE.md
    └── ... (more guides)
```

## 🤝 Contributing

Found an issue or want to improve documentation? Contributions welcome!

---

**Ready to start?** → [Quick Start v0.2](guides/QUICKSTART_V02.md) | **Full install?** → [Getting Started](GETTING_STARTED.md)
