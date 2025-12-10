# Documentation Guide

**Quick navigation to all RAGiCamp documentation.**

---

## 🚀 Start Here

| Document | Best For | Time |
|----------|----------|------|
| **[5-Min Quick Start](docs/guides/QUICKSTART_V02.md)** | Trying v0.2 features | 5 min |
| **[Quick Reference](QUICK_REFERENCE.md)** | Command lookup | 2 min |
| **[Getting Started](docs/GETTING_STARTED.md)** | First-time setup | 10 min |

---

## 🆕 v0.2 Features

### New in This Release

| Feature | Guide | Description |
|---------|-------|-------------|
| **MLflow** | [MLflow Guide](docs/guides/MLFLOW_RAGAS_GUIDE.md#mlflow-tracking) | Automatic experiment tracking |
| **Ragas** | [Ragas Guide](docs/guides/MLFLOW_RAGAS_GUIDE.md#ragas-metrics) | RAG evaluation metrics |
| **State Mgmt** | [State Guide](docs/guides/MLFLOW_RAGAS_GUIDE.md#state-management) | Phase-level resumption |

### Learn More

- **[Complete v0.2 Guide](docs/guides/MLFLOW_RAGAS_GUIDE.md)** - All features in detail
- **[Release Notes](docs/V02_RELEASE_NOTES.md)** - What changed
- **[Migration Guide](docs/V02_RELEASE_NOTES.md#migration-guide)** - Upgrading from v0.1

---

## 📖 Core Documentation

### System Overview

| Document | Description |
|----------|-------------|
| **[Architecture](docs/ARCHITECTURE.md)** | How RAGiCamp works |
| **[Agents Guide](docs/AGENTS.md)** | Agent types (DirectLLM, RAG, etc.) |
| **[Usage Patterns](docs/USAGE.md)** | Common usage patterns |

### Configuration

| Document | Description |
|----------|-------------|
| **[Config-Based Evaluation](docs/guides/CONFIG_BASED_EVALUATION.md)** | Using YAML configs |
| **[Two-Phase Evaluation](docs/guides/TWO_PHASE_EVALUATION.md)** | Separate generation/metrics |
| **[Batch Processing](docs/guides/BATCH_PROCESSING.md)** | Parallel evaluation |

### Metrics & Evaluation

| Document | Description |
|----------|-------------|
| **[Metrics Guide](docs/guides/METRICS.md)** | Choosing metrics |
| **[LLM Judge](docs/guides/LLM_JUDGE.md)** | GPT-4 evaluation |
| **[Faithfulness](docs/guides/FAITHFULNESS_METRICS_SUMMARY.md)** | RAG-specific metrics |
| **[Normalization](docs/guides/NORMALIZATION_GUIDE.md)** | Text preprocessing |

### Data Management

| Document | Description |
|----------|-------------|
| **[Dataset Management](docs/guides/DATASET_MANAGEMENT.md)** | Working with datasets |
| **[Output Structure](docs/guides/OUTPUT_STRUCTURE.md)** | Understanding outputs |
| **[Path Utilities](docs/guides/PATH_UTILITIES.md)** | File management |

### Help & Troubleshooting

| Document | Description |
|----------|-------------|
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues |
| **[Matplotlib Issues](docs/guides/TROUBLESHOOTING_MATPLOTLIB.md)** | Display problems |

---

## 🎯 By Task

### "I want to..."

#### Track Experiments
→ [MLflow Guide](docs/guides/MLFLOW_RAGAS_GUIDE.md#mlflow-tracking)

#### Use Better RAG Metrics
→ [Ragas Metrics](docs/guides/MLFLOW_RAGAS_GUIDE.md#ragas-metrics)

#### Resume from Failures
→ [State Management](docs/guides/MLFLOW_RAGAS_GUIDE.md#state-management)

#### Run My First Experiment
→ [Quick Start](docs/guides/QUICKSTART_V02.md)

#### Evaluate Without RAG
→ [Baseline Evaluation](docs/guides/BASELINE_EVALUATION.md)

#### Use Config Files
→ [Config-Based Evaluation](docs/guides/CONFIG_BASED_EVALUATION.md)

#### Add Custom Metrics
→ [Metrics Guide](docs/guides/METRICS.md)

#### Use GPT-4 for Evaluation
→ [LLM Judge Guide](docs/guides/LLM_JUDGE.md)

#### Understand Agent Types
→ [Agents Guide](docs/AGENTS.md)

#### Manage Datasets
→ [Dataset Management](docs/guides/DATASET_MANAGEMENT.md)

#### Fix Issues
→ [Troubleshooting](docs/TROUBLESHOOTING.md)

---

## 📊 Documentation Structure

```
ragicamp/
├── README.md                          # Main readme
├── DOCUMENTATION.md                   # This file
├── QUICK_REFERENCE.md                # Command cheat sheet
│
├── docs/
│   ├── README.md                     # Documentation index
│   ├── V02_RELEASE_NOTES.md         # v0.2 release notes
│   ├── GETTING_STARTED.md           # Installation
│   ├── ARCHITECTURE.md              # System design
│   ├── AGENTS.md                    # Agent types
│   ├── USAGE.md                     # Usage patterns
│   ├── TROUBLESHOOTING.md           # Common issues
│   │
│   ├── guides/                       # Feature guides
│   │   ├── QUICKSTART_V02.md        # ⭐ Quick start
│   │   ├── MLFLOW_RAGAS_GUIDE.md    # ⭐ v0.2 features
│   │   ├── CONFIG_BASED_EVALUATION.md
│   │   ├── TWO_PHASE_EVALUATION.md
│   │   ├── METRICS.md
│   │   ├── LLM_JUDGE.md
│   │   └── ... (more guides)
│   │
│   └── archives/                     # Historical docs
│       ├── development/
│       ├── fixes/
│       └── ...
│
└── experiments/
    └── configs/                      # Example configs
        └── example_mlflow_ragas.yaml # ⭐ v0.2 example
```

---

## 🔍 Find What You Need

### By Experience Level

**Beginner:**
1. [Quick Start](docs/guides/QUICKSTART_V02.md) - Try features
2. [Getting Started](docs/GETTING_STARTED.md) - Full setup
3. [Quick Reference](QUICK_REFERENCE.md) - Commands

**Intermediate:**
1. [Config Guide](docs/guides/CONFIG_BASED_EVALUATION.md) - YAML configs
2. [Metrics Guide](docs/guides/METRICS.md) - Choosing metrics
3. [Architecture](docs/ARCHITECTURE.md) - How it works

**Advanced:**
1. [MLflow & Ragas](docs/guides/MLFLOW_RAGAS_GUIDE.md) - Advanced features
2. [Two-Phase](docs/guides/TWO_PHASE_EVALUATION.md) - Optimization
3. [Agents Guide](docs/AGENTS.md) - Custom agents

### By Topic

**Evaluation:**
- [Config-Based](docs/guides/CONFIG_BASED_EVALUATION.md)
- [Baseline](docs/guides/BASELINE_EVALUATION.md)
- [Two-Phase](docs/guides/TWO_PHASE_EVALUATION.md)
- [Batch Processing](docs/guides/BATCH_PROCESSING.md)

**Metrics:**
- [Overview](docs/guides/METRICS.md)
- [LLM Judge](docs/guides/LLM_JUDGE.md)
- [Faithfulness](docs/guides/FAITHFULNESS_METRICS_SUMMARY.md)
- [Ragas](docs/guides/MLFLOW_RAGAS_GUIDE.md#ragas-metrics)

**Tracking:**
- [MLflow](docs/guides/MLFLOW_RAGAS_GUIDE.md#mlflow-tracking)
- [State Management](docs/guides/MLFLOW_RAGAS_GUIDE.md#state-management)
- [Output Structure](docs/guides/OUTPUT_STRUCTURE.md)

---

## 📝 Contributing to Docs

Found an issue? Want to improve documentation?

1. Check [existing docs](docs/)
2. Update or create markdown file
3. Update [docs/README.md](docs/README.md) index
4. Submit PR

**Style Guide:**
- Use clear headings (h2 ## for main sections)
- Include code examples
- Add "See also" links
- Keep it concise

---

## 🆘 Still Can't Find It?

1. **Search:** Use GitHub search or `grep -r "your term" docs/`
2. **Ask:** Open an issue with tag `documentation`
3. **Browse:** Check [docs/](docs/) folder directly

---

**Quick Links:**
- **New?** → [Quick Start](docs/guides/QUICKSTART_V02.md)
- **Commands?** → [Quick Reference](QUICK_REFERENCE.md)
- **All Docs?** → [Documentation Index](docs/README.md)
