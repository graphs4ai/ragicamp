# RAGiCamp Documentation

Complete documentation for the RAGiCamp framework.

> **Quick Start:** [Cheatsheet →](../CHEATSHEET.md) | [Getting Started →](GETTING_STARTED.md)

---

## 🚀 Getting Started

| Document | Description |
|----------|-------------|
| **[Cheatsheet](../CHEATSHEET.md)** ⭐ | Quick reference for all commands |
| **[Getting Started](GETTING_STARTED.md)** | Installation and first steps |
| **[Contributing](../CONTRIBUTING.md)** | How to contribute |

## 🏗️ Core Documentation

| Document | Description |
|----------|-------------|
| **[Architecture](ARCHITECTURE.md)** | System design and components |
| **[Agents Guide](guides/AGENTS.md)** | Understanding DirectLLM and FixedRAG agents |
| **[Usage Guide](USAGE.md)** | Detailed usage patterns |
| **[Troubleshooting](TROUBLESHOOTING.md)** | Common issues and solutions |

## 📖 Feature Guides

### Evaluation

| Guide | Description |
|-------|-------------|
| **[Baseline Evaluation](guides/BASELINE_EVALUATION.md)** | Evaluating without retrieval |
| **[Comprehensive Baseline](guides/COMPREHENSIVE_BASELINE.md)** | Full baseline study guide |

### Metrics

| Guide | Description |
|-------|-------------|
| **[Metrics Guide](guides/METRICS.md)** | Choosing and using metrics |
| **[LLM Judge](guides/LLM_JUDGE.md)** | Using GPT-4/OpenAI for evaluation |

## 🎯 Quick Links

### New Users

- **[Cheatsheet](../CHEATSHEET.md)** - Quick reference for all commands
- **[Getting Started](GETTING_STARTED.md)** - Full installation guide

### Common Tasks

- **Running experiments?** → `ragicamp run conf/study/my_study.yaml`
- **Check status?** → `ragicamp health outputs/my_study`
- **Compare results?** → `ragicamp compare outputs/my_study`
- **Recompute metrics?** → `ragicamp metrics outputs/my_study/exp -m f1,llm_judge`
- **Using LLM judge?** → [LLM Judge Guide](guides/LLM_JUDGE.md)
- **Choosing metrics?** → [Metrics Guide](guides/METRICS.md)

### Understanding the System

- **How it works?** → [Architecture](ARCHITECTURE.md)
- **Agent types?** → [Agents Guide](guides/AGENTS.md)
- **Having issues?** → [Troubleshooting](TROUBLESHOOTING.md)

## 💡 Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── GETTING_STARTED.md     # Installation and setup
├── ARCHITECTURE.md        # System design
├── USAGE.md               # Detailed usage
├── TROUBLESHOOTING.md     # Common issues
│
└── guides/                # Feature guides
    ├── AGENTS.md          # DirectLLM, FixedRAG
    ├── BASELINE_EVALUATION.md
    ├── COMPREHENSIVE_BASELINE.md
    ├── METRICS.md
    └── LLM_JUDGE.md
```

## 🤝 Contributing

See **[CONTRIBUTING.md](../CONTRIBUTING.md)** for how to contribute.

---

**Ready to start?** → [Cheatsheet](../CHEATSHEET.md) | **Full install?** → [Getting Started](GETTING_STARTED.md)
