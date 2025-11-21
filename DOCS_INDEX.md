# Documentation Index 📚

Complete guide to RAGiCamp documentation.

---

## 🚀 Getting Started

| Document | Purpose | Audience |
|----------|---------|----------|
| **[README.md](README.md)** | Project overview, quick start | Everyone (start here!) |
| **[WHATS_NEW.md](WHATS_NEW.md)** ⭐ | Latest features | Existing users |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Command cheat sheet | Quick lookup |

---

## 📖 Main Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** | Detailed setup guide | New users |
| **[docs/USAGE.md](docs/USAGE.md)** | Usage examples | All users |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design | Developers |
| **[AGENTS.md](AGENTS.md)** | Agent types & usage | Researchers |

---

## 📝 Guides

Located in `docs/guides/`:

| Guide | Purpose |
|-------|---------|
| **[TWO_PHASE_EVALUATION.md](docs/guides/TWO_PHASE_EVALUATION.md)** ⭐ **NEW!** | Robust two-phase evaluation (MUST READ) |
| **[CONFIG_BASED_EVALUATION.md](docs/guides/CONFIG_BASED_EVALUATION.md)** | Complete config guide |
| **[METRICS.md](docs/guides/METRICS.md)** | Choosing metrics |
| **[LLM_JUDGE.md](docs/guides/LLM_JUDGE.md)** | Using GPT-4 for evaluation |
| **[BATCH_PROCESSING.md](docs/guides/BATCH_PROCESSING.md)** | Batch generation guide |
| **[DATASET_MANAGEMENT.md](docs/guides/DATASET_MANAGEMENT.md)** | Dataset downloading & caching |
| **[PATH_UTILITIES.md](docs/guides/PATH_UTILITIES.md)** | File & directory utilities |
| **[BASELINE_EVALUATION.md](docs/guides/BASELINE_EVALUATION.md)** | Running baselines |
| **[OUTPUT_STRUCTURE.md](docs/guides/OUTPUT_STRUCTURE.md)** | Understanding output files |
| **[NORMALIZATION_GUIDE.md](docs/guides/NORMALIZATION_GUIDE.md)** | Text normalization |
| **[TROUBLESHOOTING_MATPLOTLIB.md](docs/guides/TROUBLESHOOTING_MATPLOTLIB.md)** | Fixing matplotlib issues |
| **[FAITHFULNESS_METRICS_SUMMARY.md](docs/guides/FAITHFULNESS_METRICS_SUMMARY.md)** | Faithfulness metrics |

---

## 🧪 Testing & CI Documentation

| Document | Purpose |
|----------|---------|
| **[tests/README.md](tests/README.md)** ⭐ **NEW!** | Complete testing guide |
| **[.github/workflows/README.md](.github/workflows/README.md)** ⭐ **NEW!** | CI/CD pipeline documentation |

**Test Categories:**
- Two-phase evaluation tests
- Checkpointing system tests  
- Config validation tests
- Metrics computation tests
- Factory pattern tests
- Agent functionality tests

**CI Pipeline (Python 3.12):**
- Automated tests on every push/PR
- Code quality checks (Black, isort)
- Test coverage reports
- Config validation

---

## 📜 Project Information

| Document | Purpose |
|----------|---------|
| **[CHANGELOG.md](CHANGELOG.md)** | Version history & changes |
| **[WHATS_NEW.md](WHATS_NEW.md)** | Latest features summary |

---

## 🔧 Technical References

| Document | Purpose | Audience |
|----------|---------|----------|
| **[docs/fixes/config_factory_fixes.md](docs/fixes/config_factory_fixes.md)** | Config/factory bug fixes | Developers |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Common issues | All users |

---

## 🛠️ Development & Internal Docs

Located in `docs/development/`:

| Document | Purpose |
|----------|---------|
| **[ARCHITECTURE_REVIEW.md](docs/development/ARCHITECTURE_REVIEW.md)** | Comprehensive codebase analysis |
| **[CONFIG_SYSTEM_BENEFITS.md](docs/development/CONFIG_SYSTEM_BENEFITS.md)** | Config system benefits deep dive |
| **[REFACTOR_COMPLETE.md](docs/development/REFACTOR_COMPLETE.md)** | Recent refactoring summary |

---

## 📦 Archives

Historical documents (completed work):

- **[docs/archives/IMPLEMENTATION_CHECKLIST.md](docs/archives/IMPLEMENTATION_CHECKLIST.md)** - Completed refactoring tasks
- **[docs/archives/REFACTORING_SUMMARY.md](docs/archives/REFACTORING_SUMMARY.md)** - Previous refactoring work

---

## 🗺️ Navigation Guide

### If you're...

#### 👋 **New to RAGiCamp**
1. Start with [README.md](README.md)
2. Follow [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
3. Read [docs/guides/TWO_PHASE_EVALUATION.md](docs/guides/TWO_PHASE_EVALUATION.md) ⭐ **Essential!**
4. Try commands from [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
5. Read [docs/guides/CONFIG_BASED_EVALUATION.md](docs/guides/CONFIG_BASED_EVALUATION.md)

#### 🔁 **Returning User**
1. Check [WHATS_NEW.md](WHATS_NEW.md) for latest features
2. Review [CHANGELOG.md](CHANGELOG.md) for changes
3. Validate your configs: `make validate-all-configs`

#### 🔬 **Researcher**
1. [docs/AGENTS.md](docs/AGENTS.md) - Understanding agents
2. [docs/guides/METRICS.md](docs/guides/METRICS.md) - Choosing metrics
3. [docs/guides/CONFIG_BASED_EVALUATION.md](docs/guides/CONFIG_BASED_EVALUATION.md) - Running experiments
4. [docs/guides/LLM_JUDGE.md](docs/guides/LLM_JUDGE.md) - High-quality evaluation

#### 💻 **Developer**
1. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
2. [tests/README.md](tests/README.md) ⭐ - Testing guide
3. [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) - Detailed analysis
4. [CONFIG_SYSTEM_BENEFITS.md](CONFIG_SYSTEM_BENEFITS.md) - Config system
5. [docs/guides/PATH_UTILITIES.md](docs/guides/PATH_UTILITIES.md) - Utility functions

**Before contributing:**
```bash
make test              # Run all tests
make test-coverage     # Check coverage
make format            # Format code
```

#### 🐛 **Troubleshooting**
1. [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues
2. [docs/guides/TROUBLESHOOTING_MATPLOTLIB.md](docs/guides/TROUBLESHOOTING_MATPLOTLIB.md) - Matplotlib fixes
3. Check [CHANGELOG.md](CHANGELOG.md) for known issues

---

## 📦 Examples & Utilities

| Document | Purpose |
|----------|---------|
| **[examples/README.md](examples/README.md)** ⭐ **NEW!** | Guide to utility examples |

**Available Examples:**
- Dataset download & caching
- Dataset filtering
- Path utilities
- Per-question metrics analysis

**Note:** For running evaluations, use the config-based approach in `experiments/configs/` instead!

---

## 📊 Documentation Statistics

- **Total Documents**: 30+
- **Main Guides**: 12
- **Testing Docs**: 2 (tests + CI)
- **Technical Docs**: 5
- **Development Docs**: 3
- **Example Docs**: 1
- **Test Files**: 6 test files, 86 unit tests
- **Archives**: 2
- **Last Updated**: November 21, 2025

**Code Quality:**
- ✅ Python 3.12 (compatible with 3.9+)
- ✅ 78/86 tests passing (90.7%)
- ✅ Automated CI pipeline
- ✅ Type hints throughout
- ✅ Formatted with Black & isort

---

## 🔍 Search Tips

### By Topic

- **Evaluation**: TWO_PHASE_EVALUATION.md ⭐, CONFIG_BASED_EVALUATION.md
- **Robustness**: TWO_PHASE_EVALUATION.md (checkpointing, error recovery)
- **Testing**: tests/README.md ⭐ (unit tests, coverage, test guide)
- **Configuration**: CONFIG_BASED_EVALUATION.md, CONFIG_SYSTEM_BENEFITS.md
- **Metrics**: METRICS.md, LLM_JUDGE.md, FAITHFULNESS_METRICS_SUMMARY.md
- **Datasets**: DATASET_MANAGEMENT.md
- **Performance**: BATCH_PROCESSING.md, WHATS_NEW.md
- **Agents**: AGENTS.md, docs/ARCHITECTURE.md
- **Troubleshooting**: TROUBLESHOOTING.md, guides/TROUBLESHOOTING_MATPLOTLIB.md

### By File Type

- **Guides** → `docs/guides/`
- **Examples** → `examples/`
- **Archives** → `docs/archives/`
- **Bug Fixes** → `docs/fixes/`

---

## 🤝 Contributing

Documentation improvements welcome! Please:
1. Keep docs in appropriate directories
2. Update this index when adding new docs
3. Follow existing formatting style
4. Include examples where helpful

---

## 📞 Still Can't Find It?

1. Check [README.md](README.md) for overview
2. Try `make help` for commands
3. Search docs with `grep -r "your search" docs/`
4. Check examples in `examples/`

---

**💡 Tip:** Bookmark this page for quick access to all documentation!

