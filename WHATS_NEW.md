# What's New in RAGiCamp 🎉

**November 21, 2025** - Robust two-phase evaluation!

---

## 🚀 TL;DR

RAGiCamp now has:
- ✅ **Two-phase evaluation** - Never lose progress to API failures!
- ✅ **Automatic checkpointing** - LLM judge resumes from where it left off
- ✅ **Modular design** - Generate predictions once, compute metrics many times
- ✅ **Three evaluation modes** - generate, evaluate, or both
- ✅ **Type-safe configs** with validation
- ✅ **100x faster LLM judge**
- ✅ **Better error handling**

**Everything is backward compatible!** Your existing code still works.

---

## ✨ Top New Features

### 1. 🛡️ Two-Phase Evaluation (GAME CHANGER!)

**The Problem**: Spend 2 hours generating predictions, then LLM judge fails at minute 100 with API 403 error. Lost all progress! 😢

**The Solution**: Two-phase evaluation

```bash
# Phase 1: Generate predictions (saved immediately)
mode: generate → Takes 50 minutes → predictions_raw.json ✓

# Phase 2: Compute metrics (can retry!)
python scripts/compute_metrics.py → If it fails, just run again!
```

**Benefits**:
- ✅ **Never lose predictions** - Saved before metrics computation
- ✅ **Retry on failure** - API errors? Just run metrics again
- ✅ **Experiment freely** - Try different metrics on same predictions
- ✅ **Auto-checkpointing** - LLM judge saves progress every 5 batches

**Three modes**:
```yaml
evaluation:
  mode: generate  # Generate predictions only (safest)
  mode: evaluate  # Compute metrics only (on existing predictions)
  mode: both      # Do everything (classic mode, but still saves predictions first!)
```

**Real example**:
```bash
# You ran this and it failed at batch 35/57 with 403 error:
python scripts/compute_metrics.py --predictions outputs/predictions.json --config llm_judge.yaml
# ❌ Error: openai.PermissionDeniedError: Error code: 403

# No problem! Just run again:
python scripts/compute_metrics.py --predictions outputs/predictions.json --config llm_judge.yaml
# ✓ Resumed from batch 35, continued from there
# ✓ Success!
```

See **[Two-Phase Evaluation Guide](docs/guides/TWO_PHASE_EVALUATION.md)**

### 2. Config Validation

Catch errors **before** running experiments:

```bash
# Validate your config
make validate-config CONFIG=my_experiment.yaml

# ✓ Configuration validated successfully
# ✓ All required fields present
# ✓ All types correct
```

**Benefits:**
- Save time (no more 20-minute runs that fail instantly)
- Better error messages
- Type safety

### 3. Config Templates

Generate configs instead of copying:

```bash
# Create a baseline config
make create-config OUTPUT=my_baseline.yaml TYPE=baseline

# Edit and run
vim my_baseline.yaml
make validate-config CONFIG=my_baseline.yaml  # Check it
make eval CONFIG=my_baseline.yaml              # Run it
```

**Benefits:**
- Faster setup
- Always up-to-date
- No copy-paste errors

### 4. Component Registry

Add custom components without modifying core code:

```python
from ragicamp import ComponentRegistry
from ragicamp.agents.base import RAGAgent

# Register your custom agent
@ComponentRegistry.register_agent("my_awesome_agent")
class MyAwesomeAgent(RAGAgent):
    def answer(self, query: str, **kwargs):
        # Your logic here
        pass

# Use in config immediately:
# agent:
#   type: my_awesome_agent
#   name: "awesome"
```

**Benefits:**
- Easy extensibility
- No core code changes
- Clean separation

### 5. Better Imports

Cleaner, simpler imports:

```python
# Before ❌
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.datasets.nq import NaturalQuestionsDataset

# After ✅
from ragicamp.agents import DirectLLMAgent
from ragicamp.models import HuggingFaceModel
from ragicamp.datasets import NaturalQuestionsDataset
```

**Benefits:**
- Less typing
- Easier to remember
- Better IDE autocomplete

### 6. 100x Faster LLM Judge

Fixed caching bug - LLM judge now reuses judgments:

```
Before: 101 API calls (1 batch + 100 individual)
After:  1 API call (just the batch)

Speedup: 100x faster! 🚀
Cost savings: 100x cheaper! 💰
```

**Benefits:**
- Dramatically faster evaluation
- Much lower API costs
- More experiments, same budget

### 7. 🧪 Comprehensive Test Suite

**NEW!** Full unit test coverage for robustness:

```bash
# Run all tests
make test

# Test specific features
make test-two-phase      # Two-phase evaluation
make test-checkpoint     # Checkpoint system
make test-config         # Config validation
```

**What's tested:**
- ✅ Two-phase evaluation (generate + evaluate modes)
- ✅ Checkpointing (save & resume)
- ✅ Config validation (all three modes)
- ✅ Metrics computation
- ✅ Factory pattern
- ✅ Error handling

**Coverage targets:**
- Core components: **>80%**
- Critical paths: **100%**

See **[tests/README.md](tests/README.md)** for details.

---

## 🛠️ New Commands

### Config Management

```bash
# Validate a config
make validate-config CONFIG=experiments/configs/my_config.yaml

# Validate all configs
make validate-all-configs

# Create config template
make create-config OUTPUT=my_config.yaml TYPE=baseline
```

### Existing Commands Still Work

All your existing commands work exactly as before:

```bash
make eval-baseline-quick
make eval-baseline-full
make eval-rag
# ... etc
```

---

## 📖 Example Workflow

### Before (Old Way - Still Works!)

```bash
# Copy config manually
cp experiments/configs/nq_baseline_gemma2b_full.yaml my_experiment.yaml

# Edit it
vim my_experiment.yaml

# Run (might fail if config is wrong)
uv run python experiments/scripts/run_experiment.py --config my_experiment.yaml --mode eval
# 💥 Error after 20 minutes!
```

### After (New Way - Recommended!)

```bash
# Create from template
make create-config OUTPUT=my_experiment.yaml TYPE=baseline

# Edit it
vim my_experiment.yaml

# Validate (catches errors immediately)
make validate-config CONFIG=my_experiment.yaml
# ✓ Configuration validated successfully

# Run (confident it will work)
uv run python experiments/scripts/run_experiment.py --config my_experiment.yaml --mode eval
# ✓ Success!
```

---

## 🎯 What This Means for You

### If You're a Researcher

- ✅ **Faster experimentation** - Validation catches errors early
- ✅ **Reproducible** - Configs are type-safe and validated
- ✅ **Extensible** - Easy to add custom models/agents/metrics
- ✅ **Cost-effective** - LLM judge 100x cheaper

### If You're a Developer

- ✅ **Better DX** - Cleaner imports, better errors
- ✅ **Maintainable** - Factory pattern, clean architecture
- ✅ **Testable** - Easy to mock components
- ✅ **Extensible** - Registry for custom components

### If You're Learning

- ✅ **Better errors** - Understand what went wrong
- ✅ **Templates** - Start with working examples
- ✅ **Validation** - Learn correct config structure
- ✅ **Examples** - More docs and examples

---

## 🔄 Migration (Optional)

### You Don't Have to Change Anything!

All existing code works. But if you want to use new features:

### Step 1: Validate Your Configs (Optional)

```bash
make validate-all-configs
```

Fix any issues if they come up (probably none).

### Step 2: Update Imports (Optional)

Change your imports to use the cleaner style:

```python
# Old (still works)
from ragicamp.agents.direct_llm import DirectLLMAgent

# New (cleaner)
from ragicamp.agents import DirectLLMAgent
```

### Step 3: Start Using Validation (Recommended)

Before running experiments:

```bash
make validate-config CONFIG=my_config.yaml
```

That's it! You're done.

---

## 📊 Performance Improvements

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| LLM Judge | 101 calls | 1 call | **100x faster** |
| Config Errors | Runtime | Validation | **Instant feedback** |
| Imports | Deep paths | Module root | **Cleaner** |
| Script Length | 426 LOC | 295 LOC | **31% shorter** |

---

## 🎓 Learn More

- **Quick Start**: See [README.md](README.md)
- **Config Guide**: See [docs/guides/CONFIG_BASED_EVALUATION.md](docs/guides/CONFIG_BASED_EVALUATION.md)
- **What Changed**: See [CHANGELOG.md](CHANGELOG.md)
- **Architecture**: See [REFACTOR_COMPLETE.md](REFACTOR_COMPLETE.md)
- **All Benefits**: See [CONFIG_SYSTEM_BENEFITS.md](CONFIG_SYSTEM_BENEFITS.md)

---

## 🤔 FAQ

### Q: Do I need to update my code?

**A:** No! Everything is backward compatible.

### Q: Will my existing configs work?

**A:** Yes! Just run `make validate-all-configs` to confirm.

### Q: How do I use the new features?

**A:** Start with `make create-config` and `make validate-config`. Read the updated [CONFIG_BASED_EVALUATION.md](docs/guides/CONFIG_BASED_EVALUATION.md) guide.

### Q: Is this a breaking change?

**A:** No! Everything is backward compatible. New features are opt-in.

### Q: What if I find a bug?

**A:** Please report it! Check existing docs first, then let us know.

### Q: Can I use both old and new styles?

**A:** Yes! Mix and match as needed.

---

## 💡 Try It Now

```bash
# Create a config
make create-config OUTPUT=test.yaml TYPE=baseline

# Validate it
make validate-config CONFIG=test.yaml

# Run a quick test
make eval-baseline-quick

# Check results
ls outputs/
```

---

## 🙏 Feedback Welcome!

We'd love to hear:
- What you like about the new features
- What could be improved
- What features you'd like to see next

---

**Happy experimenting!** 🚀

*Built with ❤️ for the RAG community*

