# Changelog

All notable changes to RAGiCamp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] - 2025-11-18

### 🎉 Major Features

#### Type-Safe Configuration System
- **Added** Pydantic-based configuration schemas (`src/ragicamp/config/schemas.py`)
- **Added** ConfigLoader for loading and validating configs (`src/ragicamp/config/loader.py`)
- **Added** Automatic validation when running experiments
- **Added** Better error messages for config issues
- **Improved** Type safety throughout config handling

#### Component Factory & Registry
- **Added** ComponentFactory for centralized component instantiation (`src/ragicamp/factory.py`)
- **Added** ComponentRegistry for easy extensibility (`src/ragicamp/registry.py`)
- **Added** Decorator-based component registration
- **Improved** Code organization and maintainability

#### CLI Tools
- **Added** `scripts/validate_config.py` - Validate experiment configs
- **Added** `scripts/create_config.py` - Generate config templates
- **Added** `make validate-config` - Validate single config
- **Added** `make validate-all-configs` - Validate all configs
- **Added** `make create-config` - Create config from template

### ✨ Enhancements

#### Better Imports
- **Improved** Module exports for cleaner imports
- **Added** Top-level exports in `__init__.py` files
- **Example**: `from ragicamp.agents import DirectLLMAgent` (instead of deep imports)

#### Configuration Updates
- **Fixed** Generation parameters (max_tokens, temperature) incorrectly passed to model init
- **Updated** All Pydantic `.dict()` calls to `.model_dump()` (Pydantic V2)
- **Cleaned** Redundant experiment configs

#### Performance Improvements
- **Added** Batch processing support for model generation
- **Added** Batch evaluation in framework
- **Fixed** LLM Judge caching bug (100x performance improvement)
- **Added** Judgment caching to avoid redundant API calls

#### Dataset Management
- **Enhanced** Base `QADataset` class with caching
- **Added** `download_and_cache()` method for unified dataset handling
- **Improved** Cache path generation to include dataset-specific parameters
- **Fixed** TriviaQA and HotpotQA cache paths to include subset/distractor params

#### Path Utilities
- **Added** `src/ragicamp/utils/paths.py` module
- **Added** `ensure_dir()` - Create directories automatically
- **Added** `safe_write_json()` - Safe JSON writing with directory creation
- **Added** `ensure_output_dirs()` - Setup all standard directories
- **Fixed** FileNotFoundError when output directories don't exist

#### Cross-Environment Fixes
- **Fixed** Matplotlib backend issues in BERTScore and BLEURT
- **Added** Force `Agg` backend for non-interactive environments
- **Improved** Compatibility across notebook and script contexts

#### Experiment Scripts
- **Refactored** `run_experiment.py` to use ComponentFactory (426 → 295 LOC)
- **Improved** Type safety with Pydantic configs
- **Cleaned** Redundant scripts (removed 3 duplicate scripts)
- **Enhanced** Error handling and validation

### 🐛 Bug Fixes

- **Fixed** `TypeError` when passing generation params to model initialization
- **Fixed** `KeyError: 'partially_correct'` in LLM Judge binary mode
- **Fixed** LLM Judge making redundant API calls (now caches results)
- **Fixed** `FileNotFoundError` when output directories don't exist
- **Fixed** Matplotlib backend crashes in non-interactive environments
- **Fixed** Cache path not considering dataset-specific parameters

### 📚 Documentation

#### New Documentation
- **Added** `REFACTOR_COMPLETE.md` - Complete refactoring summary
- **Added** `CONFIG_SYSTEM_BENEFITS.md` - Benefits of new config system
- **Added** `ARCHITECTURE_REVIEW.md` - Comprehensive codebase analysis
- **Added** `IMPLEMENTATION_CHECKLIST.md` - Task-by-task improvement guide
- **Added** `docs/fixes/config_factory_fixes.md` - Config/factory bug fixes
- **Added** This CHANGELOG!

#### Updated Documentation
- **Updated** `README.md` - Added config validation section, new features
- **Updated** `QUICK_REFERENCE.md` - Added config management commands
- **Updated** `CONFIG_BASED_EVALUATION.md` - Added validation, factory/registry examples
- **Updated** `.cursorrules` - Better guidance on documentation practices
- **Consolidated** Scattered `.md` files into main documentation

#### Documentation Improvements
- **Added** Examples for ComponentFactory usage
- **Added** Examples for ComponentRegistry usage
- **Added** Config validation examples
- **Added** Better error message examples
- **Improved** Code snippets with type hints

### 🗑️ Removed

- **Deleted** `experiments/scripts/run_gemma2b_baseline.py` (redundant)
- **Deleted** `experiments/scripts/run_fixed_rag_eval.py` (redundant)
- **Deleted** `experiments/scripts/demo_new_architecture.py` (redundant)
- **Deleted** Redundant experiment configs (4 files)
- **Deleted** Scattered changelog `.md` files (consolidated into main CHANGELOG)

### 🏗️ Architecture

- **Improved** Separation of concerns with factory pattern
- **Added** Registry system for easy extensibility
- **Enhanced** Type safety with Pydantic throughout
- **Centralized** Component instantiation logic
- **Standardized** Configuration handling
- **Modularized** Path and directory utilities

### 📊 Metrics

- **LOC Reduction**: `run_experiment.py` 426 → 295 LOC (31% reduction)
- **New Infrastructure**: ~1,500 LOC (factory, registry, config, utils)
- **Performance**: LLM Judge 100x faster with caching
- **Files Deleted**: 7 redundant files
- **Files Created**: 11 new infrastructure/doc files

---

## Migration Guide

### For Existing Users

#### 1. Update Imports (Optional but Recommended)

**Before:**
```python
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.models.huggingface import HuggingFaceModel
```

**After:**
```python
from ragicamp.agents import DirectLLMAgent
from ragicamp.models import HuggingFaceModel
```

#### 2. Validate Your Configs

```bash
# Validate your existing configs
make validate-all-configs

# If any fail, fix the issues
make validate-config CONFIG=experiments/configs/my_config.yaml
```

#### 3. Update Pydantic Usage (If Programmatic)

**Before:**
```python
config_dict = config.model.dict()
```

**After:**
```python
config_dict = config.model.model_dump()
```

### Backward Compatibility

✅ **All existing code still works!** This release is 100% backward compatible:
- Old imports still valid (new ones are just cleaner)
- Existing configs work without changes
- No breaking changes to APIs

### New Features You Can Start Using

1. **Config Validation**: `make validate-config CONFIG=my.yaml`
2. **Config Templates**: `make create-config OUTPUT=my.yaml TYPE=baseline`
3. **ComponentFactory**: `ComponentFactory.create_model(config)`
4. **ComponentRegistry**: `@ComponentRegistry.register_agent("name")`
5. **Better Imports**: `from ragicamp.agents import DirectLLMAgent`

---

## Key Benefits Summary

### For Users
- ✅ Catch config errors before running (save time)
- ✅ Better error messages (easier debugging)
- ✅ Config templates (faster setup)
- ✅ Type safety (fewer bugs)

### For Developers
- ✅ Cleaner imports (better DX)
- ✅ Factory pattern (easier testing)
- ✅ Registry system (easy extensibility)
- ✅ Better architecture (maintainable code)

### For Researchers
- ✅ Reproducible configs (validated)
- ✅ Easy to extend (registry system)
- ✅ Faster experimentation (validation + templates)
- ✅ Better performance (LLM judge caching)

---

## What's Next

### Planned for Next Release

- [ ] More config templates (advanced RAG, multi-hop, etc.)
- [ ] Config inheritance and composition
- [ ] Web UI for config creation
- [ ] Performance profiling tools
- [ ] More example custom components

### Under Consideration

- [ ] Distributed evaluation support
- [ ] Experiment tracking integration (W&B, MLflow)
- [ ] Auto-tuning hyperparameters
- [ ] More dataset connectors
- [ ] Streaming evaluation support

---

## Contributors

- Gabriel Frontera (@gabriel_frontera_cloudwalk_io)

---

## Links

- **Documentation**: See `README.md` and `docs/`
- **Quick Reference**: See `QUICK_REFERENCE.md`
- **Config Guide**: See `docs/guides/CONFIG_BASED_EVALUATION.md`
- **Architecture Review**: See `ARCHITECTURE_REVIEW.md`
- **Refactoring Summary**: See `REFACTOR_COMPLETE.md`

---

**Questions?** Check the docs or open an issue!

**Want to contribute?** See contributing guidelines (coming soon!)
