# Changelog

All notable changes to RAGiCamp will be documented in this file.

## [Unreleased]

### Added - 2025-11-18

**Batch Processing (2-3x Faster Evaluations)**
- Added `batch_answer()` method to `RAGAgent` base class for parallel processing
- Implemented optimized batch processing in `DirectLLMAgent`
- Added `batch_size` parameter to `Evaluator.evaluate()`
- New configs: `nq_baseline_gemma2b_quick_batch.yaml`, `nq_baseline_gemma2b_full_batch.yaml`
- New Makefile commands: `make eval-baseline-quick-batch`, `make eval-baseline-full-batch`
- Complete guide: `docs/guides/BATCH_PROCESSING.md`

**Dataset Caching System (10-60x Faster Loading)**
- Added automatic caching to all dataset classes
- New methods: `save_to_cache()`, `load_from_cache()`, `download_and_cache()` on `QADataset`
- New Makefile commands: `make download-nq`, `make download-triviaqa`, `make download-hotpotqa`, `make list-datasets`
- Complete guide: `docs/guides/DATASET_MANAGEMENT.md`

**Path Utilities**
- New module: `src/ragicamp/utils/paths.py`
- Functions: `ensure_dir()`, `safe_write_json()`, `ensure_output_dirs()`
- Prevents `FileNotFoundError` when saving results
- Example: `examples/path_utilities_example.py`

**Expected answer in results**: JSON output now includes `expected_answer` (primary) and `all_acceptable_answers` fields
**Answer filtering**: New `--filter-no-answer` flag to filter out questions without explicit answers
**Dataset filtering methods**: Added `filter_with_answers()` and `get_examples_with_answers()` to QADataset base class

### Fixed - 2025-11-18

**Matplotlib Backend Issues**
- Fixed ValueError when running BERTScore/BLEURT in non-interactive environments (Google Colab, scripts)
- Automatic backend detection in `bertscore.py` and `bleurt.py`
- Guide: `docs/guides/TROUBLESHOOTING_MATPLOTLIB.md`

**File Creation Errors**
- Fixed `FileNotFoundError` in evaluator when outputs directory doesn't exist
- Now automatically creates directories using `ensure_dir()`

### Changed

- Results JSON now has three answer fields:
  - `expected_answer`: Primary/first expected answer (NEW)
  - `all_acceptable_answers`: All acceptable answers (NEW)
  - `references`: All acceptable answers (kept for backward compatibility)
- Makefile commands now use `--filter-no-answer` by default for cleaner evaluations
- Updated `.cursorrules` with documentation location guidelines (no loose `.md` files in root)

### Documentation - 2025-11-18

- **New guides**: Batch Processing, Dataset Management, Path Utilities, Matplotlib Troubleshooting
- **Updated**: Quick Reference (batch examples), Troubleshooting (quick fixes table)
- **Improved**: `.cursorrules` with clear documentation structure guidelines

## [0.1.0] - 2024-10-29

### Added
- Initial RAGiCamp framework with modular architecture
- Agents: DirectLLM, FixedRAG, BanditRAG, MDPRAG
- Models: HuggingFace and OpenAI interfaces
- Retrievers: Dense (FAISS) and Sparse (TF-IDF)
- Datasets: Natural Questions, HotpotQA, TriviaQA
- Metrics: Exact Match, F1, BERTScore, LLM-as-judge
- Policies: Epsilon-Greedy, UCB, Q-Learning, Random
- Training and evaluation utilities
- Configuration-driven experiment system
- Comprehensive documentation

### Added (Package Manager Migration)
- Migrated from pip to uv package manager
- Added uv.lock for reproducible builds
- Added .python-version file
- Updated pyproject.toml with hatchling build backend

### Added (Gemma 2B Baseline)
- Dedicated Gemma 2B evaluation script
- Gemma 2B configuration file
- CPU/GPU support with 8-bit quantization option
- Multiple dataset support (NQ, HotpotQA, TriviaQA)
- Comprehensive documentation (GEMMA2B_QUICKSTART.md, QUICK_START_GEMMA.md)
- Makefile with convenient commands

