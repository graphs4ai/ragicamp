# Dataset Management Guide

This guide explains how to download, manage, and use QA datasets in RAGiCamp.

## Overview

RAGiCamp's dataset system features **automatic caching** built directly into the dataset classes. This means:

- ✅ **First load**: Downloads from HuggingFace (10-30 seconds)
- ✅ **Subsequent loads**: Instant loading from cache (< 1 second)
- ✅ **Automatic**: Works transparently - no code changes needed
- ✅ **Offline-friendly**: Work without internet after initial download

## Quick Start

```bash
# Download Natural Questions (validation split)
make download-nq

# List downloaded datasets
make list-datasets

# Use in experiments (automatic - no changes needed)
make eval-baseline-quick
```

### Python Usage

```python
from ragicamp.datasets import NaturalQuestionsDataset

# Just use normally - caching happens automatically!
dataset = NaturalQuestionsDataset(split="validation")
# First run: downloads from HuggingFace
# Second run: loads from cache (much faster!)
```

## Available Datasets

RAGiCamp supports three major QA datasets:

### 1. Natural Questions (NQ)

- **Source:** Google's Natural Questions dataset
- **Description:** Real Google search queries with answers from Wikipedia
- **Splits:** train (~300k), validation (~8k)
- **Format:** Open-domain QA (questions + multiple acceptable answers)

```bash
# Download validation set (recommended for evaluation)
make download-nq

# Download train set (for training experiments)
make download-nq-train

# Download all splits
make download-nq-full

# Download sample for testing (1000 examples)
make download-nq-sample
```

### 2. TriviaQA

- **Source:** TriviaQA dataset
- **Description:** Trivia questions from quiz leagues
- **Splits:** train (~78k), validation (~8k), test (~11k)
- **Format:** Open-domain QA with multiple answer aliases

```bash
# Download validation set
make download-triviaqa

# Download all splits
make download-triviaqa-full
```

### 3. HotpotQA

- **Source:** HotpotQA dataset
- **Description:** Multi-hop reasoning questions requiring multiple documents
- **Splits:** train (~90k), validation (~7k)
- **Format:** Open-domain QA with question types (bridge/comparison)

```bash
# Download validation set
make download-hotpotqa

# Download all splits
make download-hotpotqa-full
```

## Dataset Commands

### Download Commands

```bash
# Natural Questions
make download-nq              # Validation split (~8k examples)
make download-nq-train        # Train split (~300k examples)
make download-nq-full         # All splits
make download-nq-sample       # Sample (1000 examples)

# TriviaQA
make download-triviaqa        # Validation split
make download-triviaqa-full   # All splits

# HotpotQA
make download-hotpotqa        # Validation split
make download-hotpotqa-full   # All splits

# All datasets
make download-all             # All datasets (validation only)
make download-all-full        # All datasets (all splits) ⚠️ Large!
```

### Management Commands

```bash
# List downloaded datasets
make list-datasets

# Clean downloaded datasets
make clean-datasets

# Clean everything (including datasets)
make clean-all
```

## Dataset Features

### Automatic Filtering

By default, the download script filters out questions without explicit answers:

```bash
# Questions WITHOUT answers are filtered out by default
make download-nq
```

To include all questions (even those without answers):

```bash
python experiments/scripts/download_datasets.py \
    --dataset natural_questions \
    --split validation \
    --no-filter
```

### Preprocessing

The download script automatically:

1. **Downloads** from HuggingFace datasets
2. **Filters** questions without answers (optional)
3. **Standardizes** format (all datasets → same structure)
4. **Caches** locally for fast loading
5. **Shows statistics** (counts, sample questions)

### Dataset Format

Downloaded datasets are saved as JSON with this structure:

```json
{
  "info": {
    "dataset": "natural_questions",
    "split": "validation",
    "original_size": 7842,
    "filtered_size": 7830,
    "filtered_out": 12,
    "filter_no_answer": true
  },
  "examples": [
    {
      "id": "nq_validation_0",
      "question": "who wrote the song i can see clearly now",
      "answers": ["Johnny Nash"],
      "source": "natural_questions",
      "split": "validation"
    },
    ...
  ]
}
```

## Storage & Caching

### Local Storage

Downloaded datasets are saved to:

```
data/datasets/
├── natural_questions_validation.json    (~2-5 MB)
├── natural_questions_train.json         (~50-100 MB)
├── triviaqa_validation.json             (~2-5 MB)
└── hotpotqa_validation.json             (~2-5 MB)
```

### HuggingFace Cache

HuggingFace datasets are also cached (typically in `~/.cache/huggingface/`):

- Natural Questions train: ~10 GB
- TriviaQA full: ~8 GB
- HotpotQA full: ~15 GB

**Note:** The first download will take longer as HuggingFace caches the dataset. Subsequent runs are much faster!

## Advanced Usage

### Command-Line Script

You can also use the script directly for more control:

```bash
# Download specific dataset and split
uv run python experiments/scripts/download_datasets.py \
    --dataset natural_questions \
    --split validation \
    --output-dir data/datasets

# Limit number of examples (for testing)
uv run python experiments/scripts/download_datasets.py \
    --dataset natural_questions \
    --split validation \
    --max-examples 100

# Download without saving (just show stats)
uv run python experiments/scripts/download_datasets.py \
    --dataset natural_questions \
    --split validation \
    --no-save

# Custom cache directory
uv run python experiments/scripts/download_datasets.py \
    --dataset natural_questions \
    --split validation \
    --cache-dir /path/to/cache
```

### Using Downloaded Datasets

The existing dataset loaders automatically work with or without downloaded files. They will:

1. Try to load from HuggingFace datasets (default)
2. Apply any filtering/preprocessing

**No code changes needed!** The download step is optional but recommended for:

- Faster loading times
- Offline access
- Preprocessing/filtering
- Version control of exact dataset used

## Common Workflows

### For Quick Experiments

```bash
# Use HuggingFace directly (no download needed)
make eval-baseline-quick
```

The dataset will be loaded on-demand from HuggingFace.

### For Reproducibility

```bash
# 1. Download and preprocess datasets
make download-nq
make download-triviaqa

# 2. Run experiments (uses cached versions)
make eval-baseline-full

# 3. Dataset files can be version-controlled or shared
# (they're just JSON files)
```

### For Offline Development

```bash
# 1. Download everything once (with internet)
make download-all-full

# 2. Work offline - datasets are cached locally
```

### For Large-Scale Training

```bash
# Download full training set
make download-nq-train

# Use in training experiments
python experiments/scripts/run_experiment.py \
    --config experiments/configs/my_training_config.yaml \
    --mode train
```

## Troubleshooting

### Slow Download

**Problem:** Download is very slow

**Solution:**
- First download caches data from HuggingFace (~GB of data)
- Subsequent runs are much faster (uses cache)
- Consider downloading smaller datasets first (`download-nq-sample`)

### Disk Space

**Problem:** Running out of disk space

**Solution:**
```bash
# Clean downloaded JSON files
make clean-datasets

# Clean HuggingFace cache (more aggressive)
rm -rf ~/.cache/huggingface/datasets/
```

### Missing jq

**Problem:** `make list-datasets` shows "?" for counts

**Solution:**
```bash
# Install jq for JSON parsing
sudo apt-get install jq  # Ubuntu/Debian
brew install jq          # macOS
```

## Dataset Statistics

Quick reference for planning experiments:

| Dataset | Split | Size | Filtered | Avg Q Length |
|---------|-------|------|----------|--------------|
| Natural Questions | train | ~300k | ~280k | ~40 chars |
| Natural Questions | validation | ~8k | ~7.8k | ~40 chars |
| TriviaQA | train | ~78k | ~78k | ~45 chars |
| TriviaQA | validation | ~8k | ~8k | ~45 chars |
| HotpotQA | train | ~90k | ~90k | ~50 chars |
| HotpotQA | validation | ~7k | ~7k | ~50 chars |

**Filtered:** Examples with explicit answers (after filtering)

## How It Works (Under the Hood)

The dataset classes inherit caching methods from `QADataset`:

```python
class NaturalQuestionsDataset(QADataset):
    def __init__(self, split="train", use_cache=True, **kwargs):
        super().__init__(name="natural_questions", split=split, **kwargs)
        
        # Try cache first (automatic!)
        if use_cache and self.load_from_cache():
            return  # ✓ Loaded from cache
        
        # Otherwise load from HuggingFace
        self.load()
```

### Cache Methods Available

All dataset classes have these methods:

- `get_cache_path()` - Get cache file path
- `save_to_cache(info)` - Save to cache
- `load_from_cache()` - Load from cache
- `download_and_cache(...)` - Explicit download with options

### Performance Impact

| Operation | Without Cache | With Cache | Speedup |
|-----------|---------------|------------|---------|
| Load NQ validation (~8k) | ~15 seconds | ~0.5 seconds | **30x** |
| Load NQ train (~280k) | ~120 seconds | ~2 seconds | **60x** |
| Load TriviaQA validation | ~20 seconds | ~0.5 seconds | **40x** |

## Implementation Details

The caching system is built directly into the dataset base class (`src/ragicamp/datasets/base.py`), following these principles:

- **Backward compatible**: Existing code works unchanged
- **Type-safe**: Uses existing `QAExample` dataclass
- **Optional**: Can be disabled with `use_cache=False`
- **Automatic**: Transparent to users
- **Clean**: No separate download scripts needed

## Next Steps

- **[Evaluation Guide](CONFIG_BASED_EVALUATION.md)** - Run experiments with datasets
- **[Metrics Guide](METRICS.md)** - Evaluate model performance
- **[Architecture](../ARCHITECTURE.md)** - Understand dataset loaders
- **[Example Script](../../examples/dataset_download_example.py)** - See usage examples

---

**Questions?** Check the main [README](../../README.md) or [TROUBLESHOOTING](../TROUBLESHOOTING.md) guide.

