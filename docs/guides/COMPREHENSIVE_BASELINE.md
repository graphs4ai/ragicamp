# Comprehensive Baseline Study Guide

This guide walks through running a full baseline study with multiple models and retrieval configurations.

## Overview

| Dimension | Options |
|-----------|---------|
| **Datasets** | NQ, TriviaQA, HotPotQA (all from Wikipedia) |
| **DirectLLM Models** | 2 OpenAI + 4 HuggingFace = 6 models |
| **RAG Models** | 1 OpenAI + 1 HuggingFace = 2 models |
| **Retrievers** | 6 index variations |
| **Prompts** | 3 (default, concise, detailed) |
| **Top-k** | 3, 5, 10 |

## Step 1: Build Indexes (One-time)

You need 6 indexes across 2 corpora:

```bash
# === Simple Wikipedia (faster, ~200k articles) ===
# Good for initial validation

# 1. MiniLM + 512 chunks (fastest embedding)
python experiments/scripts/index_corpus.py \
  --corpus-name wikipedia_simple \
  --corpus-version 20231101.simple \
  --embedding-model all-MiniLM-L6-v2 \
  --artifact-name simple_minilm_recursive_512 \
  --chunk-strategy recursive \
  --chunk-size 512

# 2. MiniLM + 1024 chunks
python experiments/scripts/index_corpus.py \
  --corpus-name wikipedia_simple \
  --corpus-version 20231101.simple \
  --embedding-model all-MiniLM-L6-v2 \
  --artifact-name simple_minilm_recursive_1024 \
  --chunk-strategy recursive \
  --chunk-size 1024

# 3. MPNet + 512 chunks (better embeddings)
python experiments/scripts/index_corpus.py \
  --corpus-name wikipedia_simple \
  --corpus-version 20231101.simple \
  --embedding-model all-mpnet-base-v2 \
  --artifact-name simple_mpnet_recursive_512 \
  --chunk-strategy recursive \
  --chunk-size 512


# === Full English Wikipedia (production, ~6M articles) ===
# Takes ~2-4 hours to index, much higher coverage

# 4. English MiniLM
python experiments/scripts/index_corpus.py \
  --corpus-name wikipedia_en \
  --corpus-version 20231101.en \
  --embedding-model all-MiniLM-L6-v2 \
  --artifact-name en_minilm_recursive_512 \
  --chunk-strategy recursive \
  --chunk-size 512 \
  --max-docs 500000  # Start with 500k for testing

# 5. English MPNet
python experiments/scripts/index_corpus.py \
  --corpus-name wikipedia_en \
  --corpus-version 20231101.en \
  --embedding-model all-mpnet-base-v2 \
  --artifact-name en_mpnet_recursive_512 \
  --chunk-strategy recursive \
  --chunk-size 512 \
  --max-docs 500000

# 6. English E5 (SOTA embeddings)
python experiments/scripts/index_corpus.py \
  --corpus-name wikipedia_en \
  --corpus-version 20231101.en \
  --embedding-model intfloat/e5-small-v2 \
  --artifact-name en_e5_recursive_512 \
  --chunk-strategy recursive \
  --chunk-size 512 \
  --max-docs 500000
```

**Or use the batch builder:**

```bash
# Build all 6 indexes at once
python scripts/data/build_all_indexes.py \
  --corpora wiki_simple wiki_en \
  --embeddings minilm mpnet e5 \
  --chunk-configs recursive_512 recursive_1024
```

## Step 2: Run Experiments

### Option A: Full Study (Recommended for Production)

```bash
# Full comprehensive baseline (all experiments)
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml
```

This runs ~162 experiments:
- DirectLLM: 6 models × 3 prompts × 3 datasets = 54 experiments
- RAG: 2 models × 6 retrievers × 3 top_k × 3 datasets = 108 experiments

### Option B: Phased Approach (Recommended)

Run in phases to validate setup and catch issues early:

```bash
# Phase 1: Quick validation (10 questions, 1 dataset)
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml \
  --override num_questions=10 \
  --override datasets=[nq] \
  --override direct.models=[openai:gpt-4o-mini,hf:google/gemma-2b-it] \
  --override rag.enabled=false

# Phase 2: DirectLLM full (no RAG yet)
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml \
  --override rag.enabled=false

# Phase 3: RAG experiments
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml \
  --override direct.enabled=false
```

### Option C: Sequential (for limited resources)

Run each model type separately:

```bash
# 1. OpenAI only (fastest, requires API key)
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml \
  --override direct.models=[openai:gpt-4o-mini,openai:gpt-4o] \
  --override rag.models=[openai:gpt-4o-mini]

# 2. HuggingFace small models
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml \
  --override direct.models=[hf:google/gemma-2b-it,hf:microsoft/phi-3-mini-4k-instruct] \
  --override rag.models=[hf:google/gemma-2b-it]

# 3. HuggingFace large models (if you have 24GB+ VRAM)
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml \
  --override direct.models=[hf:mistralai/Mistral-7B-Instruct-v0.3]
```

## Step 3: Evaluate (Heavy Metrics)

Run expensive metrics after all predictions are generated:

```bash
# Compute BERTScore, BLEURT, LLM Judge on all results
make evaluate PATH=outputs/comprehensive_baseline METRICS=bertscore,bleurt,llm_judge

# Or individually for specific experiments
python scripts/experiments/evaluate_predictions.py \
  outputs/comprehensive_baseline/direct_* \
  --metrics bertscore bleurt llm_judge
```

## Step 4: Analysis

```bash
# Generate comparison report
python scripts/eval/compare.py outputs/comprehensive_baseline --output report.json

# Visual comparison
python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline
```

## Hardware Requirements

| Model Size | VRAM Required | Quantization |
|------------|---------------|--------------|
| 2-3B (Gemma-2B, Phi-3-mini) | 6-8 GB | 4bit: 4GB |
| 7B (Mistral, Llama-3.2) | 14-16 GB | 4bit: 6GB |
| 8B (Llama-3.1) | 16-20 GB | 4bit: 8GB |

## Cost Estimate (OpenAI)

Assuming ~5000 total questions across all datasets:

| API | Cost per 1K tokens | Estimated Cost |
|-----|-------------------|----------------|
| gpt-4o-mini (input) | $0.00015 | ~$2-5 |
| gpt-4o-mini (output) | $0.0006 | ~$3-8 |
| gpt-4o (input) | $0.0025 | ~$30-50 |
| gpt-4o (output) | $0.01 | ~$50-100 |
| **LLM Judge** (for metrics) | ~$0.001/q | ~$5 |

**Total estimate**: $80-170 for full study with GPT-4o, or ~$10-20 with just GPT-4o-mini.

## Index vs Dataset Considerations

### Why Wikipedia for All Datasets?

All three datasets derive their questions from Wikipedia:

- **NQ**: Questions from Google search, answers from Wikipedia
- **TriviaQA**: Wikipedia subset explicitly uses Wikipedia articles
- **HotPotQA**: Multi-hop questions requiring 2+ Wikipedia articles

### Index Size vs Coverage Trade-off

| Index | Articles | Chunks (~512) | Build Time | Disk |
|-------|----------|---------------|------------|------|
| Simple Wiki | 200k | ~2M | 30 min | 3 GB |
| English Wiki (full) | 6M | ~60M | 4-6 hrs | 80 GB |
| English Wiki (500k) | 500k | ~5M | 1 hr | 8 GB |

**Recommendation**: Start with Simple Wikipedia, then add English Wikipedia indexes for production experiments.

## Monitoring Progress

```bash
# Watch experiment progress
watch -n 30 "ls -la outputs/comprehensive_baseline | tail -20"

# Check GPU utilization
watch -n 2 nvidia-smi

# Count completed experiments
find outputs/comprehensive_baseline -name "results.json" | wc -l
```

## Recovery from Failures

The study runner supports checkpointing:

```bash
# Resume from where you left off
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --resume

# Skip already completed experiments
python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --skip-existing
```

