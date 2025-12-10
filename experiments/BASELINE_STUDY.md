# Baseline Study: RAG vs Direct LLM

## Objective

Establish robust, reproducible baselines comparing:
1. **Direct LLM** (parametric knowledge only)
2. **Fixed RAG** (with retrieval augmentation)

Across multiple datasets, models, and configurations.

---

## Experiment Matrix

### Datasets (3)
| Dataset | Questions | Domain | Difficulty |
|---------|-----------|--------|------------|
| Natural Questions (NQ) | Real Google queries | Wikipedia | Medium |
| TriviaQA | Trivia questions | Web/Wikipedia | Easy-Medium |
| HotpotQA | Multi-hop reasoning | Wikipedia | Hard |

### Models (3)
| Model | Size | Quantization | Notes |
|-------|------|--------------|-------|
| Gemma 2B | 2B | 4-bit | Fast, good baseline |
| Phi-3 Mini | 3.8B | 4-bit | Strong reasoning |
| Llama 3 8B | 8B | 4-bit | Best quality |

### Agent Types (2 + variations)
| Agent | Variants |
|-------|----------|
| DirectLLM | 3 prompt styles |
| FixedRAG | top_k ∈ {1, 3, 5, 10} |

### Prompt Variations (3)
| Style | Description | Example Output |
|-------|-------------|----------------|
| `concise` | Short, direct answer | "Paris" |
| `sentence` | Complete sentence | "The capital of France is Paris." |
| `explained` | With reasoning | "Paris is the capital of France, known for..." |

---

## Experiment Runs

### Phase 1: DirectLLM Baselines
```
3 datasets × 3 models × 3 prompts = 27 runs
```

### Phase 2: FixedRAG Variations
```
3 datasets × 3 models × 4 top_k × 1 prompt = 36 runs
```

### Phase 3: Best Config Deep Dive
- Best model + dataset combination
- Prompt style comparison on RAG
- Corpus size comparison (10k vs full)

**Total: ~65 runs** (can be parallelized)

---

## Metrics

| Metric | Type | Notes |
|--------|------|-------|
| Exact Match | Token | Binary, strict |
| F1 | Token | Partial credit |
| LLM Judge | Semantic | GPT-4 evaluation |
| Faithfulness | RAG | Ragas (RAG only) |

---

## Run Commands

### Quick Test (verify setup)
```bash
make baseline-study-test
```

### Full DirectLLM Sweep
```bash
make baseline-study-direct
```

### Full RAG Sweep
```bash
make baseline-study-rag
```

### Complete Study
```bash
make baseline-study-full
```

---

## Expected Outputs

```
outputs/baseline_study/
├── YYYY-MM-DD_HH-MM-SS/          # Timestamped run
│   ├── config.yaml               # Full Hydra config
│   ├── results.json              # Metrics
│   └── predictions.json          # All predictions
├── summary/
│   ├── comparison.csv            # All results
│   ├── report.html               # Visual report
│   └── best_configs.json         # Top configurations
└── plots/
    ├── model_comparison.png
    ├── dataset_comparison.png
    └── rag_vs_direct.png
```

---

## Reproducibility

All experiments use:
- **Fixed seed**: 42
- **Saved configs**: Full Hydra config saved with each run
- **MLflow tracking**: All metrics logged
- **Git commit**: Recorded in metadata

To reproduce any run:
```bash
python -m ragicamp.cli.run --config-path outputs/baseline_study/run_001/.hydra --config-name config
```
