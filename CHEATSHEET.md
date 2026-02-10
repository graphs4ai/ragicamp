# RAGiCamp Cheatsheet

Quick reference for common tasks.

---

## Quick Start

```bash
uv sync                                                          # Install
uv run ragicamp run conf/study/smart_retrieval_slm.yaml --dry-run  # Preview
uv run ragicamp run conf/study/smart_retrieval_slm.yaml            # Run
uv run ragicamp health outputs/smart_retrieval_slm                 # Check status
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `ragicamp run <config>` | Run study from YAML config |
| `ragicamp health <dir>` | Check experiment health |
| `ragicamp resume <dir>` | Resume incomplete experiments |
| `ragicamp metrics <dir> -m <metrics>` | Recompute metrics |
| `ragicamp compare <dir>` | Compare results |
| `ragicamp evaluate <file>` | Compute metrics on predictions |
| `ragicamp index` | Build retrieval index |
| `ragicamp cache stats` | Show cache statistics |
| `ragicamp cache clear` | Clear embedding cache |
| `ragicamp backup` | Backup to Backblaze B2 |
| `ragicamp download` | Download from Backblaze B2 |

### Run Options

```bash
ragicamp run <config.yaml> [OPTIONS]
  --dry-run          Preview experiments and their status
  --skip-existing    Skip completed experiments
  --validate         Validate config only
  --limit N          Max examples per experiment
  --force            Force re-run even if complete/failed
  --sample N         Sample N experiments (random or TPE)
```

### Compare Options

```bash
ragicamp compare <output_dir> [OPTIONS]
  --metric, -m       Metric to compare (default: f1)
  --group-by, -g     model | dataset | prompt | retriever | type
  --pivot A B        Create pivot table (rows=A, cols=B)
  --top N            Show top N results (default: 10)
```

---

## Study Config

```yaml
name: my_study
num_questions: 1000
datasets: [nq, triviaqa, hotpotqa]
metrics: [f1, exact_match, bertscore]

direct:
  enabled: true
  models:
    - vllm:Qwen/Qwen2.5-7B-Instruct
  prompts: [concise, fewshot_3]

rag:
  enabled: true
  models:
    - vllm:Qwen/Qwen2.5-7B-Instruct

  sampling:
    mode: tpe
    n_experiments: 1000
    optimize_metric: f1
    agent_types: [fixed_rag, iterative_rag, self_rag]

  retriever_names: [dense_bge_large_512, hybrid_bge_large_bm25]
  top_k_values: [3, 5, 10, 15, 20]
  query_transform: [none, hyde, multiquery]
  prompts: [concise, extractive, cot, fewshot_3]

output_dir: outputs/my_study
batch_size: 128
```

Full example: `conf/study/smart_retrieval_slm.yaml`

---

## Model Specs

| Provider | Format | Example |
|----------|--------|---------|
| vLLM | `vllm:org/model` | `vllm:Qwen/Qwen2.5-7B-Instruct` |
| HuggingFace | `hf:org/model` | `hf:google/gemma-2-2b-it` |
| OpenAI | `openai:model` | `openai:gpt-4o-mini` |

---

## Agent Types

| Agent | Description |
|-------|-------------|
| `direct_llm` | No retrieval — LLM-only baseline |
| `fixed_rag` | Standard single-pass RAG |
| `iterative_rag` | Multi-iteration with sufficiency checking |
| `self_rag` | Adaptive — model decides when to retrieve |

---

## Retriever Types

| Type | Description |
|------|-------------|
| `dense` | FAISS HNSW index with embedding model |
| `hybrid` | Dense + sparse (BM25/TF-IDF) with RRF fusion |
| `hierarchical` | Search small child chunks, return larger parents |

---

## Query Transforms

| Transform | Description |
|-----------|-------------|
| `none` | Use original query (enables retrieval caching) |
| `hyde` | Generate hypothetical doc, embed that instead |
| `multiquery` | Generate multiple query variations |

---

## Metrics

| Metric | Type | GPU |
|--------|------|-----|
| `f1` | Token overlap | No |
| `exact_match` | String match | No |
| `bertscore` | Semantic similarity | Yes |
| `bleurt` | Learned metric | Yes |
| `llm_judge` | LLM-as-judge | No (API) |
| `faithfulness` | Context grounding | No (API) |
| `hallucination` | Hallucination detection | No (API) |

---

## Prompt Strategies

| Prompt | Description |
|--------|-------------|
| `concise` | Minimal "just answer" |
| `concise_strict` | Anti-hallucination (STOP instruction) |
| `extractive` | Strict context-only |
| `extractive_quoted` | Force verbatim quotes |
| `cot` | Chain-of-thought reasoning |
| `cot_final` | CoT with "FINAL ANSWER:" marker |
| `fewshot_1` | 1 in-context example |
| `fewshot_3` | 3 in-context examples |

---

## Output Structure

```
outputs/my_study/rag_Qwen257BI_nq_a3f8c2d1/
├── state.json          # Phase tracking
├── questions.json      # Exported questions
├── metadata.json       # Full experiment config
├── predictions.json    # Answers + per-item metrics
├── results.json        # Aggregate metrics + timing
└── experiment.log      # Subprocess output
```

---

## Utility Scripts

```bash
# Validate cache integrity
python scripts/validate_cache.py -v --cross-check --live

# Migrate old experiment names to hash-based
python scripts/migrate_naming.py --study outputs/my_study          # dry-run
python scripts/migrate_naming.py --study outputs/my_study --execute

# Clean incorrectly named experiments
python scripts/clean_study.py --study outputs/my_study
python scripts/clean_study.py --study outputs/my_study --execute

# Quarantine tainted experiments
python scripts/quarantine_affected_exps.py outputs/my_study

# Repair metadata
python scripts/repair_metadata.py outputs/my_study
```

---

## Analysis Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_overview` | Dataset statistics |
| `02_rag_effectiveness` | RAG vs Direct comparison |
| `03_component_analysis` | Ablation by retriever/prompt/model |
| `05_failure_analysis` | Error patterns |
| `06_pipeline_profiling` | Phase and metric timing |
| `rag_strategy_analysis` | Agent strategy comparison |

---

## Make Targets

```bash
make install                # uv sync
make test                   # pytest
make lint                   # ruff check
make format                 # ruff format + fix
make pre-push               # format + lint + test
make index-simple           # 1 small index (500 docs)
make index-full             # All standard indexes
make run-baseline-simple    # Quick 10-question test
make evaluate DIR=... METRICS=bertscore,bleurt
make compare DIR=... SORT=f1
make clean                  # rm -rf outputs/
```

---

## Caching

Both caches live in `artifacts/cache/ragicamp_cache.db` (SQLite, WAL mode).

**Embedding cache** — `(model_name, sha256(text))` → float32 vector. Shared across all experiments using the same embedding model.

**Retrieval cache** — `(retriever_name, query_hash, top_k)` → search results. Only populated when `query_transform=none` (HyDE/multiquery produce LLM-dependent queries). Shared across experiments differing only in LLM model or prompt.

```bash
ragicamp cache stats       # Check cache size
python scripts/validate_cache.py -v  # Validate integrity
```

Override path: `RAGICAMP_CACHE_DIR=/path/to/dir`

---

## Tips

- **Resume after crash** — Just re-run the same command; experiments auto-resume
- **Phase-aware resume** — Crashed in metrics? Won't reload the model
- **Subprocess isolation** — CUDA crashes don't kill the orchestrator
- **Dry-run first** — Always `--dry-run` before large studies
- **Limit for testing** — `--limit 10` runs only 10 questions per experiment
- **Retrieval cache** — qt=none experiments cache retrievals, reused across models/prompts
- **GPU cleanup** — `ResourceManager.clear_gpu_memory()` between experiments

---

## Development

```bash
uv run pytest tests/ -x -q     # Fast test run
uv run ruff check src/         # Lint
uv run ruff format src/        # Format
```

Ruff config: line-length=100, target=py310
