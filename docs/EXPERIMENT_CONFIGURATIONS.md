# RAGiCamp Experiment Configurations

This document outlines the experiments we should run to systematically improve RAG performance.

---

## Part 1: Embedding Models for Dense Retrieval

Based on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (2026), here are embedding models worth testing:

### Tier 1: High Performance (Large Models)

| Model | Retrieval Score | Size | Dimensions | Notes |
|-------|-----------------|------|------------|-------|
| **gte-Qwen2-7B-instruct** | 60.25 | ~14GB | 4096 | Best overall, but very large |
| **NV-Embed-v1** (NVIDIA) | 59.36 | ~8GB | 4096 | Strong retrieval, needs GPU |
| **Linq-Embed-Mistral** | 60.19 | ~7GB | 4096 | Excellent retrieval score |
| **SFR-Embedding-Mistral** | ~58 | ~7GB | 4096 | Salesforce, good quality |

**Use case**: When you have ample GPU memory and prioritize quality over speed.

### Tier 2: Balanced (Medium Models) - RECOMMENDED

| Model | Retrieval Score | Size | Dimensions | Notes |
|-------|-----------------|------|------------|-------|
| **BAAI/bge-large-en-v1.5** | ~54 | ~1.3GB | 1024 | Current default, solid baseline |
| **BAAI/bge-m3** | ~55 | ~2.3GB | 1024 | Supports dense+sparse+multi-vector |
| **voyage-large-2-instruct** | 58.28 | API | 1024 | API-based, high quality |
| **nomic-embed-text-v1.5** | 59.4 | ~0.5GB | 768 | Good quality, efficient |

**Use case**: Production systems with moderate GPU memory (8-16GB).

### Tier 3: Efficient (Small Models)

| Model | Retrieval Score | Size | Dimensions | Notes |
|-------|-----------------|------|------------|-------|
| **all-MiniLM-L6-v2** | ~56 | ~80MB | 384 | Fast prototyping, lightweight |
| **mxbai-embed-xsmall-v1** | ~55 | ~50MB | 384 | Smallest efficient option |
| **BAAI/bge-small-en-v1.5** | ~51 | ~130MB | 384 | BGE family, fast |

**Use case**: CPU-only environments, rapid iteration, limited memory.

### Special: Hybrid-Native Model

| Model | Features | Size | Notes |
|-------|----------|------|-------|
| **BAAI/bge-m3** | Dense + Sparse + Multi-vector | ~2.3GB | Native hybrid retrieval support |

BGE-M3 is unique because it produces both dense and sparse representations in a single forward pass, potentially making hybrid retrieval more efficient than running separate dense and sparse models.

---

## Part 2: Experiments to Run

### Phase A: Baseline Comparisons

Before testing advanced strategies, establish baselines.

```yaml
# A1: DirectLLM baseline (no retrieval)
- name: a1_direct_llm
  hypothesis: "Model's own knowledge baseline"
  agent_type: direct
  model: hf:meta-llama/Llama-3.2-3B-Instruct
  prompt: concise

# A2: Simple RAG baseline
- name: a2_rag_baseline
  hypothesis: "Standard RAG with default settings"
  agent_type: fixed_rag
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 512
  top_k: 5
  prompt: concise
```

---

### Phase B: Embedding Model Comparison

Test different embedding models with identical RAG settings.

```yaml
# B1: BGE Large (current default)
- name: b1_bge_large
  hypothesis: "BGE-large baseline embedding"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 512
  top_k: 5

# B2: BGE-M3 (newer, multilingual)
- name: b2_bge_m3
  hypothesis: "BGE-M3 may capture more semantic nuance"
  retriever:
    type: dense
    embedding_model: BAAI/bge-m3
    chunk_size: 512
  top_k: 5

# B3: Nomic (efficient, high quality)
- name: b3_nomic
  hypothesis: "Nomic offers good quality at lower cost"
  retriever:
    type: dense
    embedding_model: nomic-ai/nomic-embed-text-v1.5
    chunk_size: 512
  top_k: 5

# B4: MiniLM (fast baseline)
- name: b4_minilm
  hypothesis: "MiniLM as fast/cheap baseline"
  retriever:
    type: dense
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    chunk_size: 512
  top_k: 5
```

---

### Phase C: Chunk Size & Strategy

Test how document chunking affects retrieval quality.

```yaml
# C1: Small chunks (precise matching)
- name: c1_chunks_256
  hypothesis: "Smaller chunks = more precise retrieval"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 256
    chunk_overlap: 25
  top_k: 5

# C2: Medium chunks (baseline)
- name: c2_chunks_512
  hypothesis: "Standard chunk size"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 512
    chunk_overlap: 50
  top_k: 5

# C3: Large chunks (more context)
- name: c3_chunks_1024
  hypothesis: "Larger chunks provide more context per doc"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 1024
    chunk_overlap: 100
  top_k: 3  # Fewer docs since larger

# C4: Very large chunks
- name: c4_chunks_2048
  hypothesis: "Maximize context per retrieval"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 2048
    chunk_overlap: 200
  top_k: 2  # Just 2 large docs

# C5: Paragraph-based chunking
- name: c5_paragraph
  hypothesis: "Respecting paragraph boundaries improves coherence"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 1024
    chunking_strategy: paragraph
  top_k: 3

# C6: Sentence-based chunking
- name: c6_sentence
  hypothesis: "Sentence boundaries improve semantic units"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
    chunk_size: 5  # 5 sentences per chunk
    chunking_strategy: sentence
  top_k: 5
```

---

### Phase D: Retrieval Strategies

Compare retrieval approaches.

```yaml
# D1: Dense only
- name: d1_dense
  hypothesis: "Pure semantic retrieval"
  retriever:
    type: dense
    embedding_model: BAAI/bge-large-en-v1.5
  top_k: 5

# D2: Hybrid (dense + sparse)
- name: d2_hybrid
  hypothesis: "Hybrid catches keyword matches dense misses"
  retriever:
    type: hybrid
    embedding_model: BAAI/bge-large-en-v1.5
    alpha: 0.5  # Equal weight
  top_k: 5

# D3: Hybrid (dense-heavy)
- name: d3_hybrid_dense
  hypothesis: "Favor dense, use sparse as tiebreaker"
  retriever:
    type: hybrid
    embedding_model: BAAI/bge-large-en-v1.5
    alpha: 0.7  # 70% dense
  top_k: 5

# D4: Hybrid (sparse-heavy)
- name: d4_hybrid_sparse
  hypothesis: "Favor keyword matching"
  retriever:
    type: hybrid
    embedding_model: BAAI/bge-large-en-v1.5
    alpha: 0.3  # 30% dense
  top_k: 5

# D5: Hierarchical
- name: d5_hierarchical
  hypothesis: "Search precise, return broad context"
  retriever:
    type: hierarchical
    embedding_model: BAAI/bge-large-en-v1.5
    parent_chunk_size: 2048
    child_chunk_size: 256
  top_k: 3
```

---

### Phase E: Top-K and Noise Reduction

Test how much context helps vs. hurts.

```yaml
# E1: Minimal (single best)
- name: e1_topk_1
  hypothesis: "Single best document minimizes noise"
  retriever: dense_bge
  top_k: 1

# E2: Conservative
- name: e2_topk_3
  hypothesis: "3 documents balances info vs noise"
  retriever: dense_bge
  top_k: 3

# E3: Standard
- name: e3_topk_5
  hypothesis: "5 documents (default)"
  retriever: dense_bge
  top_k: 5

# E4: Rich context
- name: e4_topk_10
  hypothesis: "More context if model can handle it"
  retriever: dense_bge
  top_k: 10

# E5: Rerank minimal (fetch many, keep few)
- name: e5_rerank_minimal
  hypothesis: "Overfetch + rerank = best precision"
  retriever: dense_bge
  top_k: 1
  fetch_k: 20
  reranker: bge

# E6: Rerank standard
- name: e6_rerank_standard
  hypothesis: "Rerank top 3 from 15"
  retriever: dense_bge
  top_k: 3
  fetch_k: 15
  reranker: bge
```

---

### Phase F: Query Transformation

Test query enhancement strategies.

```yaml
# F1: No transformation
- name: f1_no_transform
  hypothesis: "Raw query baseline"
  retriever: dense_bge
  query_transform: none
  top_k: 5

# F2: HyDE (Hypothetical Document Embeddings)
- name: f2_hyde
  hypothesis: "LLM-generated answer helps match vocabulary"
  retriever: dense_bge
  query_transform: hyde
  top_k: 5

# F3: Multi-query
- name: f3_multiquery
  hypothesis: "Multiple query variations increase recall"
  retriever: dense_bge
  query_transform: multiquery
  top_k: 5

# F4: HyDE + Rerank
- name: f4_hyde_rerank
  hypothesis: "Transform + filter = best quality"
  retriever: dense_bge
  query_transform: hyde
  top_k: 3
  fetch_k: 10
  reranker: bge
```

---

### Phase G: Reranker Comparison

Test different reranking models.

```yaml
# G1: No reranker
- name: g1_no_rerank
  hypothesis: "Baseline without reranking"
  retriever: dense_bge
  top_k: 5
  reranker: none

# G2: BGE Reranker (high quality)
- name: g2_bge_reranker
  hypothesis: "BGE reranker for quality"
  retriever: dense_bge
  top_k: 5
  fetch_k: 20
  reranker: bge  # BAAI/bge-reranker-large

# G3: MS-MARCO (faster)
- name: g3_msmarco_reranker
  hypothesis: "MS-MARCO reranker for speed"
  retriever: dense_bge
  top_k: 5
  fetch_k: 20
  reranker: ms-marco  # cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

### Phase H: Agent Architectures (After Implementation)

Test different RAG agent types.

```yaml
# H1: Fixed RAG (baseline)
- name: h1_fixed_rag
  hypothesis: "Standard retrieve-then-generate"
  agent_type: fixed_rag
  retriever: dense_bge
  top_k: 5

# H2: Iterative RAG
- name: h2_iterative_1iter
  hypothesis: "Refine query once after initial retrieval"
  agent_type: iterative_rag
  retriever: dense_bge
  top_k: 5
  iterative:
    max_iterations: 1

# H3: Iterative RAG (2 iterations)
- name: h3_iterative_2iter
  hypothesis: "Two refinement rounds for complex queries"
  agent_type: iterative_rag
  retriever: dense_bge
  top_k: 5
  iterative:
    max_iterations: 2

# H4: Self-RAG (conservative)
- name: h4_selfrag_conservative
  hypothesis: "Model decides retrieval, high threshold"
  agent_type: self_rag
  retriever: dense_bge
  top_k: 5
  self_rag:
    retrieval_threshold: 0.7  # Only retrieve when confident
    verify_answer: false

# H5: Self-RAG (aggressive)
- name: h5_selfrag_aggressive
  hypothesis: "Model decides, but leans toward retrieval"
  agent_type: self_rag
  retriever: dense_bge
  top_k: 5
  self_rag:
    retrieval_threshold: 0.3
    verify_answer: false

# H6: Self-RAG with verification
- name: h6_selfrag_verify
  hypothesis: "Verify answer is supported by context"
  agent_type: self_rag
  retriever: dense_bge
  top_k: 5
  self_rag:
    retrieval_threshold: 0.5
    verify_answer: true

# H7: Context summarization
- name: h7_summarize
  hypothesis: "Summarize long context before answering"
  agent_type: fixed_rag
  retriever: dense_bge
  top_k: 10  # Get many docs
  context_processing:
    strategy: summarize
    max_tokens: 1024
```

---

### Phase I: Prompt Engineering

Test different prompting strategies.

```yaml
# I1: Concise
- name: i1_concise
  hypothesis: "Minimal instructions"
  prompt: concise

# I2: Few-shot (1 example)
- name: i2_fewshot_1
  hypothesis: "One example teaches format"
  prompt: fewshot_1

# I3: Few-shot (3 examples)
- name: i3_fewshot_3
  hypothesis: "More examples improve consistency"
  prompt: fewshot_3

# I4: CoT (chain of thought)
- name: i4_cot
  hypothesis: "Reasoning helps complex questions"
  prompt: cot
```

---

## Part 3: Recommended Experiment Order

### Round 1: Quick Baseline (1-2 hours)

Run these first to establish baselines:

| # | Experiment | What it tests |
|---|------------|---------------|
| 1 | a1_direct_llm | Model without retrieval |
| 2 | a2_rag_baseline | Standard RAG |
| 3 | e1_topk_1 | Minimal context |
| 4 | e3_topk_5 | Standard context |
| 5 | d2_hybrid | Hybrid vs dense |

**Analysis**: Compare direct vs RAG. If RAG hurts, focus on noise reduction.

---

### Round 2: Chunk Size Exploration (2-3 hours)

```
c1_chunks_256 → c2_chunks_512 → c3_chunks_1024 → c4_chunks_2048
```

**Analysis**: Find the chunk size that maximizes F1/exact_match.

---

### Round 3: Noise Reduction (2-3 hours)

```
e1_topk_1 → e5_rerank_minimal → e6_rerank_standard
```

**Analysis**: Does reranking help? What's the optimal fetch_k/top_k ratio?

---

### Round 4: Query Enhancement (2-3 hours)

```
f1_no_transform → f2_hyde → f3_multiquery → f4_hyde_rerank
```

**Analysis**: Does query transformation improve retrieval? Is the latency worth it?

---

### Round 5: Embedding Models (3-4 hours)

```
b1_bge_large → b2_bge_m3 → b3_nomic → b4_minilm
```

**Analysis**: Quality vs efficiency tradeoff. Is BGE-M3 worth the extra memory?

---

### Round 6: Advanced Agents (After Implementation)

```
h1_fixed_rag → h2_iterative → h4_selfrag_conservative → h6_selfrag_verify
```

**Analysis**: Do advanced strategies help the "context doesn't help" problem?

---

## Part 4: Metrics to Track

For each experiment, track:

| Metric | What it measures |
|--------|------------------|
| **exact_match** | Binary correctness |
| **f1** | Token overlap quality |
| **llm_judge_qa** | Semantic correctness (GPT-4 judgment) |
| **retrieval_precision** | Are retrieved docs relevant? |
| **retrieval_recall** | Are all relevant docs retrieved? |
| **latency_ms** | End-to-end response time |
| **tokens_used** | Context length efficiency |

---

## Part 5: Analysis Dimensions

When comparing experiments, analyze along these dimensions:

### By Query Type
- Factual questions (who, what, when)
- Reasoning questions (why, how)
- Multi-hop questions (require combining info)

### By Context Quality
- High-quality context (docs clearly answer the question)
- Partial context (docs are related but don't answer directly)
- Misleading context (docs contain wrong information)

### By Model Behavior
- Uses context appropriately
- Ignores context (answers from knowledge)
- Says "I don't know" or "context doesn't help"

---

## Part 6: Experiment Config Template

Use this template for the singleton experiment format:

```yaml
name: experiment_study_name
description: "Purpose of this experiment batch"

num_questions: 100
datasets: [nq]

models: &model
  - hf:meta-llama/Llama-3.2-3B-Instruct

direct:
  enabled: true
  models: *model
  prompts: [concise]

rag:
  enabled: true
  
  corpus:
    source: wikimedia/wikipedia
    version: 20231101.en
    max_docs: 150000

  experiments:
    - name: your_experiment_name
      hypothesis: "What you expect to learn"
      model: *model
      retriever:
        type: dense|hybrid|hierarchical
        embedding_model: BAAI/bge-large-en-v1.5
        chunk_size: 512
        chunk_overlap: 50
        chunking_strategy: recursive  # fixed, sentence, paragraph, recursive
      top_k: 5
      fetch_k: null  # Optional: for reranking
      query_transform: none  # none, hyde, multiquery
      reranker: none  # none, bge, ms-marco
      prompt: concise
      dataset: nq
      # For advanced agents (when implemented):
      # agent_type: fixed_rag|iterative_rag|self_rag
      # iterative:
      #   max_iterations: 2
      # self_rag:
      #   retrieval_threshold: 0.5
      #   verify_answer: false

metrics:
  - exact_match
  - f1
  - llm_judge_qa

llm_judge:
  model: openai:gpt-4o-mini

output_dir: outputs/your_study_name
batch_size: 64
```

---

## References

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding model benchmarks
- [BGE-M3 Documentation](https://bge-model.com/bge/bge_m3.html) - Hybrid embedding model
- [MTEB Info](https://mteb.info/) - Detailed benchmark results
