# SOTA Gap Analysis: RAGiCamp vs Current RAG Ecosystem

> **Last updated**: 2026-02-18
>
> This document analyzes what RAGiCamp implements vs what exists in the broader
> RAG research ecosystem, identifies gaps, and documents why certain techniques
> are intentionally out of scope.

---

## What We Have vs What Exists

| Category | RAGiCamp | SOTA / Ecosystem | Gap? |
|----------|----------|------------------|------|
| **Dense Retrieval** | FAISS (Flat, IVF, HNSW) + BGE/BGE-M3 | ColBERT, PLAID, DPR | Yes (ColBERT) |
| **Sparse Retrieval** | BM25 (bm25s) | SPLADE, DeepImpact, uniCOIL | Yes (learned sparse) |
| **Hybrid Retrieval** | RRF fusion (dense + sparse) | Learned fusion, adaptive weighting | Minor |
| **Reranking** | Cross-encoder (BGE-reranker family) | Listwise rerankers, RankGPT | Minor |
| **Query Transform** | HyDE, MultiQuery | Query decomposition, step-back prompting | Minor |
| **RAG Agents** | Direct, Fixed, Iterative, Self-RAG | CRAG, Adaptive-RAG, GraphRAG | Partial |
| **Chunking** | Fixed-size + hierarchical (parent-child) | Semantic chunking, proposition-based | Yes |
| **Indexing** | Single-vector per chunk | Multi-vector (ColBERT), proposition indexing | Yes |
| **Answer Metrics** | EM, F1, BERTScore, BLEURT, LLM-judge | Same (standard) | No |
| **RAG Metrics** | Faithfulness, Hallucination, AnswerInContext, ContextRecall | RAGAS full suite, ARES, ALCE | Partial |
| **Caching** | Embedding + retrieval SQLite cache | Standard | No |
| **Experiment Mgmt** | Optuna HPO, subprocess isolation, phase-aware resume | Standard | No |

---

## Detailed Analysis

### Retrieval: Not a Blunder

**ColBERT / Late Interaction**
- Multi-vector representations with MaxSim scoring
- Significantly better quality than single-vector, especially for long documents
- **Why out of scope**: Requires fundamentally different index structure (token-level
  embeddings, PLAID compression). Index size is ~5x larger. The thesis focus is on
  comparing RAG *strategies* (agent types, query transforms, reranking), not retrieval
  backends. Adding ColBERT would confound the comparison.

**SPLADE / Learned Sparse**
- Learns term importance weights, more semantic than BM25
- **Why out of scope**: Requires model fine-tuning infrastructure and special index
  format. BM25 + dense hybrid already captures the sparse+dense paradigm. SPLADE
  would be an incremental improvement to the sparse component, not a new dimension.

### Chunking: Acknowledged Gap

**Semantic Chunking**
- Splits on topic boundaries using embedding similarity
- Would improve retrieval quality for variable-length documents
- **Status**: Listed in FUTURE_WORK.md Phase 5.3. Not blocking thesis experiments
  since we use consistent chunking across all comparisons.

**Proposition-Based Indexing**
- Decomposes chunks into atomic facts
- Much more precise retrieval but 5-10x more items
- **Status**: Research extension (FUTURE_WORK.md Phase 8.2). Would be interesting
  follow-up work but changes the experimental setup fundamentally.

### RAG Agents: Intentional Scope

**GraphRAG / Knowledge Graphs**
- Structures documents into entity-relation graphs
- Excels at multi-hop reasoning and global queries
- **Why out of scope**: Completely different paradigm from vector retrieval.
  Would require graph construction pipeline, entity extraction, and graph
  traversal logic. Thesis scope is vector-based RAG strategies.

**CRAG (Corrective RAG)**
- Detects poor retrieval and falls back to web search
- **Why out of scope**: Similar to Self-RAG (which we implement) but with web
  search fallback. The retrieval quality assessment aspect is partially covered
  by our IterativeRAG agent.

### Metrics: Now Substantially Covered

**What we have (post this PR):**
- Answer quality: EM, F1, BERTScore, BLEURT, LLM-as-Judge
- Faithfulness: NLI-based groundedness check (now wired with contexts)
- Hallucination: NLI-based contradiction detection (now wired with contexts)
- Retrieval proxy: AnswerInContext (binary) and ContextRecall (graded)

**What's missing:**
- **Answer Relevance** (RAGAS): generates questions from the answer and checks
  similarity to original question. Requires an LLM call per prediction.
- **Context Precision** (RAGAS): LLM rates each retrieved passage for relevance.
  Expensive (LLM call per passage per prediction).
- **MRR / NDCG / Recall@k**: Classical IR metrics requiring ground-truth relevant
  passage IDs. Only feasible with datasets that annotate passages (NQ has this
  partially).
- **ARES**: Automated RAG evaluation using prediction-powered inference. Requires
  fine-tuned classifiers.

**Justification**: Our metric suite covers the core dimensions (correctness,
groundedness, retrieval quality) without requiring expensive LLM calls for every
metric. The missing metrics are either expensive (Answer Relevance, Context
Precision) or require data we don't have (MRR/NDCG ground-truth passages).

---

## Limitations

1. **Single-vector retrieval only** — No multi-vector (ColBERT) or learned sparse
   (SPLADE) representations. This limits retrieval quality ceiling but keeps the
   comparison fair across all agent strategies.

2. **Fixed chunking** — No semantic or adaptive chunking. All experiments use the
   same chunk size, making results comparable but potentially leaving quality on
   the table.

3. **No graph-based retrieval** — GraphRAG and knowledge-graph approaches are
   entirely out of scope. These represent a different paradigm.

4. **Truncated context in metrics** — `retrieved_docs.content` is truncated to
   ~500 chars by `AgentResult.to_dict()`. This means faithfulness/hallucination
   NLI checks operate on truncated passages, which may miss information present
   in the full retrieved text. Acceptable for comparative evaluation (all
   experiments truncated equally) but noted as a limitation.

5. **No passage-level ground truth** — Without annotated relevant passages, we
   cannot compute classical IR metrics (MRR, NDCG, Recall@k). AnswerInContext
   and ContextRecall serve as proxies.

---

## Conclusion

RAGiCamp is **not trying to be a SOTA retrieval system**. It's a benchmarking
framework for comparing RAG strategies (agent types, query transforms, reranking,
hybrid retrieval) under controlled conditions. The techniques we omit (ColBERT,
SPLADE, GraphRAG) would each be interesting additions but would either confound
the comparison or require fundamentally different infrastructure.

The metric suite (post this PR) covers the essential dimensions: answer
correctness, retrieval quality proxy, and generation faithfulness — sufficient
for the thesis evaluation goals.
