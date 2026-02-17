# Future Work: RAGiCamp Roadmap

> **Last updated**: 2026-02-13
>
> **Scope**: Engineering fixes, performance optimizations, test coverage, and research extensions.
> Organized by priority and effort. Items in **Phase 0** are bugs found in the Feb 2026 code review
> that should be fixed before new feature work.

---

## Table of Contents

1. [Phase 0: Known Bugs & Correctness Fixes](#phase-0-known-bugs--correctness-fixes)
2. [Phase 1: Performance & Robustness](#phase-1-performance--robustness)
3. [Phase 2: Test Coverage Gaps](#phase-2-test-coverage-gaps)
4. [Phase 3: Quick-Win Features](#phase-3-quick-win-features)
5. [Phase 4: Query Processing Research](#phase-4-query-processing-research)
6. [Phase 5: Retrieval Research](#phase-5-retrieval-research)
7. [Phase 6: Post-Retrieval Processing](#phase-6-post-retrieval-processing)
8. [Phase 7: Evaluation & Analysis](#phase-7-evaluation--analysis)
9. [Phase 8: Advanced RAG Patterns](#phase-8-advanced-rag-patterns)
10. [Current Capabilities](#current-capabilities)
11. [Implementation Roadmap](#implementation-roadmap)
12. [References](#references)

---

## Phase 0: Known Bugs & Correctness Fixes

Issues found during the Feb 2026 comprehensive code review. These should be addressed before
new feature work to keep the foundation solid.

### 0.1 StopIteration risk in cache/search result interleaving

- **Severity**: High
- **File**: `agents/base.py:515`
- **Bug**: `next(miss_iter)` assumes the search backend returns exactly as many results as there
  are cache misses. If the count diverges (e.g., search backend returns fewer results due to
  insufficient candidates), an unhandled `StopIteration` propagates.
- **Fix**: Add a length assertion after search:
  ```python
  assert len(search_results) == len(miss_indices), (
      f"Search returned {len(search_results)} results for {len(miss_indices)} misses"
  )
  ```
- **Effort**: Small (1 line)

### 0.2 Pickle migration reads only first 4KB

- **Severity**: Medium
- **File**: `cli/commands.py:527`
- **Bug**: `_pickle_needs_migration()` reads only the first 4096 bytes to check for old module
  paths (`ragicamp.retrievers.base`). If the module path string is deeper in the pickle, migration
  is silently skipped and the index fails at load time with a `ModuleNotFoundError`.
- **Fix**: Read larger header (32KB or entire file for small pickles), or use a more reliable
  detection method (try unpickling with `find_class` override).
- **Effort**: Small

### 0.3 Duplicate judge config resolution logic

- **Severity**: Medium (maintenance hazard)
- **Files**: `cli/study.py:create_judge_model()` and `cli/commands.py:_resolve_judge_config()`
- **Bug**: Two independent implementations of the same API-key-detection and provider-selection
  logic. Both currently prefer DeepInfra over OpenAI when both keys are set, but if one is modified
  without the other, judge behavior silently diverges between `ragicamp run` and `ragicamp metrics`.
- **Fix**: Extract into a shared `cli/judge.py:resolve_judge_config()` function. Both call sites
  become one-liners.
- **Effort**: Small

### 0.4 Provider refcount not thread-safe

- **Severity**: Medium (latent — currently single-threaded by design)
- **Files**: `models/providers/generator.py:43,58,88`, `embedder.py`, `reranker.py`
- **Bug**: `self._refcount` is a plain integer incremented/decremented without locking. Safe today
  because agents are single-threaded, but undocumented. If async agents or threaded embedders are
  added later, this causes double-loads or premature unloads.
- **Fix**: Either (a) add `threading.Lock` to protect refcount, or (b) add explicit docstring
  stating single-threaded contract with a runtime assertion.
- **Effort**: Small

### 0.5 State health check crashes on corrupted JSON

- **Severity**: Medium
- **File**: `state/health.py:344`
- **Bug**: `_load_json()` returns `None` for corrupted files, but the caller at line 344 calls
  `q_data.get("questions", [])` without null-checking. Corrupted `questions.json` → `AttributeError`
  on `NoneType` during health check.
- **Fix**: Add `if q_data is None: return ...` guard before access.
- **Effort**: Small

### 0.6 Broad exception handling hides errors

- **Severity**: Low-Medium
- **Files**: `cli/commands.py:751-753`, `factory/agents.py:288-294`
- **Bug**: `except Exception as e: print(f"ERROR: {e}")` discards stack traces. Debug information
  is lost, making production failures harder to diagnose.
- **Fix**: Use `logger.exception()` to capture full traceback, or at minimum `logger.warning(..., exc_info=True)`.
- **Effort**: Small

### 0.7 Retrieval store docstring describes nonexistent in-memory cache

- **Severity**: Low (documentation bug)
- **File**: `cache/retrieval_store.py`
- **Bug**: Docstring describes a `_mem` in-memory dict sitting in front of SQLite, but the code
  doesn't implement it.
- **Fix**: Either implement the in-memory cache (beneficial for repeated queries in same
  experiment) or correct the docstring.
- **Effort**: Small

### 0.8 `np.frombuffer()` without copy in embedding cache

- **Severity**: Low
- **File**: `cache/embedding_store.py:179`
- **Bug**: `np.frombuffer(blob, dtype=np.float32)` creates a view over the SQLite buffer without
  copy. Memory layout assumption (C-contiguous) is not explicitly verified.
- **Fix**: Use `np.frombuffer(blob, dtype=np.float32).copy()` for safety, or document the
  C-order assumption with `tobytes(order='C')` on write (line 210).
- **Effort**: Small

### 0.9 Metrics re-computation special case is implicit

- **Severity**: Low
- **File**: `experiment.py:254-255`
- **Bug**: `COMPUTING_METRICS` phase is always re-run via a hardcoded `!= ExperimentPhase.COMPUTING_METRICS`
  check. This implicit behavior would be missed if new phases are added.
- **Fix**: Make this explicit in `ExperimentState` (e.g., `phase.always_rerun` property) rather
  than phase-name matching.
- **Effort**: Small

---

## Phase 1: Performance & Robustness

Optimizations and hardening for production workloads.

### 1.1 Hardcoded candidate multipliers

- **Files**: `retrievers/hybrid.py:68,96` (`top_k * 3`), `indexes/hierarchical.py:243,320` (`top_k * 5`)
- **Problem**: Candidate expansion factors are hardcoded. For some query distributions, 3x or 5x
  may be insufficient (too few unique parents in hierarchical) or wasteful (pure dense with
  well-clustered results).
- **Fix**: Add `candidate_expansion_factor` parameter to `HybridSearcher` and `HierarchicalIndex`,
  with current values as defaults.
- **Effort**: Small

### 1.2 Reranker held on GPU during generation phase

- **File**: `agents/fixed_rag.py:150-151`
- **Problem**: Reranker provider is loaded via `ExitStack` for the entire retrieval+generation
  lifecycle, but it's only used during retrieval. On a single-GPU setup, the reranker occupies
  memory while the generator also tries to load.
- **Fix**: Release reranker after retrieval completes (before generation starts). Either restructure
  the ExitStack scope or explicitly unload.
- **Effort**: Medium

### 1.3 No explicit GPU cleanup between agent phases

- **File**: `agents/fixed_rag.py:157-164`
- **Problem**: Between retrieval (embedder loaded) and generation (generator loaded), there's no
  explicit `ResourceManager.clear_gpu_memory()`. Relies on provider unload to clean up, but
  fragmented GPU memory can prevent the generator from allocating contiguous blocks.
- **Fix**: Add `ResourceManager.clear_gpu_memory()` between phases in all RAG agents.
- **Effort**: Small

### 1.4 ResilientExecutor abort too aggressive

- **File**: `execution/executor.py:184-202`
- **Problem**: 5 consecutive batch failures aborts the entire experiment. No retry/backoff for
  transient errors (GPU OOM that recovers after cleanup, network timeouts, rate limits).
- **Fix**: Add exponential backoff between retries. Consider clearing GPU memory and retrying
  with smaller batch size before counting as consecutive failure.
- **Effort**: Medium

### 1.5 GPU memory cleanup interval hardcoded

- **File**: `execution/executor.py:221-225`
- **Problem**: `gc.collect()` + `torch.cuda.empty_cache()` every 10 batches. Suboptimal for both
  small GPUs (too infrequent) and large GPUs with small models (unnecessary overhead).
- **Fix**: Make interval configurable or adaptive based on memory pressure.
- **Effort**: Small

### 1.6 VLLMEmbedder auto-detect can be aggressive

- **File**: `models/vllm_embedder.py:82-101`
- **Problem**: `GPUProfile.detect()` sets very high batch parameters (262144 tokens for 160GB
  GPU). If other processes share the GPU, this causes OOM.
- **Fix**: Use actual free memory (`torch.cuda.mem_get_info()`) rather than total GPU memory
  for profile detection.
- **Effort**: Small

### 1.7 Tee thread exception handling missing

- **File**: `execution/runner.py:377-381`
- **Problem**: The `_tee()` function has no try/except. If log file becomes unavailable or stdout
  is closed, the thread dies silently. Experiment monitoring continues but logs are incomplete.
- **Fix**: Wrap the loop body in try/except with a warning on failure.
- **Effort**: Small

### 1.8 Index migration without backup

- **File**: `cli/commands.py:606-628`
- **Problem**: `_migrate_single_index()` modifies pickles in-place. If migration fails partway
  through, the original pickle is corrupted.
- **Fix**: Write to temp file first, then rename (same atomic pattern used for JSON writes).
- **Effort**: Small

---

## Phase 2: Test Coverage Gaps

Areas with insufficient test coverage that increase regression risk.

### 2.1 DirectLLMAgent functional tests

- **Current**: Only factory wiring tests verify the agent is created, but no tests exercise
  the `run()` method with mock providers.
- **Needed**: Test batch generation, error handling, checkpoint resume.
- **Effort**: Small

### 2.2 FixedRAGAgent functional tests

- **Current**: Minimal tests compared to IterativeRAG (11 tests) and SelfRAG (9 tests).
- **Needed**: Test retrieval → reranking → generation pipeline with mocks. Test reranker-only
  and no-reranker paths. Test query transform integration.
- **Effort**: Medium

### 2.3 Faithfulness and Hallucination metrics edge cases

- **Current**: Basic tests exist but edge cases not covered.
- **Needed**: Test NLI pipeline fallback behavior, token-based method with empty inputs,
  LLM method with various response formats (yes/no/maybe/garbage), GPU cleanup on failure.
- **Effort**: Small-Medium

### 2.4 Retrieval cache consistency tests

- **Current**: 1 test for retrieval cache.
- **Needed**: Test cache key correctness (same query + different top_k → different entries),
  partial hit behavior, concurrent read/write safety.
- **Effort**: Small

### 2.5 Spec builder edge cases

- **Current**: Good coverage of happy paths.
- **Needed**: Test missing retriever configs (should error early, not KeyError at line 364),
  stratified sampling with exhausted strata, singleton multi-dataset expansion warnings.
- **Effort**: Medium

### 2.6 Performance regression benchmarks

- **Current**: None.
- **Needed**: Benchmark tests that track embedding throughput, search latency, and end-to-end
  experiment time. Flag regressions in CI.
- **Effort**: Medium-Large

---

## Phase 3: Quick-Win Features

High-impact, low-effort improvements (1-2 weeks total).

### 3.1 Retrieval Quality Metrics (P0 priority)

**Problem**: We only evaluate final answer quality, not retrieval quality. When experiments
fail, we can't tell if the issue is bad retrieval or bad generation.

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Recall@k** | % of relevant docs retrieved | `\|retrieved ∩ relevant\| / \|relevant\|` |
| **Precision@k** | % of retrieved docs relevant | `\|retrieved ∩ relevant\| / k` |
| **MRR** | Rank of first relevant result | `1 / rank_of_first_relevant` |
| **NDCG@k** | Graded relevance ranking quality | Normalized DCG |

**Why It Matters**:
```
Scenario A: Retrieval gets 5/5 relevant docs, LLM fails to extract answer
Scenario B: Retrieval gets 0/5 relevant docs, LLM can't answer

Both show F1=0, but the fix is completely different.
```

**Implementation**:
```python
# In metrics/retrieval.py
class RetrievalMetrics(Metric):
    def compute(self, retrieved_ids: list, relevant_ids: list, k: int) -> dict:
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        hits = len(retrieved_set & relevant_set)

        return {
            "recall@k": hits / len(relevant_set) if relevant_set else 0,
            "precision@k": hits / k,
            "mrr": self._compute_mrr(retrieved_ids, relevant_set),
        }
```

**Challenge**: Requires ground-truth relevant passages. Options:
- Use datasets with annotated passages (NQ has this)
- Use LLM-as-judge for passage relevance
- Proxy: check if gold answer appears in retrieved text

**Effort**: Low | **Impact**: High

### 3.2 Lost-in-the-Middle Mitigation

**Problem**: LLMs attend poorly to information in the middle of long contexts. Studies show
U-shaped attention: strong at start/end, weak in middle.

**Evidence**: "Lost in the Middle" (Liu et al., 2023) showed 20-30% performance drop when
relevant info is in middle positions.

**Solution**: Reorder retrieved passages — most relevant at start and end, least in middle.

```python
def reorder_passages_for_attention(passages: list[Document]) -> list[Document]:
    """Reorder passages: most relevant at start and end, least relevant in middle.

    Given passages ranked [1, 2, 3, 4, 5] by relevance:
    Returns: [1, 3, 5, 4, 2] — alternating from edges inward.
    """
    if len(passages) <= 2:
        return passages

    n = len(passages)
    reordered = [None] * n

    left, right = 0, n - 1
    for i, passage in enumerate(passages):
        if i % 2 == 0:
            reordered[left] = passage
            left += 1
        else:
            reordered[right] = passage
            right -= 1

    return reordered
```

**When to Use**: When `top_k > 3` (middle problem doesn't exist with few passages).

**Effort**: Low | **Impact**: Medium

### 3.3 Context Recall Metric

Check if gold answer text is present in retrieved context (no LLM needed):

```python
class ContextRecall(Metric):
    def compute_single(self, context: str, expected: list[str]) -> float:
        context_lower = context.lower()
        for gold in expected:
            if gold.lower() in context_lower:
                return 1.0
        return 0.0
```

**Effort**: Low | **Impact**: High (immediate diagnostic value)

---

## Phase 4: Query Processing Research

Techniques that improve how queries are formulated before retrieval (2-4 weeks).

### 4.1 Query Decomposition (for Multi-Hop QA)

**Problem**: Complex questions require multiple retrieval steps. Neither HyDE nor Iterative RAG
handles true multi-hop reasoning.

| Method | What It Does | Handles Multi-Hop? |
|--------|-------------|-------------------|
| **HyDE** | Generates hypothetical answer, embeds that | No — reformats single query |
| **MultiQuery** | Generates paraphrases of same question | No — variations of same question |
| **Iterative RAG** | Refines query if retrieval insufficient | Partial — refines, doesn't decompose |
| **Query Decomposition** | Breaks into sub-questions, answers each | Yes — designed for this |

**Example**:
```
Original: "Who is the spouse of the director of Inception?"

HyDE:     "The spouse of the director of Inception is Emma Thomas..."
          (Still one query, just reformatted)

Decomposition:
  Sub-Q1: "Who directed Inception?" → "Christopher Nolan"
  Sub-Q2: "Who is Christopher Nolan's spouse?" → "Emma Thomas"
  Final:  "Emma Thomas"
```

**Implementation**:
```python
class QueryDecomposer(QueryTransformer):
    DECOMPOSITION_PROMPT = """Break this question into simpler sub-questions.
Only decompose if the question requires multiple pieces of information.

Question: {question}

If decomposition needed, output each sub-question on a new line prefixed with "SUB: ".
If no decomposition needed, output "ATOMIC: {question}" """

    def transform(self, query: str) -> list[str]:
        response = self.model.generate(
            self.DECOMPOSITION_PROMPT.format(question=query)
        )
        if response.startswith("ATOMIC:"):
            return [query]

        sub_questions = []
        for line in response.split("\n"):
            if line.startswith("SUB:"):
                sub_questions.append(line[4:].strip())
        return sub_questions if sub_questions else [query]
```

**Evaluation**: HotpotQA is specifically designed for multi-hop QA.

**Effort**: Medium | **Impact**: High (HotpotQA)

### 4.2 Query Expansion (Classic IR Technique)

**Problem**: Vocabulary mismatch between query and documents.

```
Query:    "heart attack symptoms"
Document: "Myocardial infarction presents with chest pain..."

Without expansion: low similarity (different words)
With expansion:    "heart attack myocardial infarction symptoms signs" → better match
```

**Option A: LLM-based expansion**
```python
class LLMQueryExpander(QueryTransformer):
    PROMPT = """Add synonyms and related terms to this query for better search.
Keep it under 50 words. Output just the expanded query.

Query: {query}
Expanded:"""

    def transform(self, query: str) -> str:
        return self.model.generate(self.PROMPT.format(query=query))
```

**Option B: Embedding-based expansion** (no LLM cost)
```python
def expand_with_similar_terms(query: str, embedder, top_k: int = 3) -> str:
    words = query.split()
    expansions = []
    for word in words:
        similar = embedder.most_similar(word, topn=top_k)
        expansions.extend([w for w, score in similar if score > 0.7])
    return query + " " + " ".join(set(expansions))
```

**When to Use**: Especially useful for technical domains (medical, legal).

**Effort**: Low | **Impact**: Medium

---

## Phase 5: Retrieval Research

New retrieval backends and index types (4-8 weeks).

### 5.1 SPLADE (Learned Sparse Retrieval)

**Problem**: BM25/TF-IDF use hand-crafted term weights. They can't learn domain-specific
importance.

**Solution**: SPLADE learns sparse representations that are:
- More semantic than BM25 (learns term importance)
- More interpretable than dense (you can see which terms matched)
- Still fast (sparse operations)

```
Query: "What causes global warming?"

BM25 weights:   global=1.2, warming=1.5, causes=0.8, what=0.1
SPLADE weights: global=0.8, warming=1.2, climate=0.9, greenhouse=0.7, CO2=0.5
                (SPLADE adds semantically related terms!)
```

**Implementation**:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

class SPLADEIndex(Index):
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def encode(self, text: str) -> scipy.sparse.csr_matrix:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**tokens).logits
        weights = torch.log1p(torch.relu(logits)).max(dim=1).values
        return self._to_sparse(weights)
```

**Expected Improvement**: 5-15% better than BM25 on semantic queries, similar speed.

**Effort**: Medium | **Impact**: Medium-High

### 5.2 ColBERT (Late Interaction)

**Problem**: Single-vector embeddings lose fine-grained token information.

**Solution**: ColBERT keeps token-level embeddings and uses MaxSim scoring.

```
Standard Dense:
  Query → [single 768-dim vector]
  Doc   → [single 768-dim vector]
  Score = cosine(query_vec, doc_vec)

ColBERT:
  Query "what is RAG" → [[vec_what], [vec_is], [vec_RAG]]   # 3 vectors
  Doc "RAG means..."  → [[vec_RAG], [vec_means], ...]       # N vectors
  Score = Σ max_j(cosine(q_i, d_j)) for each query token i
```

**Trade-offs**:
- Index size: ~5x larger than dense
- Indexing time: ~3x slower
- Search time: Similar (with PLAID optimization)
- Quality: Significant improvement, especially for long docs

**Effort**: High | **Impact**: High

### 5.3 Semantic Chunking

**Problem**: Fixed-size chunks split in arbitrary places, breaking semantic coherence.

**Bad chunking** (fixed 512 chars):
```
Chunk 1: "...World War II began in 1939 when Germany invaded Poland. The"
Chunk 2: "war lasted until 1945 and resulted in millions of casualties..."
```

**Semantic chunking** splits where topics change:

```python
class SemanticChunker:
    def __init__(self, embedder, similarity_threshold: float = 0.5):
        self.embedder = embedder
        self.threshold = similarity_threshold

    def chunk(self, text: str, min_size: int = 100, max_size: int = 1000) -> list[str]:
        sentences = self._split_sentences(text)
        embeddings = self.embedder.encode(sentences)
        chunks, current_chunk = [], [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = cosine_similarity(embeddings[i-1], embeddings[i])
            current_size = sum(len(s) for s in current_chunk)

            if similarity < self.threshold and current_size >= min_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            elif current_size >= max_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
```

**Effort**: Medium | **Impact**: Medium

---

## Phase 6: Post-Retrieval Processing

Techniques applied between retrieval and generation.

### 6.1 Context Compression

**Problem**: Sending 5 full passages to LLM is expensive, slow, and sometimes worse (irrelevant
info distracts).

**Option A: Extractive compression** (fast, no LLM):
```python
def extract_relevant_sentences(passage: str, query: str, max_sentences: int = 3) -> str:
    sentences = passage.split(". ")
    query_embedding = embedder.encode(query)
    scored = []
    for sent in sentences:
        sent_embedding = embedder.encode(sent)
        score = cosine_similarity(query_embedding, sent_embedding)
        scored.append((score, sent))
    top_sentences = sorted(scored, reverse=True)[:max_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: passage.find(x[1]))
    return ". ".join(s for _, s in top_sentences)
```

**Option B: LLMLingua** (token-level compression):
```python
from llmlingua import PromptCompressor
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)
compressed = compressor.compress_prompt(context=passages, question=query, rate=0.5)
```

**Expected Impact**: 40-60% token reduction, 0-5% quality loss (sometimes improvement).

**Effort**: Medium | **Impact**: Medium

### 6.2 Passage Attribution

**Problem**: We can't tell which passage contributed to the answer. Important for debugging,
explainability, and hallucination detection.

```python
ATTRIBUTION_PROMPT = """Based on the passages below, answer the question.
After your answer, cite which passage(s) you used with [1], [2], etc.

{passages}

Question: {question}

Answer (with citations):"""

def extract_citations(response: str) -> tuple[str, list[int]]:
    import re
    citations = [int(c) for c in re.findall(r'\[(\d+)\]', response)]
    answer = re.sub(r'\[\d+\]', '', response).strip()
    return answer, citations
```

**Effort**: Low | **Impact**: Medium

---

## Phase 7: Evaluation & Analysis

Improvements to how we measure and compare experiments.

### 7.1 Statistical Significance Testing

- **Problem**: Current comparison tools show mean/min/max/std but no confidence intervals or
  p-values. We can't tell if a 2% improvement is real or noise.
- **Status**: **Partially addressed** — `notebooks/nb10_statistical_tests.ipynb` provides paired
  bootstrap tests, Holm-Bonferroni correction, and LaTeX tables for thesis-ready results.
  Still needed: integrate into `analysis/comparison.py` for CLI-level access.
- **Effort**: Medium | **Impact**: High

### 7.2 RAGAS-Style Pipeline Evaluation

Evaluate the full RAG pipeline, not just final answers:

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Faithfulness** | Answer grounded in context? | LLM checks each claim |
| **Answer Relevance** | Answer addresses question? | Generate questions from answer, compare |
| **Context Precision** | Retrieved passages relevant? | LLM rates each passage |
| **Context Recall** | Gold answer in context? | Check if gold appears in passages |

```python
class FaithfulnessMetric(Metric):
    PROMPT = """Given the context and answer, determine if the answer is fully
supported by the context.

Context: {context}
Answer: {answer}

Is every claim in the answer supported by the context?
Reply with just "yes" or "no"."""

    def compute_single(self, answer: str, context: str) -> float:
        response = self.judge_model.generate(
            self.PROMPT.format(context=context, answer=answer)
        )
        return 1.0 if "yes" in response.lower() else 0.0
```

**Effort**: Medium | **Impact**: High

### 7.3 Config Validation Improvements

Several config validation gaps identified in the review:

| Gap | Location | Fix |
|-----|----------|-----|
| `load_in_8bit` and `load_in_4bit` not mutually exclusive | `config/schemas.py:ModelConfig` | Add `@model_validator` |
| `chunk_size > chunk_overlap` not enforced | `config/schemas.py:ChunkingConfig` | Add field validator |
| Retriever configs not validated early | `spec/builder.py:364` | Validate in `build_specs()` |
| Grid explosion not warned | `spec/builder.py:_build_rag_specs()` | Warn when >1000 specs |
| Metric names not validated | `spec/experiment.py:59` | Check against known metrics |

**Effort**: Small | **Impact**: Medium (DX improvement)

---

## Phase 8: Advanced RAG Patterns

Research extensions requiring significant design work (8+ weeks).

### 8.1 Corrective RAG (CRAG)

**Pattern**: Detect poor retrieval and recover.

1. Retrieve documents
2. Evaluate retrieval quality (are docs relevant?)
3. If low quality: fall back to web search or decompose query
4. Generate answer from best sources

```python
class CorrectiveRAGAgent(Agent):
    def answer(self, question: str) -> RAGResponse:
        docs = self.retriever.retrieve(question)
        quality = self._evaluate_quality(question, docs)

        if quality < self.quality_threshold:
            web_docs = self.web_search.search(question)
            docs = self._merge_sources(docs, web_docs)

        return self._generate(question, docs)
```

**Note**: Similar to Self-RAG but with explicit quality evaluation and web fallback.

**Effort**: High | **Impact**: Medium

### 8.2 Proposition-Based Indexing

**Problem**: Chunks contain multiple facts, making retrieval imprecise.

**Solution**: Index atomic propositions (single facts).

```
Original chunk:
"Barack Obama was born in Hawaii in 1961. He served as the 44th President
of the United States from 2009 to 2017."

Propositions:
1. "Barack Obama was born in Hawaii."
2. "Barack Obama was born in 1961."
3. "Barack Obama served as the 44th President of the United States."
4. "Barack Obama served as President from 2009 to 2017."
```

**Trade-offs**:
- Much more precise retrieval
- 5-10x more items to index
- Requires linking propositions back to source documents

**Effort**: High | **Impact**: Medium

---

## Current Capabilities

What's already implemented and production-ready.

| Component | Implementation | Status |
|-----------|---------------|--------|
| **Dense Retrieval** | BGE-large, BGE-M3 embeddings | Production |
| **Sparse Retrieval** | TF-IDF, BM25 | Production |
| **Hybrid Retrieval** | RRF fusion (dense + sparse) | Production |
| **Hierarchical Retrieval** | Child-chunk → parent-doc | Production |
| **Query Transform** | HyDE, MultiQuery | Production |
| **Reranking** | Cross-encoder (BGE-reranker) | Production |
| **IterativeRAG** | Multi-iteration query refinement | Production |
| **SelfRAG** | Adaptive retrieval decision | Production |
| **Index Types** | Flat, IVF, HNSW | Production |
| **Answer Metrics** | F1, EM, BERTScore, BLEURT, LLM-judge | Production |
| **Embedding Cache** | SQLite WAL, shared across experiments | Production |
| **Retrieval Cache** | SQLite WAL, query-hash keyed | Production |
| **Provider Ref-Counting** | Nested load/unload with refcount | Production |
| **Subprocess Isolation** | Per-experiment process for CUDA safety | Production |
| **Atomic IO** | Temp-then-rename for all JSON writes | Production |

---

## Implementation Roadmap

### Sprint 1: Foundation Fixes (3-5 days)

| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| Fix StopIteration bug (0.1) | P0 | Small | High |
| Fix pickle migration (0.2) | P0 | Small | Medium |
| Consolidate judge config (0.3) | P0 | Small | Medium |
| Fix health check null crash (0.5) | P0 | Small | Medium |
| Add length assertion in cache interleaving | P0 | Small | High |
| Fix exception handling (0.6) | P0 | Small | Low |
| Fix retrieval store docstring (0.7) | P0 | Small | Low |

### Sprint 2: Retrieval Metrics + Diagnostics (1-2 weeks)

| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| Retrieval quality metrics (3.1) | P3 | Low | High |
| Context Recall metric (3.3) | P3 | Low | High |
| Lost-in-the-middle reordering (3.2) | P3 | Low | Medium |
| Statistical significance tests (7.1) | P7 | Medium | High |
| Config validation improvements (7.3) | P7 | Small | Medium |

### Sprint 3: Query Processing (2-4 weeks)

| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| Query Decomposition agent (4.1) | P4 | Medium | High |
| Query Expansion transformer (4.2) | P4 | Low | Medium |
| RAGAS Faithfulness metric (7.2) | P7 | Medium | High |

### Sprint 4: Advanced Retrieval (4-8 weeks)

| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| SPLADE integration (5.1) | P5 | Medium | Medium-High |
| Semantic chunking (5.3) | P5 | Medium | Medium |
| Context compression (6.1) | P6 | Medium | Medium |

### Sprint 5: Research Extensions (8+ weeks)

| Task | Phase | Effort | Impact |
|------|-------|--------|--------|
| ColBERT integration (5.2) | P5 | High | High |
| Corrective RAG (8.1) | P8 | High | Medium |
| Proposition indexing (8.2) | P8 | High | Medium |

---

## References

1. **Lost in the Middle**: Liu et al., 2023 — "Lost in the Middle: How Language Models Use Long Contexts"
2. **SPLADE**: Formal et al., 2021 — "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
3. **ColBERT**: Khattab & Zaharia, 2020 — "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
4. **HyDE**: Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
5. **Self-RAG**: Asai et al., 2023 — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
6. **CRAG**: Yan et al., 2024 — "Corrective Retrieval Augmented Generation"
7. **RAGAS**: Es et al., 2023 — "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
8. **LLMLingua**: Jiang et al., 2023 — "LLMLingua: Compressing Prompts for Accelerated Inference"

---

## Method Comparison Matrix

| Method | Query Processing | Retrieval | Post-Processing | Multi-Hop |
|--------|-----------------|-----------|-----------------|-----------|
| **HyDE** | Transform | - | - | No |
| **MultiQuery** | Paraphrase | - | - | No |
| **Iterative RAG** | Refine | - | - | Partial |
| **Query Decomposition** | Decompose | - | - | Yes |
| **Reranking** | - | Two-stage | - | - |
| **SPLADE** | - | Learned sparse | - | - |
| **ColBERT** | - | Late interaction | - | - |
| **Hybrid (RRF)** | - | Fusion | - | - |
| **Context Compression** | - | - | Compress | - |
| **Passage Reordering** | - | - | Reorder | - |
| **Self-RAG** | - | Adaptive | Critique | No |
| **CRAG** | Fallback | Quality check | - | No |

This matrix helps identify which techniques are complementary and can be combined.
