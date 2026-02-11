# RAGiCamp Engineering Backlog

> **Purpose**: Track all code quality, reliability, and maintainability issues identified
> in the 2026-02-10 comprehensive review. Living document — update status as issues are resolved.
>
> **Created**: 2026-02-10
> **Last updated**: 2026-02-11
> **Review scope**: Full codebase — abstractions, tests, execution pipeline, models/metrics/RAG

## Status Key

- `[ ]` Not started
- `[~]` In progress
- `[x]` Fixed
- `[-]` Won't fix (with reason)

## Priority Key

- **P0** — Data corruption, silent wrong results, or crashes in production paths
- **P1** — Reliability/resource issues that degrade robustness under failure
- **P2** — Missing test coverage that blocks safe refactoring
- **P3** — Interface/design issues that hurt maintainability and extensibility
- **P4** — Minor polish, consistency, documentation

---

## Phase 1: Data Integrity & Correctness (P0)

Issues that can cause silent data corruption, wrong experiment results, or crashes.

### 1.1 Division-by-zero in embedding normalization during index builds

- **Status:** `[x]` Fixed 2026-02-10
- **Files:** `indexes/builders/embedding_builder.py:255`, `indexes/builders/hierarchical_builder.py:262`, `indexes/hierarchical.py:219`
- **Bug:** `np.divide(embeddings, norms)` without guarding against zero-norm vectors. Empty or whitespace-only chunks that slip through chunking produce zero-norm embeddings → `nan`/`inf` values → corrupted FAISS index. Queries against a corrupted index return garbage results silently.
- **Note:** `models/vllm_embedder.py:251` already does this correctly with `np.maximum(norms, 1e-12)`.
- **Fix:** Added `norms = np.maximum(norms, 1e-12)` before dividing in all three locations.
- **Impact:** Prevents corrupted indexes from poisoning entire experiment runs. Low risk fix — one-line guard in each location.
- **Effort:** Small (3 one-line changes)

### 1.2 CrossEncoderReranker.batch_rerank mutates caller's Document objects

- **Status:** `[x]` Fixed 2026-02-10
- **File:** `rag/rerankers/cross_encoder.py:179-180` (batch_rerank), `:131-132` (rerank)
- **Bug:** Directly overwrites `doc.score` on the caller's Document objects. The provider-level `RerankerWrapper` was already fixed in the 2026-02-09 audit to use `copy.copy()`, but this lower-level class was missed. Any code path that calls `CrossEncoderReranker` directly (or retains references to docs after reranking) sees silently corrupted scores.
- **Fix:** Use `copy.copy()` on documents before assigning scores, matching the `RerankerWrapper` pattern at `models/providers/reranker.py:121-122`.
- **Impact:** Prevents score corruption propagating through retrieval pipelines. Minimal perf cost.
- **Effort:** Small

### 1.3 Non-atomic JSON writes in multiple locations

- **Status:** `[x]` Fixed 2026-02-10
- **Files:**
  - `experiment.py:475` — `_save_partial_result` writes `results.json` directly
  - `experiment.py:117` — `ExperimentResult.save` writes directly
  - `utils/paths.py:65` — `safe_write_json` is named "safe" but is NOT atomic
  - `cli/study.py:532` — `study_meta.json` write
  - `utils/artifacts.py:54` — `ArtifactManager.save_json`
  - `cli/commands.py:349` — `cmd_metrics` results write
- **Bug:** All use `open(path, "w")` + `json.dump()` without temp-then-rename. A crash mid-write produces truncated/corrupted JSON. The experiment then appears to have results but they are unreadable. The codebase already has correct atomic write implementations in `ExperimentIO._atomic_write`, `init_phase._atomic_write`, and `metrics_phase._save_predictions` — but they are separate copies.
- **Fix:**
  1. Create a single `atomic_write_json(data, path)` utility function (consolidating the 3+ existing implementations).
  2. Replace all direct JSON writes with this utility.
  3. Rename `safe_write_json` to `write_json` or make it use the atomic pattern.
- **Impact:** Prevents data loss on crash across all experiment phases. High confidence fix.
- **Effort:** Medium (create utility + update ~8 call sites)

### 1.4 BERTScoreMetric OOM retry discards all partial results

- **Status:** `[x]` Fixed 2026-02-10
- **File:** `metrics/bertscore.py:98-153`
- **Bug:** When OOM occurs mid-computation, code halves `batch_size` and restarts from scratch (`all_P, all_R, all_F1 = [], [], []` inside the `while` loop). All successfully computed batches before OOM are thrown away and recomputed. For large evaluation sets, this wastes significant GPU time.
- **Fix:** Move result accumulation lists (`all_P`, `all_R`, `all_F1`) outside the `while` loop. Only reprocess the failed batch with smaller size.
- **Impact:** Prevents wasted GPU computation on OOM recovery. Significant for large evaluations.
- **Effort:** Small (move 3 lines outside the loop + adjust batch tracking)

### 1.5 HuggingFaceModel.get_embeddings sends tensors to wrong device for quantized models

- **Status:** `[x]` Fixed 2026-02-10
- **File:** `models/huggingface.py:179-180`
- **Bug:** Uses `self.device` (user-specified string) instead of the model's actual device for `device_map="auto"`. The `generate` method (lines 131-135) correctly resolves the device, but `get_embeddings` doesn't. Results in `RuntimeError: Expected all tensors to be on the same device`.
- **Fix:** Use the same device detection as `generate`: `self.model.device` or `next(self.model.parameters()).device`.
- **Impact:** Fixes crash when using HF backend for embeddings with quantized models.
- **Effort:** Small

### 1.6 OpenAI error strings returned as predictions

- **Status:** `[x]` Fixed 2026-02-10
- **File:** `models/openai.py:172`
- **Bug:** When an OpenAI API call fails, the error message `"[ERROR: ExceptionType: msg]"` is returned as a "prediction" string. Downstream metrics compare this against reference answers, producing near-zero scores that look like genuinely bad answers rather than failures. This was partially addressed in the 2026-02-09 audit (markers added), but the metrics pipeline still scores them as predictions.
- **Fix:** Either (a) raise the exception so the executor handles retries, or (b) add a sentinel check in `compute_metrics_batched` to exclude `[ERROR:` prefixed predictions from scoring.
- **Impact:** Prevents misleading experiment scores when API errors occur.
- **Effort:** Small–Medium

### 1.7 OpenAI model hardcodes embedding model name

- **Status:** `[x]` Fixed 2026-02-10
- **File:** `models/openai.py:186`, `:311` (async version)
- **Bug:** `model="text-embedding-ada-002"` is hardcoded, ignoring `self.model_name`. Users configuring a different embedding model get `ada-002` silently.
- **Fix:** Use `self.model_name` or add a separate `embedding_model` config field.
- **Impact:** Prevents silent use of wrong embedding model.
- **Effort:** Small

### 1.8 `_stratified_sample` uses global random instead of seeded RNG

- **Status:** `[x]` Fixed 2026-02-10
- **File:** `spec/builder.py:224`
- **Bug:** Parent function `_apply_sampling` creates `rng = random.Random(seed)` but `_stratified_sample` uses `random.choice` (global random). Stratified sampling is not reproducible even when seed is set.
- **Fix:** Pass `rng` to `_stratified_sample`, use `rng.choice()` / `rng.sample()`.
- **Impact:** Ensures reproducible experiment selection with seeds. Essential for scientific reproducibility.
- **Effort:** Small

---

## Phase 2: Reliability & Resource Management (P1)

Issues that degrade robustness under failure or waste resources.

### 2.1 SQLite connections never closed — resource leak

- **Status:** `[x]` Fixed 2026-02-11
- **Files:** `cache/embedding_store.py`, `cache/retrieval_store.py`
- **Bug:** Connections opened lazily via `@property` but no `close()` is ever called. No `__enter__`/`__exit__`. WAL mode holds shared locks until process exit.
- **Fix:** Added `__enter__`/`__exit__` context manager methods and `__del__` safety net to both `EmbeddingStore` and `RetrievalStore`. Existing `close()` methods were already present but now properly wired.
- **Impact:** Prevents connection leaks and potential WAL lock issues in long studies.
- **Effort:** Small

### 2.2 CachedEmbedder._load_real_model unsafe context manager entry

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `cache/cached_embedder.py:201-205`
- **Bug:** `self._real_ctx = self._provider.load(...)` followed by `self._real_ctx.__enter__()`. If `__enter__()` raises (GPU OOM), `self._real_ctx` is set but `__exit__` can't clean up properly. The `cleanup()` method checks `self._real_ctx is not None` and tries to `__exit__` an incompletely entered context.
- **Fix:** Reordered to only store `self._real_ctx` after successful `__enter__()`.
- **Impact:** Prevents GPU resource leak on OOM during embedder initialization.
- **Effort:** Small

### 2.3 FaithfulnessMetric and HallucinationMetric NLI pipelines never unloaded

- **Status:** `[x]` Fixed 2026-02-11
- **Files:** `metrics/faithfulness.py:45-62`, `metrics/hallucination.py:38-54`
- **Bug:** Both lazy-load an NLI pipeline onto GPU but never unload it. Unlike BERTScore/BLEURT which unload in `finally` blocks, these keep the model in GPU memory indefinitely after `compute()` returns.
- **Fix:** Added `_unload_pipeline()` methods and `try/finally` in `compute()`, following the BERTScore pattern.
- **Impact:** Prevents GPU memory waste after metrics computation.
- **Effort:** Small

### 2.4 HyDE/MultiQuery transformers load/unload model on every call

- **Status:** `[x]` Fixed 2026-02-11 (provider ref-counting + agent session wrapping)
- **Files:** `rag/query_transform/hyde.py:86,119`, `rag/query_transform/multiquery.py:82,155`
- **Bug:** Each call to `transform()` or `batch_transform()` does `with self.generator_provider.load() as generator:`, loading and unloading the model every time. In IterativeRAG, the transformer is called per iteration — the model is loaded/unloaded multiple times per question. Extremely expensive on GPU.
- **Fix:** Added ref-counting to all 3 providers (`GeneratorProvider`, `EmbedderProvider`, `RerankerProvider`). Nested `load()` calls now reuse the already-loaded model. Agent `run()` methods open outer provider sessions so all inner calls (HyDE, reranking, generation) hit the refcount path. Additionally, HyDE is now only applied on iteration 0 in IterativeRAG, and retrieval cache is enabled for transformed queries.
- **Impact:** Saves 2-8 minutes per Optuna trial. See `docs/AGENT_PERFORMANCE_ANALYSIS.md` for full analysis.
- **Effort:** Medium (completed)

### 2.4a Provider ref-counting for model reuse

- **Status:** `[x]` Fixed 2026-02-11
- **Files:** `models/providers/generator.py`, `models/providers/embedder.py`, `models/providers/reranker.py`
- **Fix:** Added `_refcount: int` to all providers. First `load()` actually loads; nested calls increment refcount and yield existing model. Unload only at refcount 0.
- **Impact:** Eliminates redundant model loads across all agent types.
- **Effort:** Small

### 2.4b HyDE iteration guard in IterativeRAG

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `agents/iterative_rag.py:273`
- **Fix:** Query transform only applied on iteration 0. Refined queries are already document-like, making HyDE on them wasteful.
- **Impact:** Saves 1 generator load per additional iteration.
- **Effort:** Small

### 2.4c Retrieval cache enabled for transformed queries

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `factory/agents.py:212-215`
- **Fix:** Removed conditional that disabled retrieval cache when `query_transform != "none"`. Cache keys on query text, so transformed queries get unique entries naturally.
- **Impact:** Cache hits on repeat Optuna trials with same retriever+transform+top_k.
- **Effort:** Small

### 2.4d HybridSearcher sequential RRF fusion per query

- **Status:** `[ ]`
- **File:** `retrievers/hybrid.py:143-183`
- **Bug:** After batch dense and batch sparse search complete, RRF fusion loops through each query sequentially in Python: document-to-SearchResult conversion, score computation, dict sorting. For 100 queries x 30 candidates = 3,000 object creations + sorts in Python loops.
- **Fix:** Vectorize RRF score computation using numpy arrays across all queries at once; use numpy argsort for batched sorting.
- **Impact:** Seconds-level improvement for large query batches with hybrid search.
- **Effort:** Medium

### 2.4e BM25 sequential search fallback

- **Status:** `[ ]`
- **File:** `indexes/sparse.py:181-195`
- **Bug:** `batch_search` for BM25 falls back to `[self._search_bm25(q, top_k) for q in queries]` — 100% sequential. TF-IDF path uses sklearn's vectorized `cosine_similarity`, but BM25 has no equivalent.
- **Fix:** Implement manual batched BM25 scoring with numpy, or accept as a library limitation.
- **Impact:** Seconds-level for large batches. Low priority since TF-IDF path is already vectorized.
- **Effort:** High (requires reimplementing BM25 scoring)

### 2.4f Cache stores use sequential `execute()` instead of `executemany()`

- **Status:** `[x]` Fixed 2026-02-11
- **Files:** `cache/embedding_store.py`, `cache/retrieval_store.py`
- **Bug:** Batch insertion loops with individual `cursor.execute()` calls inside a transaction. SQLite's `executemany()` is significantly faster for bulk inserts.
- **Fix:** Pre-build parameter tuples, use `cursor.executemany()`. Total `rowcount` from executemany replaces per-row tracking.
- **Impact:** Minor — only affects cache-miss writes, and SQLite is not the bottleneck.
- **Effort:** Small

### 2.5 ResourceManager.clear_gpu_memory() called every batch iteration in executor

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `execution/executor.py:210, 237`
- **Bug:** `gc.collect()` + `torch.cuda.empty_cache()` after every single batch/item. `gc.collect()` is expensive with large Python object graphs. For vLLM-managed backends, `empty_cache()` is a no-op.
- **Fix:** Gated behind interval — GPU memory cleared every 10 batches/items instead of every one.
- **Impact:** Reduces overhead in batch processing. Most benefit with large batch counts.
- **Effort:** Small

### 2.6 Checkpoint callback skips when batch crosses modulo boundary

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `execution/executor.py:206-208`
- **Bug:** `len(results) % checkpoint_every == 0` can be skipped entirely when a batch adds multiple results at once (e.g., results jumps from 45 to 77, skipping checkpoint at 50/64).
- **Fix:** Replaced modulo check with `items_since_checkpoint >= checkpoint_every` counter that resets after each checkpoint.
- **Impact:** Ensures checkpoints happen at expected intervals. Prevents data loss on crash.
- **Effort:** Small

### 2.7 `_tee` thread in subprocess runner can lose output on timeout

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `execution/runner.py:345-356`
- **Bug:** When `proc.wait()` raises `TimeoutExpired`, execution jumps to except block and `tee_thread` is never joined — it's daemonic and dies with the process, potentially losing the last buffered output (the most useful debugging info).
- **Fix:** Added `tee_thread.join(timeout=5)` in the `TimeoutExpired` except block.
- **Impact:** Preserves diagnostic output when experiments time out.
- **Effort:** Small

### 2.9 Enforce and document query transform iteration-0-only semantics

- **Status:** `[x]` Fixed 2026-02-11
- **Files:** `agents/iterative_rag.py`, `agents/base.py`, docs
- **Issue:** Query transforms (HyDE, multiquery) should only run on iteration 0 of iterative agents. On later iterations the query has been LLM-refined and is already document-like — applying HyDE again is redundant and degrades retrieval quality. The iteration guard was added in 2.4b but is implicit (a single `if iteration == 0` check). There is no warning/error if someone tries to bypass this, and no documentation explaining why.
- **Fix:**
  1. Add a `logger.warning` (or raise) if `query_transformer` is invoked on iteration > 0.
  2. Document the rationale in a docstring on the iteration loop and in `AGENT_PERFORMANCE_ANALYSIS.md`.
  3. Consider adding a config-level `query_transform_iterations` parameter if future use cases need transform on specific iterations.
- **Impact:** Prevents silent correctness bugs if the guard is accidentally removed during refactoring.
- **Effort:** Small

### 2.8 `detect_state` and `check_health` read JSON files multiple times

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `state/health.py`
- **Bug:** `check_health` calls `detect_state` which reads `state.json`, `results.json`, or `predictions.json`. Then `check_health` reads `questions.json` and `predictions.json` again. For experiments with large `predictions.json` (thousands of entries), this repeated parsing is expensive.
- **Fix:** Added `_load_json()` helper; `check_health` pre-loads `questions.json` and `predictions.json` once and reuses them. `detect_state` still loads independently (it's called standalone elsewhere), but `check_health`'s own redundant reads are eliminated.
- **Impact:** Faster experiment resume and health checks.
- **Effort:** Small

---

## Phase 3: Test Coverage (P2)

Missing tests that block safe refactoring of Phase 4+ items.

### 3.1 IterativeRAGAgent.run() — zero functional tests

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `agents/iterative_rag.py` (~535 lines)
- **Gap:** The multi-iteration loop, sufficiency checking, query refinement, document merging, convergence, and `_apply_reranking` are all untested. Only factory wiring tests verify the agent is _created_ but never _executed_.
- **Tests needed:**
  - Verify queries converge when sufficiency is detected
  - Verify refinement changes query text
  - Verify `max_iterations` is respected
  - Test `_merge_documents()` deduplication separately
  - Test early stopping behavior
- **Impact:** Unblocks safe refactoring of iterative agent logic (1.2, 4.1, 4.9).
- **Effort:** Medium–Large

### 3.2 SelfRAGAgent.run() — zero functional tests

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `agents/self_rag.py` (~523 lines)
- **Gap:** Adaptive retrieval pipeline (confidence assessment, retrieval decision, verification, fallback) entirely untested. `_parse_confidence()` regex parsing and `_parse_verification()` response parsing have no tests.
- **Tests needed:**
  - `_parse_confidence()` with valid (`"CONFIDENCE: 0.7"`), missing, edge cases (`1.5`, `-0.3`)
  - `_parse_verification()` with various response formats
  - `run()` with mocks verifying assess→retrieve→generate→verify pipeline
  - Test fallback generation path
- **Impact:** Unblocks safe refactoring of self-rag logic.
- **Effort:** Medium–Large

### 3.3 ResilientExecutor — zero tests

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `execution/executor.py` (~240 lines)
- **Gap:** The batch processing engine handling GPU memory management, error recovery, sequential fallback, and consecutive-failure abort (5 failures) is completely untested.
- **Tests needed:**
  - Normal batch execution
  - Sequential fallback when `batch_answer` unavailable
  - Consecutive failure abort behavior (5-failure threshold)
  - Checkpoint callback invocation
  - Error item creation format
- **Impact:** Unblocks safe changes to execution pipeline (2.5, 2.6).
- **Effort:** Medium

### 3.4 ExperimentState — zero tests

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `state/experiment_state.py`
- **Gap:** The phase state machine (INIT→GENERATING→GENERATED→COMPUTING_METRICS→COMPLETE) enabling crash recovery has no tests.
- **Tests needed:**
  - `save()`/`load()` round-trip
  - `advance_to()` phase transitions
  - `is_at_least()`/`is_past()` ordering logic for all phase pairs
  - `set_error()` marking
  - Invalid phase transitions
- **Impact:** Unblocks safe changes to resume logic.
- **Effort:** Small–Medium

### 3.5 ExperimentIO atomic write — zero direct tests

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `utils/experiment_io.py`
- **Gap:** The atomic write pattern (temp → rename) is a critical data integrity mechanism with zero tests.
- **Tests needed:**
  - Verify final file exists and temp file doesn't after write
  - Verify valid JSON content
  - Verify behavior with missing parent directories
  - Verify atomicity (concurrent reader doesn't see partial content)
- **Impact:** Validates the foundation that Phase 1.3 consolidation builds on.
- **Effort:** Small

### 3.6 PromptBuilder — zero tests

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `utils/prompts.py`
- **Gap:** The prompt template builder used by all agents has no tests. `_fewshot_cache` behavior untested.
- **Tests needed:**
  - `build_rag()` and `build_direct()` with various inputs
  - Few-shot prompt formatting
  - Context truncation behavior
- **Effort:** Small

### 3.7 HybridSearcher — no functional tests

- **Status:** `[x]` Fixed 2026-02-11
- **File:** `retrievers/hybrid.py`
- **Gap:** The alpha-weighted dense+sparse fusion and RRF merge logic is only wiring-tested (mocked). No functional test of the actual fusion algorithm.
- **Tests needed:**
  - RRF merge with known dense/sparse rankings → verify final ordering
  - Alpha=0 (sparse only), alpha=1 (dense only), alpha=0.5 (balanced)
  - Empty results from one source
- **Effort:** Medium

### 3.8 Execution phases — generation and metrics untested

- **Status:** `[ ]`
- **Files:** `execution/phases/generation.py`, `execution/phases/metrics_phase.py`
- **Gap:** Generation phase handler and metrics phase handler have zero tests. Only `init_phase.py` is tested.
- **Effort:** Medium

### 3.9 Provider ref-counting — zero tests

- **Status:** `[x]` Fixed 2026-02-11
- **Files:** `models/providers/generator.py`, `models/providers/embedder.py`, `models/providers/reranker.py`
- **Gap:** The ref-counting mechanism added in 2026-02-11 (2.4a) has no unit tests. Nested `load()` should reuse the model; only the outermost exit should unload. Edge cases: exception during first load with refcount>0, concurrent-like interleaving.
- **Tests needed:**
  - Single `load()` → model loaded, exit → model unloaded (backward compat)
  - Nested `load()` → model loaded once, inner exit → model still loaded, outer exit → unloaded
  - Exception during first `load()` → refcount reset to 0, no stale state
  - `_refcount` never goes negative
- **Impact:** Validates the foundation that all agent session wrapping depends on.
- **Effort:** Small–Medium

### 3.10 Consolidate duplicate mocks across test files

- **Status:** `[ ]`
- **Files:** `tests/test_agents.py`, `tests/test_cache.py`, `tests/test_config_wiring.py`, `tests/conftest.py`
- **Gap:** Each test file defines its own `MockGenerator`, `MockEmbedder`, `MockGeneratorProvider`, `MockEmbedderProvider`, `MockVectorIndex` — duplicating what `conftest.py` already provides. Inconsistent mock behavior across files.
- **Fix:** Consolidate into `conftest.py`. Ensure shared mocks match real interfaces.
- **Impact:** Reduces duplication, ensures mock consistency, makes writing new tests easier.
- **Effort:** Medium

---

## Phase 4: Interface & Design Cleanup (P3)

Structural issues that hurt maintainability and extensibility.

### 4.1 Triplicated `_apply_reranking` across three agents

- **Status:** `[ ]`
- **Files:** `agents/fixed_rag.py:196-243`, `agents/iterative_rag.py:488-517`, `agents/self_rag.py:437-466`
- **Problem:** Three nearly-identical copies with subtle inconsistencies. `FixedRAGAgent` accesses `self.reranker_provider.model_name` directly and uses manual `perf_counter`; others access `.config.model_name` and use `StepTimer`. Bug in one copy won't be fixed in others.
- **Fix:** Extract shared `apply_reranking()` function into `agents/base.py` alongside `batch_embed_and_search`.
- **Impact:** ~90 lines of duplication removed. Eliminates divergence risk.
- **Effort:** Small–Medium
- **Depends on:** 3.1, 3.2 (tests for the agents first)

### 4.2 Searcher protocol doesn't match HybridSearcher.batch_search

- **Status:** `[ ]`
- **Files:** `core/types.py:76-102`, `retrievers/hybrid.py:118-123`
- **Problem:** `Searcher` protocol requires `(query_embeddings, top_k)` but `HybridSearcher` requires extra `query_texts`. The formal type contract is broken. `is_hybrid_searcher()` duck-typing at `agents/base.py:208` works around it.
- **Fix options:**
  - (a) Add optional `query_texts: list[str] | None = None` to `Searcher` protocol
  - (b) Create a separate `HybridSearcher` protocol
  - (c) Have `HybridSearcher` accept query texts via a different mechanism (e.g., `set_query_texts()` before `batch_search`)
- **Recommendation:** Option (a) is simplest and maintains a single interface.
- **Impact:** Restores type safety for search backends.
- **Effort:** Small–Medium

### 4.3 Remove stale `core/schemas.py`

- **Status:** `[ ]`
- **File:** `core/schemas.py` (entire file)
- **Problem:** Contains `RetrievedDoc`, `PredictionRecord`, `RAGResponseMeta`, `PipelineLog`, `QueryTransformStep`, `RetrievalStep`, `RerankStep` — a complete duplicate tracing system unused by any agent. Agents use `Step`/`AgentResult`/`RetrievedDocInfo` from `agents/base.py` exclusively. Exported from `core/__init__.py`, creating confusion about canonical types.
- **Fix:** Remove the file and clean up imports from `core/__init__.py`.
- **Impact:** Eliminates confusion about canonical data types. Reduces cognitive load.
- **Effort:** Small (verify no imports, delete)

### 4.4 Pervasive `Any` typing in agents

- **Status:** `[ ]`
- **Files:** `agents/base.py:217-226`, `agents/fixed_rag.py:68-71`, `agents/iterative_rag.py:110-114`, `agents/self_rag.py:131-135`
- **Problem:** Nearly every parameter in `batch_transform_embed_and_search` and RAG agent constructors is typed `Any`. Typos in keyword arguments silently swallowed. No IDE autocompletion or type checking benefit.
- **Fix:** Use `TYPE_CHECKING` imports:
  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from ragicamp.cache.retrieval_store import RetrievalStore
      from ragicamp.models.providers.reranker import RerankerProvider
      from ragicamp.rag.query_transform import QueryTransformer
  ```
- **Impact:** Highest-ROI change for IDE support and maintainability.
- **Effort:** Medium (many signatures to update)

### 4.5 `Agent.__init__` accepts **config kwargs — dead storage

- **Status:** `[ ]`
- **File:** `agents/base.py:523-525`
- **Problem:** `self.config = config` stores kwargs but never reads them. `FixedRAGAgent(name="test", top_kk=5)` (typo) silently succeeds.
- **Fix:** Remove `**config`. If extensibility is needed, use explicit `extra_config: dict | None = None`.
- **Impact:** Catches typos at construction time.
- **Effort:** Small

### 4.6 Duplicate Embedder interface definitions

- **Status:** `[ ]`
- **Files:** `models/embedder.py:20-58` (Protocol), `models/providers/embedder.py:105-120` (ABC)
- **Problem:** Two separate `Embedder` interfaces with different method names (`encode` vs `batch_encode`, `get_sentence_embedding_dimension` vs `get_dimension`). Both importable from `models/`. Confusing for new code.
- **Fix:** Consolidate to one interface. The provider-level `Embedder` with `batch_encode`/`get_dimension` is cleaner. Adapt or alias the legacy protocol.
- **Impact:** Single source of truth for embedder interface.
- **Effort:** Medium (need to update all implementations)

### 4.7 MetricFactory and DatasetFactory use long if-elif chains

- **Status:** `[ ]`
- **Files:** `factory/metrics.py:29-149`, `factory/datasets.py:64-116`
- **Problem:** 120-line method with 7 try/except import blocks and 8 if-elif branches. Adding a new metric/dataset requires modifying the factory method. `_custom_metrics` registry exists but is unused.
- **Fix:** Use a registry dictionary pattern:
  ```python
  _METRICS = {"bertscore": ("ragicamp.metrics.bertscore", "BERTScoreMetric"), ...}
  ```
  Each entry: `(module_path, class_name)`. Factory does lazy import + instantiation from registry.
- **Impact:** Adding new metrics/datasets becomes a one-line registry entry instead of 15-line if-elif block.
- **Effort:** Medium

### 4.8 `_build_rag_specs` has 8 levels of nesting

- **Status:** `[ ]`
- **File:** `spec/builder.py:342-404`
- **Problem:** 8 nested `for` loops for the Cartesian product. Hard to read and modify.
- **Fix:** Use `itertools.product`:
  ```python
  for model, ret_name, top_k, prompt, qt, rr_cfg, dataset in product(
      models, retriever_names, top_k_values, prompts, query_transforms, reranker_cfgs, datasets
  ):
  ```
- **Impact:** Readability improvement. Easier to add/remove dimensions.
- **Effort:** Small

### 4.9 HybridSearcher duplicates logic between search and batch_search

- **Status:** `[ ]`
- **File:** `retrievers/hybrid.py`
- **Problem:** Entire sparse-to-SearchResult conversion and RRF merge logic is copy-pasted between `search()` and `batch_search()`.
- **Fix:** Have `batch_search` call `search` per query, or extract shared `_rrf_merge()`.
- **Impact:** Single implementation of RRF merge logic.
- **Effort:** Small

### 4.10 Migrate `@validator` to `@field_validator` (Pydantic v2)

- **Status:** `[ ]`
- **File:** `config/schemas.py` (lines 95, 170, 178, 209, 276, 284, 292, 300, 310)
- **Problem:** Deprecated Pydantic v1 `@validator` generates deprecation warnings.
- **Fix:** Migrate to `@field_validator` with v2 API.
- **Effort:** Medium (9 validators to migrate)

### 4.11 `DocumentCorpus.load` uses `raise NotImplementedError` instead of `@abstractmethod`

- **Status:** `[ ]`
- **File:** `corpus/base.py:77`
- **Problem:** Allows creating instances of `DocumentCorpus` directly (fails at runtime instead of instantiation time).
- **Fix:** Make `load` an `@abstractmethod`.
- **Effort:** Small

### 4.12 `LanguageModel.get_embeddings` is abstract but not applicable to all backends

- **Status:** `[ ]`
- **File:** `models/base.py:67-77`
- **Problem:** Required as abstract method, but `VLLMModel` (the primary backend) raises `NotImplementedError`.
- **Fix:** Make non-abstract with default `NotImplementedError`, or split into separate generator/embedder ABCs.
- **Effort:** Small

### 4.13 Metric per-item scores stored as mutable instance state

- **Status:** `[ ]`
- **File:** `metrics/base.py:54, 93-94`
- **Problem:** `self._last_per_item` is a side-channel. If `compute()` is called again before `get_per_item_scores()`, old values are lost. `AsyncAPIMetric` uses `self._last_results` instead — inconsistent.
- **Fix:** Return per-item scores as part of `compute()` return value (e.g., `ComputeResult` dataclass with `scores` + `per_item`), or always use `compute_with_details()`.
- **Impact:** Eliminates subtle stateful bugs in metric computation.
- **Effort:** Medium (interface change across all metrics)

### 4.14 `SentenceChunker` misuses `chunk_size` as sentence count

- **Status:** `[ ]`
- **File:** `corpus/chunking.py:142-143`
- **Problem:** `ChunkConfig.chunk_size` is documented as "target size in characters" but `SentenceChunker` uses it as number of sentences. `chunk_size=512` (characters) → 512-sentence chunks.
- **Fix:** Use a separate config field for sentence count, or convert character size to approximate sentence count.
- **Effort:** Small–Medium

---

## Phase 5: Minor Polish & Consistency (P4)

### 5.1 Inconsistent logging — `import logging` vs `get_logger`

- **Status:** `[ ]`
- **Files:** `cli/commands.py:11`, `state/health.py:11`
- **Fix:** Use `from ragicamp.core.logging import get_logger` consistently.
- **Effort:** Small

### 5.2 Mixed `Optional[X]` and `X | None` syntax

- **Status:** `[ ]`
- **Files:** Various (e.g., `execution/runner.py` imports `Optional` but uses `X | None` on line 147)
- **Fix:** Use `X | None` consistently (Python 3.10+ target).
- **Effort:** Small (grep and replace)

### 5.3 `gradient_checkpointing_enable` called during inference

- **Status:** `[ ]`
- **File:** `models/huggingface.py:80-81`
- **Problem:** Trades compute for memory during backprop, but this is inference-only (`model.eval()`). Adds overhead without benefit.
- **Fix:** Remove the two lines.
- **Effort:** Small

### 5.4 Missing `gc.collect()` in SentenceTransformerEmbedder.unload

- **Status:** `[ ]`
- **File:** `models/st_embedder.py:134-150`
- **Fix:** Add `gc.collect()` between model deletion and CUDA cache clearing, matching other `unload()` methods.
- **Effort:** Small

### 5.5 `spec/__init__.py` exports private `_model_short` and `_spec_hash`

- **Status:** `[ ]`
- **File:** `spec/__init__.py:11, 18-19`
- **Fix:** Either drop underscore prefix (they're intentionally public) or remove from `__all__`.
- **Effort:** Small

### 5.6 `is_hybrid_searcher` uses fragile `hasattr` duck typing

- **Status:** `[ ]`
- **File:** `agents/base.py:208-215`
- **Fix:** Use `isinstance(index, HybridSearcher)` or check `_is_hybrid` flag on `LazySearchBackend`.
- **Effort:** Small

### 5.7 Magic numbers for context truncation

- **Status:** `[ ]`
- **Files:** `agents/iterative_rag.py:328` (`2000`), `agents/self_rag.py:381` (`3000`)
- **Fix:** Define `MAX_CONTEXT_CHARS` in `core/constants.py` or make configurable via `agent_params`.
- **Effort:** Small

### 5.8 `RerankerWrapper.rerank` uses bare `list` type hints

- **Status:** `[ ]`
- **File:** `models/providers/reranker.py:91-96, 128-133`
- **Fix:** Change to `list[Document]` and `list[list[Document]]`.
- **Effort:** Small

### 5.9 Duplicate `MODELS` dict in CrossEncoderReranker and RerankerProvider

- **Status:** `[ ]`
- **Files:** `rag/rerankers/cross_encoder.py:39-45`, `models/providers/reranker.py:35-41`
- **Fix:** Define once in a shared location.
- **Effort:** Small

### 5.10 `SentenceTransformerWrapper.unload` — redundant `del` before `= None`

- **Status:** `[ ]`
- **File:** `models/providers/embedder.py:152-163`
- **Fix:** Just `self._model = None` + `gc.collect()`.
- **Effort:** Small

### 5.11 `QAExample.metadata` uses `None` default instead of `field(default_factory=dict)`

- **Status:** `[ ]`
- **File:** `datasets/base.py:30`
- **Fix:** Use `field(default_factory=dict)`.
- **Effort:** Small

### 5.12 `should_skip_file` overly broad substring match

- **Status:** `[ ]`
- **File:** `cli/backup.py:108-119`
- **Fix:** Use `fnmatch` or restrict substring matching to filename only.
- **Effort:** Small

### 5.13 `GPUProfile` threshold constants treated as dataclass fields

- **Status:** `[ ]`
- **File:** `models/providers/gpu_profile.py:25-26`
- **Fix:** Use `ClassVar[float]` annotation.
- **Effort:** Small

### 5.14 Duplicate GPU tier detection logic

- **Status:** `[ ]`
- **Files:** `models/providers/gpu_profile.py:29-71`, `models/vllm_embedder.py:82-121`
- **Fix:** Consolidate into `GPUProfile` by adding embedder-specific profiles.
- **Effort:** Medium

### 5.15 `HFGeneratorWrapper.batch_generate` hardcoded `max_length=4096`

- **Status:** `[ ]`
- **File:** `models/providers/generator.py:216`
- **Fix:** Pull from config or make a named constant.
- **Effort:** Small

### 5.16 `sys.path` mutation in tests

- **Status:** `[ ]`
- **Files:** `tests/test_analysis_utils.py:10-11`, `tests/test_config_wiring.py:527-528`
- **Fix:** Use `importlib` or make the scripts proper importable modules.
- **Effort:** Small

### 5.17 Global mutable singleton in ArtifactManager

- **Status:** `[ ]`
- **File:** `utils/artifacts.py:301-317`
- **Fix:** Raise error if `base_dir` differs from existing, or remove singleton pattern.
- **Effort:** Small

### 5.18 `_fewshot_cache` is mutable class variable, never invalidated

- **Status:** `[ ]`
- **File:** `utils/prompts.py:67`
- **Fix:** Use instance-level cache or add invalidation.
- **Effort:** Small

### 5.19 `IndexBuilder._process_batch` returns 1D empty array for empty batches

- **Status:** `[ ]`
- **File:** `indexes/index_builder.py:256`
- **Fix:** Return `np.empty((0, dim), dtype=np.float32)` or guard in caller to skip `faiss_index.add` when empty.
- **Effort:** Small

### 5.20 Widespread ruff lint violations (30+ errors across codebase)

- **Status:** `[ ]`
- **Files:** 60 files flagged by `ruff format --check`; 30 `ruff check` errors across agents, providers, factory
- **Problem:** Pre-existing violations include:
  - **UP035** — `from typing import Iterator/Callable` → should use `collections.abc` (Python 3.10+)
  - **B905** — `zip()` without `strict=` parameter (20+ occurrences)
  - **F401** — Unused imports (`batch_embed_and_search`, `Optional`, `Any`)
  - **B027** — Empty method in ABC without `@abstractmethod` (`Generator.unload`, `Embedder.unload`)
  - **I001** — Unsorted import blocks
- **Fix:** Run `ruff check --fix --unsafe-fixes` for auto-fixable issues; manually review B905 (add `strict=True` where lengths are guaranteed equal).
- **Impact:** Clean CI, consistent code style, catches real bugs (unused imports, unsafe zip).
- **Effort:** Small–Medium (mostly auto-fixable)

### 5.21 Pickle security boundary undocumented

- **Status:** `[ ]`
- **Files:** `indexes/sparse.py:315`, `indexes/vector_index.py:418`, `indexes/hierarchical.py:496`
- **Fix:** Add security note in CLAUDE.md and near pickle.load calls. Consider checksum verification for downloaded artifacts.
- **Effort:** Small

---

## Future Enhancements (Not bugs — design improvements)

These are not issues per se but architectural improvements for when the project grows.

| ID | Enhancement | Impact |
|----|-------------|--------|
| F1 | Unified `SearchBackend` protocol with optional `query_texts` | Clean polymorphism for all search backends |
| F2 | Registry pattern for metrics, datasets, and model backends | Eliminates all factory if-elif chains |
| F3 | Enums for FAISS index types (`flat`/`ivf`/`hnsw`/`ivfpq`) and embedding backends (`vllm`/`sentence_transformers`) | Prevents stringly-typed config bugs |
| F4 | `ComputeResult` return type for metrics (scores + per_item + metadata) | Eliminates stateful per-item side-channel |
| F5 | Extract `_QueryState` base from IterativeRAG/SelfRAG | Shared state management for iterative agents |
| F6 | Remove dead `create_*_agent` convenience functions at bottom of agent modules | Reduces dead code |
| F7 | Document scoring semantics (Faithfulness: 0=faithful, Hallucination: 1=hallucinated) | Prevents misinterpretation of metric scores |
| F8 | Add `__all__` exports to `rag/__init__.py` and sub-packages | Cleaner public API surface |
| F9 | `ExperimentSpec.to_dict` always include all fields for predictable round-trips | Symmetric serialization |
| F10 | Property-based tests (hypothesis) for ExperimentSpec round-trip, parsing functions | Stronger correctness guarantees |
| F11 | Vectorized RRF merge in HybridSearcher using numpy arrays | Faster hybrid search for large batches |
| F12 | Batched BM25 scoring (replace `rank_bm25` sequential calls) | Faster sparse search, hard due to library limitation |

---

## Progress Summary

| Phase | Total | Done | Remaining |
|-------|-------|------|-----------|
| P0 — Data Integrity | 8 | 8 | 0 |
| P1 — Reliability | 15 | 13 | 2 |
| P2 — Test Coverage | 10 | 8 | 2 |
| P3 — Interface/Design | 14 | 0 | 14 |
| P4 — Polish | 21 | 0 | 21 |
| **Total** | **68** | **29** | **39** |
