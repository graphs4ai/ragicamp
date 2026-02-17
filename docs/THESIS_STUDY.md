# Thesis Study: Smart Retrieval with Small Language Models

> **Last updated**: 2026-02-17
> **Status**: Phase 1 running (~60% complete, resuming with trimmed search space)

---

## Research Question

**"Can high-quality retrieval compensate for smaller, faster LLMs in open-domain QA?"**

We invest computational budget in retrieval (strong embeddings, reranking, advanced
agents) rather than in large generators. This tests whether RAG quality is bottlenecked
by retrieval or generation — and establishes baselines for a future RL agent that will
learn to make retrieval decisions dynamically per query.

## Thesis Phases

### Phase 1: Bayesian Landscape Exploration (current)
**Study**: `smart_retrieval_slm` | **Config**: `conf/study/smart_retrieval_slm.yaml`

Optuna TPE explores a ~200K-combination search space to understand what helps each
model x dataset pair. Dataset and model are stratified (round-robin), so every combo
gets equal trial budget. TPE optimizes the remaining dimensions (retriever, top_k,
prompt, query_transform, reranker, agent_type).

**Purpose**: Establish baselines and identify which RAG components matter most per
model tier (tiny/small/medium) and per dataset difficulty (TriviaQA/NQ/HotpotQA).

### Phase 2: RL Agent (planned)
An RL agent will receive a query and choose actions (retrieve, rerank, change prompt,
generate, etc.) to maximize performance. Phase 1 results serve as the baseline and
initial training signal — the RL agent should match or exceed the best static configs
found by Optuna.

---

## Study Design

### Models (7, was 8 — Phi-3 removed)

| Tier | Model | Params | Notes |
|------|-------|--------|-------|
| Tiny | Qwen2.5-1.5B-Instruct | 1.5B | Smallest viable SLM |
| Tiny | gemma-2-2b-it | 2B | Gemma 2 architecture |
| Small | Llama-3.2-3B-Instruct | 3B | Meta, solid baseline |
| Small | Qwen2.5-3B-Instruct | 3B | Best MMLU in class |
| Medium | Mistral-7B-Instruct-v0.3 | 7B | Efficient 7B |
| Medium | Qwen2.5-7B-Instruct | 7B | Strong performer |
| Medium | gemma-2-9b-it | 9B | Best in class (upper bound) |

**Removed**: Phi-3-mini-4k-instruct (4096 context limit caused ~24% of all trial
failures — most RAG configs overflow it).

### Datasets (3)

| Dataset | Type | Difficulty |
|---------|------|------------|
| Natural Questions (NQ) | Wikipedia-based, open-domain | Medium |
| TriviaQA | Trivia, web documents | Easy |
| HotpotQA | Multi-hop reasoning | Hard |

1000 questions per experiment for statistical significance.

### Search Space

| Dimension | Values | Count | Optimized by |
|-----------|--------|-------|-------------|
| model | 7 models | 7 | Stratified (round-robin) |
| dataset | nq, triviaqa, hotpotqa | 3 | Stratified (round-robin) |
| retriever | dense x3, hybrid x3, hierarchical x1 | 7 | TPE |
| top_k | 3, 5, 10, 15, 20 | 5 | TPE |
| prompt | concise, concise_strict, concise_json, extractive, extractive_quoted, cot, cot_final, fewshot_1, fewshot_3 | 9 | TPE |
| query_transform | none, hyde, multiquery | 3 | TPE |
| reranker | none, bge, bge-v2 | 3 | TPE |
| agent_type | fixed_rag, iterative_rag, self_rag | 3 | TPE |

Conditional dimensions (hybrid-only): `rrf_k` [20, 40, 60], `alpha` [0.3, 0.5, 0.7, 0.9]
Conditional dimensions (agent-specific): `max_iterations` [1, 2], `retrieval_threshold`, `verify_answer`, etc.

Total theoretical: ~200K+ combinations. Target: 1000 TPE trials.

### Baselines

42 direct LLM experiments (no retrieval): 7 models x 2 prompts (concise, fewshot_3) x 3 datasets.
These ran before the TPE phase and completed 100%.

---

## Progress (as of 2026-02-17)

### Overall
- **Started**: 2026-02-10 17:32
- **Runtime**: 7 days
- **Baselines**: 42/42 complete (100%)
- **TPE trials**: ~603/1000 complete (60%)
- **Last trial**: #651 (study paused for YAML trim)

### Outcome Distribution (941 trials started)

| Outcome | Count | Rate |
|---------|-------|------|
| Successful (got F1) | 610 | 65% |
| Experiment failures | 245 | 26% |
| Context-limit pruned | 25 | 3% |
| Timeouts (60min) | 20 | 2% |
| Running/incomplete | 41 | 4% |

### Top 5 Configurations

| F1 | Model | Retriever | Prompt | QT | Reranker | Agent | Dataset |
|----|-------|-----------|--------|----|----------|-------|---------|
| 0.674 | gemma-2-9b-it | hybrid_bge_large_bm25 | fewshot_1 | multiquery | bge-v2 | iterative(3) | triviaqa |
| 0.667 | gemma-2-9b-it | hier_bge_large_2048p | fewshot_1 | none | bge-v2 | self_rag | triviaqa |
| 0.665 | gemma-2-9b-it | dense_bge_large_512 | fewshot_1 | none | bge-v2 | iterative(3) | triviaqa |
| 0.662 | gemma-2-9b-it | hybrid_gte_qwen2_bm25 | fewshot_1 | none | bge-v2 | iterative(3) | triviaqa |
| 0.660 | gemma-2-9b-it | hybrid_bge_large_bm25 | fewshot_1 | none | bge-v2 | iterative(3) | triviaqa |

### Key Findings So Far

1. **Model size dominates**: All top 31 configs (F1 > 0.6) use gemma-2-9b-it or Mistral-7B.
   No 1.5B-3B model breaks into the top tier. Premium retrieval helps but doesn't close
   the gap — partially refuting the core hypothesis.

2. **TriviaQA is easy, NQ/HotpotQA are hard**: All 31 high-performing trials are on TriviaQA.
   No config achieves F1 > 0.6 on NQ or HotpotQA yet.

3. **bge-v2 reranker is essential**: 81% of successful trials use it. Single biggest
   component-level effect.

4. **fewshot_1 >> complex prompts**: Simple one-shot examples outperform CoT, extractive,
   and structured prompts. Lower token overhead + enough format guidance.

5. **HyDE barely helps**: Only 6% of successes use it, vs 73% with no transform. The
   latency cost doesn't pay off for these models/datasets.

6. **iterative_rag has highest ceiling, lowest floor**: Best F1 (0.674) but also 38%
   failure rate and most timeouts.

7. **fixed_rag is most reliable**: 81% success rate but lower ceiling (~0.64 max F1).

---

## YAML Changes Log

### 2026-02-17: Search Space Trim (reduce waste, improve remaining trial quality)

1. **Removed Phi-3-mini-4k-instruct** — 4096 context limit caused ~24% of all failures.
   Most RAG configs (especially high top_k + fewshot) overflow it. Not useful as an RL
   baseline since most actions are unavailable to it.

2. **Reduced trial_timeout: 3600 → 1800** — Configs needing >30 min are degenerate
   (usually iterative_rag with max_iter=3 + hybrid + reranker stacking). The RL agent
   won't want slow configs anyway.

3. **Dropped max_iterations: 3 for iterative_rag** — Main cause of timeouts and failures.
   Kept [1, 2]. The RL agent will learn its own iteration policy; it doesn't need a
   static max_iter=3 baseline.

These changes reduce the stratified combos from 24 (8 models × 3 datasets) to 21
(7 models × 3 datasets), giving ~47 trials per combo at 1000 total. Existing 603
trials in the Optuna DB remain valid.

---

## What This Study Provides for Phase 2 (RL Agent)

### Per model × dataset baselines
After 1000 trials with 21 combos, each pair will have ~40+ successful trials spanning
different retriever/reranker/prompt/agent configs. This establishes:

- **Ceiling per combo**: What's the best F1 achievable with static config?
- **Component effect sizes**: How much does each action (rerank, query_transform, etc.)
  help for this specific model on this specific dataset?
- **Failure modes**: Which action combinations the RL agent should learn to avoid.

### Action space definition
The RL agent's action space maps directly to the TPE dimensions:
- Choose retriever type (dense/hybrid/hierarchical)
- Choose top_k
- Choose whether to rerank (and which reranker)
- Choose whether to apply query transform (and which one)
- Choose prompt strategy
- Choose whether to iterate or stop

### Training signal
Each completed trial is a (state, action, reward) tuple:
- **State**: (model, dataset, query)
- **Action**: (retriever, top_k, prompt, qt, reranker, agent_type, agent_params)
- **Reward**: F1 score

---

## Related Docs

- `docs/OPTUNA_STUDY_DESIGN.md` — Technical details of the TPE optimization loop
- `docs/AGENT_PERFORMANCE_ANALYSIS.md` — Provider load/unload optimization (ref-counting)
- `docs/FUTURE_WORK.md` — Engineering roadmap and research extensions
- `docs/PAPER_IDEAS.md` — Intermediary publication ideas from this study
- `conf/study/smart_retrieval_slm.yaml` — Active study config
