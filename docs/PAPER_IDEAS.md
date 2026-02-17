# Intermediary Publication Ideas

> **Last updated**: 2026-02-17
> **Context**: Derived from the `smart_retrieval_slm` study (~1000 TPE trials, 7 SLMs, 3 datasets)

These are publication ideas that can be derived from the current study results
and RAGiCamp codebase, ordered roughly by ease of publication. Each can be
written independently of the full thesis.

---

## 1. RAG Component Sensitivity for Small Language Models

**Type**: Workshop paper (4 pages)
**Venue**: ACL/EMNLP workshops (Efficient NLP, Retrieval-Augmented Generation)
**Difficulty**: Low -- data and notebooks already exist

**Core claim**: Variance decomposition across ~600+ RAG experiments shows which
components matter most for SLMs (1.5B-9B). Reranker > retriever type > prompt >
query transform, and the ranking differs from what large-model studies report.

**Data needed**: Already available. `nb03_component_analysis` produces variance
decomposition figures; `nb10_statistical_tests` produces significance tables.

**Why publishable**:
- Most RAG sensitivity studies use 70B+ models; no systematic study exists for
  SLMs in the 1.5B-9B range.
- Practical takeaway: practitioners deploying SLMs know exactly which knobs to
  tune first (reranker >> everything else).

**Outline**:
1. Introduction: RAG is standard, but component importance under SLMs is unknown
2. Setup: 7 models x 3 datasets x ~200K search space, TPE exploration
3. Results: Variance decomposition, marginal effects per component
4. Key finding: Reranker is the single biggest lever for SLMs
5. Discussion: Implications for SLM deployment

**Key figures**: Variance decomposition bar chart, marginal effects with CIs,
reranker impact heatmap.

---

## 2. When Does Retrieval Help Small Models?

**Type**: Short paper (4 pages)
**Venue**: EMNLP/NAACL main conference (short) or Findings
**Difficulty**: Low-Medium

**Core claim**: RAG helps SLMs conditionally. We characterize exactly when: by
model tier (tiny/small/medium), dataset difficulty (TriviaQA/NQ/HotpotQA), and
retrieval configuration. RAG sometimes hurts small models.

**Data needed**: `nb02_rag_vs_direct`, `nb06_when_rag_fails`, `nb10_statistical_tests`.

**Why publishable**:
- Challenges the assumption that RAG always helps.
- Statistical significance tests with Holm-Bonferroni correction give confidence
  in the conclusions.
- Actionable: "Don't blindly add RAG to your 1.5B model."

**Outline**:
1. Introduction: RAG assumed beneficial; is it true for small models?
2. Experimental setup: 7 SLMs, 3 QA datasets, best-RAG vs best-direct comparison
3. Results: RAG vs Direct per model x dataset with significance tests
4. Analysis: When does RAG hurt? (model tier, dataset difficulty, config)
5. Conclusion: RAG benefit is conditional on model capacity and dataset

**Key figures**: Forest plot of RAG vs Direct effect sizes, RAG benefit
distribution histogram, per-tier comparison table.

---

## 3. Retrieval vs Generation: Diagnosing RAG Bottlenecks

**Type**: Workshop paper (4 pages) or EMNLP Findings short paper
**Venue**: GenBench, Eval4NLP, or RAG-focused workshops
**Difficulty**: Medium

**Core claim**: Using answer-in-context analysis across hundreds of RAG configs,
we disentangle retrieval failures from generation failures and show how the
bottleneck shifts with model size. Smaller models are generation-bottlenecked
even when retrieval succeeds; larger models are retrieval-bottlenecked.

**Data needed**: `nb11_retrieval_diagnostics` (answer-in-context rates, quadrant
analysis, reranking effectiveness).

**Why publishable**:
- Novel diagnostic methodology applied at scale (not just one model/retriever)
- Practical: tells practitioners whether to invest in better retrieval or better
  generation for their model size
- Connects to the "lost in the middle" literature

**Outline**:
1. Introduction: RAG failures are opaque -- retrieval or generation?
2. Method: Answer-in-context heuristic + 4-quadrant classification
3. Scale: Applied across 7 models, 7 retrievers, 3 datasets
4. Results: Bottleneck shifts by model tier, reranker is critical for bridging
5. Discussion: Practical guidelines for RAG debugging

**Key figures**: Quadrant stacked bar chart by model tier, retrieval recall vs
F1 scatter, reranking rank-change distribution.

---

## 4. Reranking is All You Need (for SLM-based RAG)

**Type**: Short paper or technical blog post
**Venue**: SIGIR short paper, or high-visibility blog (HuggingFace, Weights & Biases)
**Difficulty**: Low

**Core claim**: Cross-encoder reranking (bge-v2) is the single most impactful
RAG component for SLMs -- more than retriever type, query transform, prompt
strategy, or agent type. 81% of top-performing configs use it. The effect is
statistically significant with large Cohen's d.

**Data needed**: `nb03_component_analysis`, `nb10_statistical_tests` (reranker
pairwise comparisons with CIs).

**Why publishable**:
- Strong, simple, actionable claim backed by statistical tests.
- Blog format: high visibility, fast to write.
- Paper format: short SIGIR paper with the statistical backing.

**Outline**:
1. Setup: ~600 RAG experiments across 7 SLMs
2. Finding: Reranker explains the most variance in F1
3. Evidence: Pairwise bootstrap tests (none vs bge vs bge-v2)
4. Why: Reranker compensates for weak embeddings + wrong top-K
5. Recommendation: Always rerank when using SLMs

---

## 5. RAGiCamp: An Open Framework for Reproducible RAG Experiments

**Type**: System/demo paper (4-6 pages)
**Venue**: EMNLP Demo track, SIGIR Resource track, or JOSS (Journal of Open Source Software)
**Difficulty**: Medium (requires polishing the repo)

**Core claim**: RAGiCamp is an open-source framework for running controlled,
reproducible RAG experiments at scale, with Optuna integration, subprocess
isolation, phase-aware resume, and per-component diagnostics.

**Data needed**: The repository itself + study results as demonstration.

**Why publishable**:
- Few open RAG benchmarking frameworks exist with this level of modularity.
- Subprocess isolation for CUDA crash safety is novel for RAG benchmarks.
- Phase-aware resume enables multi-day GPU studies without data loss.
- Optuna TPE with stratified round-robin is a useful contribution to BO for NLP.

**Outline**:
1. Introduction: Need for controlled RAG experiments
2. Architecture: Config -> Specs -> Subprocess execution -> Phase resume
3. Key features: Optuna TPE integration, lazy index loading, retrieval cache
4. Case study: smart_retrieval_slm (1000 trials, 7 models, 3 datasets)
5. Availability: Open source, MIT license

**Repository preparation needed**: Clean up README, add installation guide,
ensure `make run-baseline-simple` works out of the box, add example configs.

---

## 6. Bayesian Optimization for RAG Configuration Search

**Type**: Workshop paper (4 pages)
**Venue**: AutoML workshop, SIGIR short paper, or EMNLP NLP-Power workshop
**Difficulty**: Medium-High

**Core claim**: TPE with stratified round-robin over model/dataset dimensions
efficiently explores the ~200K RAG configuration space. We characterize the
optimization landscape: which dimensions are easy to optimize (reranker), which
require more trials (prompt x agent interactions), and where grid search wastes
budget.

**Data needed**: `nb08_tpe_optimization` (requires Optuna DB), Optuna trial
history analysis.

**Why publishable**:
- Applying Bayesian optimization to RAG configuration is novel.
- Stratified round-robin for benchmark fairness is a useful technique.
- The optimization landscape characterization is practically useful.

**Outline**:
1. Introduction: RAG has many knobs; grid search is infeasible
2. Method: TPE + stratified dims + context feasibility pruning
3. Analysis: Trial efficiency curves, convergence by dimension
4. Findings: Which dimensions TPE learns fastest, interaction effects
5. Recommendation: How many trials needed for reliable RAG tuning

---

## Recommended Priority

| # | Paper | Difficulty | Time to Write | Impact |
|---|-------|-----------|---------------|--------|
| 1 | Component Sensitivity | Low | 1-2 weeks | Medium |
| 4 | Reranking is All You Need | Low | 1 week | Medium-High |
| 2 | When Does RAG Help? | Low-Medium | 2-3 weeks | High |
| 3 | Retrieval vs Generation | Medium | 2-3 weeks | Medium-High |
| 5 | RAGiCamp Framework | Medium | 3-4 weeks | Medium |
| 6 | BO for RAG Config | Medium-High | 3-4 weeks | Medium |

**Recommendation**: Start with **#1 or #4** -- they require the least new work,
the notebooks already produce the figures, and they establish citations for your
thesis. A 4-page workshop paper can be drafted in a week from existing analyses.

---

## Related Docs

- `docs/THESIS_STUDY.md` -- Current study design, progress, and findings
- `docs/OPTUNA_STUDY_DESIGN.md` -- Technical details of the TPE optimization
- `docs/FUTURE_WORK.md` -- Engineering roadmap and research extensions
- `conf/study/smart_retrieval_slm.yaml` -- Active study config

## Analysis Notebooks

| Notebook | Paper(s) It Supports |
|----------|---------------------|
| `nb02_rag_vs_direct` | #2 (When Does RAG Help?) |
| `nb03_component_analysis` | #1 (Sensitivity), #4 (Reranking) |
| `nb06_when_rag_fails` | #2 (When Does RAG Help?) |
| `nb08_tpe_optimization` | #6 (BO for RAG) |
| `nb10_statistical_tests` | #1, #2, #3, #4 (all need significance) |
| `nb11_retrieval_diagnostics` | #3 (Retrieval vs Generation) |
