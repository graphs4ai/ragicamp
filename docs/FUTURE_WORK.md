# Future Work: RAG Research Roadmap

This document outlines potential improvements to ragicamp based on current RAG literature, organized by priority and implementation complexity.

## Table of Contents

1. [Current Capabilities](#current-capabilities)
2. [Immediate Priorities](#immediate-priorities)
3. [Query Processing Improvements](#query-processing-improvements)
4. [Retrieval Improvements](#retrieval-improvements)
5. [Post-Retrieval Processing](#post-retrieval-processing)
6. [Evaluation Improvements](#evaluation-improvements)
7. [Advanced RAG Patterns](#advanced-rag-patterns)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Current Capabilities

### What We Have (Solid Foundation)

| Component | Implementation | Status |
|-----------|---------------|--------|
| **Dense Retrieval** | BGE-large, BGE-M3 embeddings | ✅ Production |
| **Sparse Retrieval** | TF-IDF, BM25 | ✅ Production |
| **Hybrid Retrieval** | RRF fusion (dense + sparse) | ✅ Production |
| **Query Transform** | HyDE, MultiQuery | ✅ Production |
| **Reranking** | Cross-encoder (BGE-reranker) | ✅ Production |
| **Iterative RAG** | Multi-turn query refinement | ✅ Production |
| **Self-RAG** | Adaptive retrieval decision | ✅ Production |
| **Index Types** | Flat, IVF, HNSW | ✅ Production |
| **Answer Metrics** | F1, EM, BERTScore, BLEURT, LLM-judge | ✅ Production |

### What's Missing

This document covers techniques from the RAG literature that we haven't implemented yet, with justification for each.

---

## Immediate Priorities

### 1. Retrieval Quality Metrics

**Problem**: We only evaluate final answer quality, not retrieval quality. When experiments fail, we can't tell if the issue is bad retrieval or bad generation.

**Solution**: Add retrieval-specific metrics.

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Recall@k** | % of relevant docs retrieved | `|retrieved ∩ relevant| / |relevant|` |
| **Precision@k** | % of retrieved docs that are relevant | `|retrieved ∩ relevant| / k` |
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

---

### 2. Lost-in-the-Middle Mitigation

**Problem**: LLMs attend poorly to information in the middle of long contexts. Studies show U-shaped attention: strong at start/end, weak in middle.

**Evidence**: "Lost in the Middle" (Liu et al., 2023) showed 20-30% performance drop when relevant info is in middle positions.

**Solution**: Reorder retrieved passages.

```python
# In utils/formatting.py
def reorder_passages_for_attention(passages: list[Document]) -> list[Document]:
    """Reorder passages: most relevant at start and end, least relevant in middle.
    
    Given passages ranked [1, 2, 3, 4, 5] by relevance:
    Returns: [1, 3, 5, 4, 2] - alternating from edges inward
    
    This exploits the U-shaped attention pattern of LLMs.
    """
    if len(passages) <= 2:
        return passages
    
    n = len(passages)
    reordered = [None] * n
    
    # Interleave: best at start, second-best at end, third at start+1, etc.
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

**When to Use**:
- When `top_k > 3` (middle problem doesn't exist with few passages)
- Especially important for long-context experiments

**Experiment Design**:
```yaml
- name: e1_reorder_passages
  hypothesis: "Passage reordering improves answer extraction for top_k=5"
  retriever: dense_bge_large_512
  top_k: 5
  passage_order: attention_optimized  # vs 'relevance' (default)
```

---

## Query Processing Improvements

### 3. Query Decomposition (for Multi-Hop QA)

**Problem**: Complex questions require multiple retrieval steps. Neither HyDE nor Iterative RAG handles this.

**Clarification of Existing Methods**:

| Method | What It Does | Handles Multi-Hop? |
|--------|-------------|-------------------|
| **HyDE** | Generates hypothetical answer, embeds that instead of query | ❌ No - just reformats single query |
| **MultiQuery** | Generates paraphrases of same question | ❌ No - variations of same question |
| **Iterative RAG** | Refines query if retrieval insufficient | ⚠️ Partial - refines, doesn't decompose |
| **Query Decomposition** | Breaks into sub-questions, answers each | ✅ Yes - designed for this |

**Example**:
```
Original: "Who is the spouse of the director of Inception?"

HyDE output: "The spouse of the director of Inception is Emma Thomas..."
(Still one query, just reformatted)

Iterative RAG: 
  Round 1: "Who is the spouse of the director of Inception?" → insufficient
  Round 2: "Inception director spouse Christopher Nolan" → still one query

Query Decomposition:
  Sub-Q1: "Who directed Inception?" → "Christopher Nolan"
  Sub-Q2: "Who is Christopher Nolan's spouse?" → "Emma Thomas"
  Final: "Emma Thomas"
```

**Implementation**:
```python
# In rag/query_transform/decomposition.py
class QueryDecomposer(QueryTransformer):
    """Decompose complex questions into sub-questions."""
    
    DECOMPOSITION_PROMPT = """Break this question into simpler sub-questions that can be answered independently.
Only decompose if the question requires multiple pieces of information.

Question: {question}

If decomposition needed, output each sub-question on a new line prefixed with "SUB: ".
If no decomposition needed, output "ATOMIC: {question}"

Examples:
Question: "Who is the spouse of the director of Inception?"
SUB: Who directed Inception?
SUB: Who is [ANSWER_1]'s spouse?

Question: "What year was Python created?"
ATOMIC: What year was Python created?
"""
    
    def transform(self, query: str) -> list[str]:
        response = self.model.generate(
            self.DECOMPOSITION_PROMPT.format(question=query)
        )
        
        if response.startswith("ATOMIC:"):
            return [query]  # No decomposition needed
        
        sub_questions = []
        for line in response.split("\n"):
            if line.startswith("SUB:"):
                sub_questions.append(line[4:].strip())
        
        return sub_questions if sub_questions else [query]
```

**Agent Integration**:
```python
# In agents/decomposition_rag.py
class DecompositionRAGAgent(Agent):
    """RAG agent that decomposes complex questions."""
    
    def answer(self, question: str) -> RAGResponse:
        # Step 1: Decompose
        sub_questions = self.decomposer.transform(question)
        
        if len(sub_questions) == 1:
            # Atomic question, use standard RAG
            return self.base_agent.answer(question)
        
        # Step 2: Answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            # Replace placeholders with previous answers
            resolved_q = self._resolve_placeholders(sub_q, sub_answers)
            response = self.base_agent.answer(resolved_q)
            sub_answers.append(response.answer)
        
        # Step 3: Synthesize final answer
        return self._synthesize(question, sub_questions, sub_answers)
```

**Evaluation**: HotpotQA is specifically designed for multi-hop QA. This should show significant improvement there.

---

### 4. Query Expansion (Classic IR Technique)

**Problem**: Vocabulary mismatch between query and documents.

**Example**:
```
Query: "heart attack symptoms"
Document: "Myocardial infarction presents with chest pain..."

Without expansion: low similarity (different words)
With expansion: "heart attack myocardial infarction symptoms signs chest pain" → better match
```

**Implementation Options**:

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

**Option B: WordNet/Embedding-based expansion** (no LLM cost)
```python
def expand_with_similar_terms(query: str, embedder, top_k: int = 3) -> str:
    """Add semantically similar terms using embedding space."""
    words = query.split()
    expansions = []
    
    for word in words:
        # Find similar words in vocabulary
        similar = embedder.most_similar(word, topn=top_k)
        expansions.extend([w for w, score in similar if score > 0.7])
    
    return query + " " + " ".join(set(expansions))
```

**When to Use**: Especially useful for technical domains (medical, legal) where terminology varies.

---

## Retrieval Improvements

### 5. SPLADE (Learned Sparse Retrieval)

**Problem**: BM25/TF-IDF use hand-crafted term weights. They can't learn domain-specific importance.

**Solution**: SPLADE learns sparse representations that are:
- More semantic than BM25 (learns term importance)
- More interpretable than dense (you can see which terms matched)
- Still fast (sparse operations)

**Comparison**:
```
Query: "What causes global warming?"

BM25 weights:     global=1.2, warming=1.5, causes=0.8, what=0.1
SPLADE weights:   global=0.8, warming=1.2, climate=0.9, greenhouse=0.7, CO2=0.5
                  (note: SPLADE adds semantically related terms!)
```

**Implementation**:
```python
# In indexes/splade.py
from transformers import AutoModelForMaskedLM, AutoTokenizer

class SPLADEIndex(Index):
    """Learned sparse retrieval using SPLADE."""
    
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    def encode(self, text: str) -> scipy.sparse.csr_matrix:
        """Encode text to sparse vector."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            logits = self.model(**tokens).logits
        
        # SPLADE: log(1 + ReLU(logits)) aggregated over sequence
        weights = torch.log1p(torch.relu(logits)).max(dim=1).values
        
        # Convert to sparse (only keep non-zero entries)
        return self._to_sparse(weights)
```

**Expected Improvement**: 5-15% better than BM25 on semantic queries, similar speed.

---

### 6. ColBERT (Late Interaction)

**Problem**: Single-vector embeddings lose fine-grained token information.

**Solution**: ColBERT keeps token-level embeddings and uses MaxSim scoring.

```
Standard Dense:
  Query → [single 768-dim vector]
  Doc   → [single 768-dim vector]
  Score = cosine(query_vec, doc_vec)

ColBERT:
  Query "what is RAG" → [[vec_what], [vec_is], [vec_RAG]]  # 3 vectors
  Doc "RAG means..."  → [[vec_RAG], [vec_means], ...]      # N vectors
  Score = Σ max_j(cosine(q_i, d_j)) for each query token i
```

**Why It's Better**:
- Captures exact term matches (like "RAG" → "RAG")
- Still semantic (embeddings, not just strings)
- Much better for long documents

**Implementation**: Use existing ColBERTv2 libraries:
```python
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig

config = ColBERTConfig(
    doc_maxlen=512,
    query_maxlen=32,
    nbits=2,  # Compression for efficiency
)

indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
indexer.index(name="wikipedia_colbert", collection=documents)

searcher = Searcher(index="wikipedia_colbert")
results = searcher.search("what is retrieval augmented generation", k=10)
```

**Trade-offs**:
- Index size: ~5x larger than dense
- Indexing time: ~3x slower
- Search time: Similar (with PLAID optimization)
- Quality: Significant improvement, especially for long docs

---

### 7. Semantic Chunking

**Problem**: Fixed-size chunks split in arbitrary places, breaking semantic coherence.

**Example of Bad Chunking** (fixed 512 chars):
```
Chunk 1: "...World War II began in 1939 when Germany invaded Poland. The"
Chunk 2: "war lasted until 1945 and resulted in millions of casualties..."

The sentence about the war's duration is split!
```

**Semantic Chunking**: Split where topics change.

```python
# In corpus/semantic_chunking.py
class SemanticChunker:
    """Split documents at semantic boundaries."""
    
    def __init__(self, embedder, similarity_threshold: float = 0.5):
        self.embedder = embedder
        self.threshold = similarity_threshold
    
    def chunk(self, text: str, min_size: int = 100, max_size: int = 1000) -> list[str]:
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Embed each sentence
        embeddings = self.embedder.encode(sentences)
        
        # Find breakpoints where similarity drops
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(embeddings[i-1], embeddings[i])
            
            current_size = sum(len(s) for s in current_chunk)
            
            if similarity < self.threshold and current_size >= min_size:
                # Semantic boundary detected
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            elif current_size >= max_size:
                # Force split at max size
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
```

**Alternative**: LlamaIndex's `SemanticSplitterNodeParser` does this out of the box.

---

## Post-Retrieval Processing

### 8. Context Compression

**Problem**: Sending 5 full passages to LLM is:
- Expensive (more tokens)
- Slow (longer generation time)
- Sometimes worse (irrelevant info distracts)

**Solution**: Compress passages before sending to LLM.

**Option A: Extractive compression** (fast, no LLM)
```python
def extract_relevant_sentences(passage: str, query: str, max_sentences: int = 3) -> str:
    """Keep only sentences most relevant to query."""
    sentences = passage.split(". ")
    query_embedding = embedder.encode(query)
    
    scored = []
    for sent in sentences:
        sent_embedding = embedder.encode(sent)
        score = cosine_similarity(query_embedding, sent_embedding)
        scored.append((score, sent))
    
    # Keep top sentences in original order
    top_sentences = sorted(scored, reverse=True)[:max_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: passage.find(x[1]))
    
    return ". ".join(s for _, s in top_sentences)
```

**Option B: LLMLingua** (token-level compression)
```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)

compressed = compressor.compress_prompt(
    context=passages,
    question=query,
    rate=0.5,  # Keep 50% of tokens
)
```

**Expected Impact**: 
- 40-60% token reduction
- 0-5% quality loss (sometimes improvement!)
- Significant cost savings for API-based LLMs

---

### 9. Passage Attribution

**Problem**: We can't tell which passage contributed to the answer. Important for:
- Debugging (which retrieval helped?)
- Explainability (cite your sources)
- Hallucination detection (is answer grounded?)

**Implementation**:
```python
ATTRIBUTION_PROMPT = """Based on the passages below, answer the question.
After your answer, cite which passage(s) you used with [1], [2], etc.

{passages}

Question: {question}

Answer (with citations):"""

# Parse response
def extract_citations(response: str) -> tuple[str, list[int]]:
    import re
    citations = [int(c) for c in re.findall(r'\[(\d+)\]', response)]
    answer = re.sub(r'\[\d+\]', '', response).strip()
    return answer, citations
```

**Evaluation**: Check if cited passages actually contain supporting evidence.

---

## Evaluation Improvements

### 10. RAGAS-Style Pipeline Evaluation

**Problem**: F1/EM only evaluate final answers. We need to evaluate the full RAG pipeline.

**RAGAS Metrics**:

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Faithfulness** | Is answer grounded in context? | LLM checks if each claim is supported |
| **Answer Relevance** | Does answer address the question? | Generate questions from answer, compare to original |
| **Context Precision** | Are retrieved passages relevant? | LLM rates each passage |
| **Context Recall** | Is gold answer in context? | Check if gold answer appears in passages |

**Implementation**:
```python
# In metrics/ragas.py
class FaithfulnessMetric(Metric):
    """Check if answer is grounded in retrieved context."""
    
    PROMPT = """Given the context and answer, determine if the answer is fully supported by the context.

Context:
{context}

Answer: {answer}

Is every claim in the answer supported by the context?
Reply with just "yes" or "no"."""
    
    def compute_single(self, answer: str, context: str) -> float:
        response = self.judge_model.generate(
            self.PROMPT.format(context=context, answer=answer)
        )
        return 1.0 if "yes" in response.lower() else 0.0


class ContextRecall(Metric):
    """Check if gold answer is present in retrieved context."""
    
    def compute_single(self, context: str, expected: list[str]) -> float:
        context_lower = context.lower()
        for gold in expected:
            if gold.lower() in context_lower:
                return 1.0
        return 0.0
```

**Why It Matters**:
```
Scenario: F1=0.0, Context_Recall=1.0, Faithfulness=0.5

Interpretation: Gold answer was in the context, but LLM:
- Partially used the context (faithfulness=0.5)
- Gave wrong answer anyway

Fix: Better prompting or passage ordering, not better retrieval.
```

---

## Advanced RAG Patterns

### 11. Corrective RAG (CRAG)

**Problem**: Sometimes retrieval fails. We should detect this and recover.

**CRAG Pattern**:
1. Retrieve documents
2. Evaluate retrieval quality (are docs relevant?)
3. If low quality: fall back to web search or decompose query
4. Generate answer from best sources

```python
class CorrectiveRAGAgent(Agent):
    """RAG with retrieval quality evaluation and fallback."""
    
    def __init__(self, retriever, web_search, quality_threshold: float = 0.5):
        self.retriever = retriever
        self.web_search = web_search
        self.quality_threshold = quality_threshold
    
    def answer(self, question: str) -> RAGResponse:
        # Step 1: Retrieve
        docs = self.retriever.retrieve(question)
        
        # Step 2: Evaluate retrieval quality
        quality = self._evaluate_quality(question, docs)
        
        if quality < self.quality_threshold:
            # Step 3a: Fallback to web search
            web_docs = self.web_search.search(question)
            docs = self._merge_sources(docs, web_docs)
        
        # Step 4: Generate answer
        return self._generate(question, docs)
    
    def _evaluate_quality(self, question: str, docs: list) -> float:
        """Use LLM to evaluate if docs are relevant."""
        prompt = f"""Rate how relevant these documents are to the question.
Question: {question}
Documents: {self._format_docs(docs)}
Score (0-1):"""
        
        response = self.model.generate(prompt)
        return float(response.strip())
```

**Note**: This is similar to Self-RAG but with explicit quality evaluation and web fallback.

---

### 12. Proposition-Based Indexing

**Problem**: Chunks contain multiple facts, making retrieval imprecise.

**Solution**: Index atomic propositions (single facts).

```
Original chunk:
"Barack Obama was born in Hawaii in 1961. He served as the 44th President 
of the United States from 2009 to 2017. He was the first African American 
to hold the office."

Propositions:
1. "Barack Obama was born in Hawaii."
2. "Barack Obama was born in 1961."
3. "Barack Obama served as the 44th President of the United States."
4. "Barack Obama served as President from 2009 to 2017."
5. "Barack Obama was the first African American President."
```

**Implementation**:
```python
PROPOSITION_PROMPT = """Break this text into atomic propositions (single facts).
Each proposition should be self-contained and verifiable.

Text: {text}

Propositions (one per line):"""

def extract_propositions(text: str, model) -> list[str]:
    response = model.generate(PROPOSITION_PROMPT.format(text=text))
    return [p.strip() for p in response.split("\n") if p.strip()]
```

**Trade-offs**:
- Much more precise retrieval
- 5-10x more items to index
- Requires linking propositions back to source documents

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| Add retrieval metrics (Recall@k, MRR) | Low | High | P0 |
| Implement passage reordering | Low | Medium | P0 |
| Add Context Recall metric | Low | High | P0 |

### Phase 2: Query Processing (2-4 weeks)

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| Query Decomposition agent | Medium | High (HotpotQA) | P1 |
| Query Expansion transformer | Low | Medium | P2 |
| RAGAS Faithfulness metric | Medium | High | P1 |

### Phase 3: Advanced Retrieval (4-8 weeks)

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| SPLADE integration | Medium | Medium-High | P2 |
| Semantic chunking | Medium | Medium | P2 |
| Context compression | Medium | Medium | P3 |

### Phase 4: Research Extensions (8+ weeks)

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| ColBERT integration | High | High | P3 |
| Corrective RAG | High | Medium | P3 |
| Proposition indexing | High | Medium | P4 |

---

## References

1. **Lost in the Middle**: Liu et al., 2023 - "Lost in the Middle: How Language Models Use Long Contexts"
2. **SPLADE**: Formal et al., 2021 - "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
3. **ColBERT**: Khattab & Zaharia, 2020 - "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
4. **HyDE**: Gao et al., 2022 - "Precise Zero-Shot Dense Retrieval without Relevance Labels"
5. **Self-RAG**: Asai et al., 2023 - "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
6. **CRAG**: Yan et al., 2024 - "Corrective Retrieval Augmented Generation"
7. **RAGAS**: Es et al., 2023 - "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
8. **LLMLingua**: Jiang et al., 2023 - "LLMLingua: Compressing Prompts for Accelerated Inference"

---

## Appendix: Method Comparison Matrix

| Method | Query Processing | Retrieval | Post-Processing | Multi-Hop |
|--------|-----------------|-----------|-----------------|-----------|
| **HyDE** | ✅ Transform | - | - | ❌ |
| **MultiQuery** | ✅ Paraphrase | - | - | ❌ |
| **Iterative RAG** | ✅ Refine | - | - | ⚠️ Partial |
| **Query Decomposition** | ✅ Decompose | - | - | ✅ |
| **Reranking** | - | ✅ Two-stage | - | - |
| **SPLADE** | - | ✅ Learned sparse | - | - |
| **ColBERT** | - | ✅ Late interaction | - | - |
| **Hybrid (RRF)** | - | ✅ Fusion | - | - |
| **Context Compression** | - | - | ✅ Compress | - |
| **Passage Reordering** | - | - | ✅ Reorder | - |
| **Self-RAG** | - | ✅ Adaptive | ✅ Critique | ❌ |
| **CRAG** | ✅ Fallback | ✅ Quality check | - | ❌ |

This matrix helps identify which techniques are complementary and can be combined.
