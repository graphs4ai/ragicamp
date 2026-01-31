# RAGiCamp Improvement Plan

Based on analysis session on 2026-01-31.

---

## Part 1: Architecture & Design Patterns

Before implementing changes, understand the existing patterns to maintain consistency.

### Current Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLI Layer (cli/)                                                        │
│  Entry points: run, health, resume, metrics, compare                     │
│  Responsibility: Parse args, load config, delegate to execution          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│  Specification Layer (spec/)                                             │
│  ExperimentSpec, build_specs(), naming conventions                       │
│  Responsibility: Transform YAML config → immutable experiment specs      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│  Execution Layer (execution/)                                            │
│  runner.py, executor.py, phases/                                         │
│  Responsibility: Orchestrate experiment lifecycle, handle failures       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│  Agent Layer (agents/)                                                   │
│  RAGAgent base, DirectLLMAgent, FixedRAGAgent                           │
│  Responsibility: Answer questions using model + optional retrieval       │
└───────────────┬─────────────────────────────────┬───────────────────────┘
                │                                 │
┌───────────────┴───────────────┐ ┌───────────────┴───────────────────────┐
│  Model Layer (models/)         │ │  RAG Pipeline Layer (rag/)            │
│  LanguageModel base            │ │  pipeline.py, query_transform/,       │
│  HuggingFace, OpenAI, vLLM     │ │  rerankers/                           │
│  Responsibility: Text gen      │ │  Responsibility: Retrieval pipeline   │
└────────────────────────────────┘ └───────────────┬───────────────────────┘
                                                   │
                                   ┌───────────────┴───────────────────────┐
                                   │  Retriever Layer (retrievers/)         │
                                   │  Retriever base, Dense, Sparse,        │
                                   │  Hybrid, Hierarchical                  │
                                   │  Responsibility: Document search       │
                                   └───────────────┬───────────────────────┘
                                                   │
                                   ┌───────────────┴───────────────────────┐
                                   │  Index Layer (indexes/)                │
                                   │  EmbeddingIndex, HierarchicalIndex     │
                                   │  Responsibility: Vector storage/search │
                                   └───────────────────────────────────────┘
```

### Design Patterns in Use

#### 1. Abstract Base Class + Factory Pattern
```python
# Base class defines interface
class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Document]: ...

# Concrete implementations
class DenseRetriever(Retriever): ...
class HybridRetriever(Retriever): ...

# Factory creates from config
retriever = ComponentFactory.create_retriever(config)
```

**When to use**: Adding new retrievers, agents, models, metrics.

#### 2. Strategy Pattern (in RAG Pipeline)
```python
# Strategies are interchangeable
class QueryTransformer(ABC):
    @abstractmethod
    def transform(self, query: str) -> List[str]: ...

# Pipeline uses strategies
class RAGPipeline:
    def __init__(self, retriever, query_transformer=None, reranker=None): ...
```

**When to use**: Query transformers, rerankers, chunking strategies.

#### 3. Phased Execution Pattern
```python
# Each phase is a handler
class PhaseHandler(ABC):
    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionContext: ...

# Phases: INIT → GENERATING → GENERATED → COMPUTING_METRICS → COMPLETE
```

**When to use**: Adding new experiment phases, modifying execution flow.

#### 4. Immutable Specification Pattern
```python
@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable - configuration doesn't change during execution."""
    name: str
    model: str
    ...
```

**When to use**: Any configuration object passed through the pipeline.

### Design Principles to Follow

#### DO:
1. **Single Responsibility**: Each class does one thing well
   - `DenseRetriever` retrieves, doesn't build indexes
   - `IndexBuilder` builds, doesn't retrieve
   
2. **Dependency Injection**: Pass dependencies, don't create them
   ```python
   # Good: accept model as parameter
   class FixedRAGAgent:
       def __init__(self, model: LanguageModel, retriever: Retriever): ...
   
   # Bad: create model internally
   class FixedRAGAgent:
       def __init__(self):
           self.model = HuggingFaceModel(...)  # Tight coupling
   ```

3. **Interface Segregation**: Small, focused interfaces
   ```python
   # Good: focused interface
   class Retriever(ABC):
       def retrieve(self, query, top_k) -> List[Document]: ...
   
   # Bad: bloated interface
   class Retriever(ABC):
       def retrieve(...): ...
       def index(...): ...
       def save(...): ...
       def load(...): ...
       def visualize(...): ...  # Doesn't belong here
   ```

4. **Composition over Inheritance**: Combine simple components
   ```python
   # Good: compose behaviors
   class RAGPipeline:
       def __init__(self, retriever, transformer=None, reranker=None):
           self.retriever = retriever
           self.transformer = transformer  # Optional strategy
           self.reranker = reranker        # Optional strategy
   ```

#### DON'T:
1. **Don't add optional parameters indefinitely**
   ```python
   # Bad: parameter explosion
   def retrieve(self, query, top_k, use_rerank=False, rerank_model=None,
                summarize=False, max_tokens=None, iterative=False, ...):
   ```
   
   Instead, use composition or configuration objects.

2. **Don't mix concerns across layers**
   - Retrievers shouldn't know about prompts
   - Agents shouldn't know about experiment phases
   - Indexes shouldn't know about metrics

3. **Don't over-abstract prematurely**
   - If there's only one implementation, a simple class is fine
   - Add abstraction when you need the second implementation

---

## Part 2: Summary of Identified Gaps

### Already Fixed (This Session)
- [x] `HybridRetriever.load()` now loads pre-built sparse index
- [x] `HybridRetriever.save()` now persists sparse components
- [x] Removed redundant GPU clearing calls

### Remaining Gaps
1. **Configuration**: Grid search only, no singleton experiments
2. **Chunking**: Strategy not exposed in config
3. **Pipeline**: No fetch_k, no context summarization
4. **Strategies**: No iterative retrieval, no Self-RAG

---

## Part 3: Task Plan

### Phase 1: Singleton Experiment Support

**Goal**: Allow targeted hypothesis-driven experiments.

**Architecture fit**: Extends `spec/builder.py` without changing other layers.

#### Task 1.1: Add singleton experiment parsing

**File**: `src/ragicamp/spec/builder.py`

```python
def build_specs(config: Dict[str, Any]) -> List[ExperimentSpec]:
    specs = []
    
    # Existing: grid search specs
    if config.get("direct", {}).get("enabled"):
        specs.extend(_build_direct_specs(...))
    if config.get("rag", {}).get("enabled"):
        specs.extend(_build_rag_specs(...))
    
    # NEW: singleton experiments
    if "experiments" in config.get("rag", {}):
        specs.extend(_build_singleton_specs(config["rag"]["experiments"], ...))
    
    return specs

def _build_singleton_specs(experiments: List[Dict], ...) -> List[ExperimentSpec]:
    """Build specs from explicit experiment definitions."""
    specs = []
    for exp in experiments:
        # Each experiment defines its own config
        specs.append(ExperimentSpec(
            name=exp["name"],
            exp_type="rag",
            model=exp.get("model") or default_model,
            retriever=_resolve_retriever(exp["retriever"]),  # Handle inline
            top_k=exp.get("top_k", 5),
            fetch_k=exp.get("fetch_k"),  # NEW field
            query_transform=exp.get("query_transform"),
            reranker=exp.get("reranker"),
            prompt=exp.get("prompt", "concise"),
            dataset=exp.get("dataset") or default_dataset,
            hypothesis=exp.get("hypothesis"),  # NEW: for documentation
            ...
        ))
    return specs
```

**Changes**:
1. Add `_build_singleton_specs()` function
2. Add `hypothesis` field to `ExperimentSpec` (optional metadata)
3. Add `fetch_k` field to `ExperimentSpec`

**Acceptance criteria**:
- [ ] Can define experiments in `rag.experiments` list
- [ ] Each experiment specifies its own complete config
- [ ] Coexists with grid search definitions

---

#### Task 1.2: Handle inline retriever definitions

**Files**: `spec/builder.py`, `indexes/builder.py`

When singleton defines retriever inline, generate name and ensure index exists:

```python
def _resolve_retriever(retriever_config: Union[str, Dict]) -> str:
    """Resolve retriever config to a retriever name."""
    if isinstance(retriever_config, str):
        return retriever_config  # Already a name
    
    # Generate deterministic name from config
    name = _generate_retriever_name(retriever_config)
    
    # Register for building if not exists
    _pending_retrievers[name] = retriever_config
    
    return name
```

---

### Phase 2: Chunking Strategy Support

**Goal**: Expose chunking strategy in retriever config.

**Architecture fit**: Extends index builders, no agent changes.

#### Task 2.1: Add chunking_strategy parameter

**File**: `src/ragicamp/indexes/builders/embedding_builder.py`

```python
def build_embedding_index(
    corpus: List[Document],
    embedding_model: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: str = "recursive",  # NEW
    ...
) -> EmbeddingIndex:
    # Select chunking strategy
    chunker = _get_chunker(chunking_strategy, chunk_size, chunk_overlap)
    ...
```

**Config format**:
```yaml
retriever:
  type: dense
  chunk_size: 1024
  chunking_strategy: paragraph  # NEW: fixed, sentence, paragraph, recursive
```

---

### Phase 3: Fetch-K Support

**Goal**: Retrieve N documents, rerank to K.

**Architecture fit**: Extends `RAGPipeline`, adds field to `ExperimentSpec`.

#### Task 3.1: Add fetch_k to pipeline

**File**: `src/ragicamp/rag/pipeline.py`

```python
class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        query_transformer: Optional[QueryTransformer] = None,
        reranker: Optional[Reranker] = None,
        fetch_k: Optional[int] = None,  # NEW
    ):
        self.fetch_k = fetch_k

    def batch_retrieve(self, queries: List[str], top_k: int) -> List[List[Document]]:
        # Use fetch_k if specified, otherwise use top_k
        retrieve_k = self.fetch_k or top_k
        
        # Retrieve more documents
        results = self.retriever.batch_retrieve(queries, retrieve_k)
        
        # Rerank and trim to top_k
        if self.reranker:
            results = self.reranker.batch_rerank(queries, results)
            results = [docs[:top_k] for docs in results]
        
        return results
```

---

### Phase 4: New RAG Strategies

**Architecture fit**: New agent classes following existing patterns.

#### Task 4.1: Iterative Retrieval Agent

**File**: NEW `src/ragicamp/agents/iterative_rag.py`

**Pattern**: New agent type, follows `RAGAgent` interface.

```python
class IterativeRAGAgent(RAGAgent):
    """Agent that iteratively refines retrieval.
    
    Flow:
    1. Initial retrieval
    2. LLM evaluates: "Is this context sufficient?"
    3. If not: Generate refined query → Retrieve again
    4. Repeat until max_iterations or sufficient context
    5. Generate final answer
    """
    
    def __init__(
        self,
        model: LanguageModel,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        max_iterations: int = 2,
        evaluation_prompt: Optional[str] = None,
    ):
        self.model = model
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.max_iterations = max_iterations
        self.evaluation_prompt = evaluation_prompt or DEFAULT_EVAL_PROMPT
    
    def answer(self, query: str, top_k: int = 5) -> RAGResponse:
        context_docs = []
        queries_used = [query]
        
        for iteration in range(self.max_iterations):
            # Retrieve with current query
            new_docs = self.retriever.retrieve(queries_used[-1], top_k)
            context_docs = self._merge_docs(context_docs, new_docs)
            
            # Evaluate context quality (skip on last iteration)
            if iteration < self.max_iterations - 1:
                is_sufficient, refined_query = self._evaluate_context(
                    query, context_docs
                )
                if is_sufficient:
                    break
                queries_used.append(refined_query)
        
        # Generate final answer
        context_text = self._format_context(context_docs)
        prompt = self.prompt_builder.build_rag(query, context_text)
        answer = self.model.generate(prompt)
        
        return RAGResponse(
            answer=answer,
            prompt=prompt,
            context=context_docs,
            metadata={"iterations": len(queries_used), "queries": queries_used}
        )
```

**Registration**:
```python
@ComponentFactory.register_agent("iterative_rag")
class IterativeRAGAgent(RAGAgent): ...
```

---

#### Task 4.2: Self-RAG Agent

**File**: NEW `src/ragicamp/agents/self_rag.py`

**Concept**: Model decides whether retrieval helps, inspired by Self-RAG paper.

```python
class SelfRAGAgent(RAGAgent):
    """Agent that decides whether to use retrieval based on query.
    
    Flow:
    1. Classify query: "Does this need retrieval?"
    2. If yes: Retrieve → Generate with context
    3. If no: Generate directly from knowledge
    4. Optionally: Verify answer against retrieved context
    
    This helps when:
    - Model's knowledge is sufficient (no retrieval needed)
    - Retrieved context is misleading (ignore it)
    - Query is better answered from reasoning (not facts)
    """
    
    def __init__(
        self,
        model: LanguageModel,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        retrieval_threshold: float = 0.5,  # Confidence to skip retrieval
        verify_answer: bool = False,       # Check answer against context
    ):
        self.model = model
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.retrieval_threshold = retrieval_threshold
        self.verify_answer = verify_answer
    
    def answer(self, query: str, top_k: int = 5) -> RAGResponse:
        # Step 1: Decide if retrieval is needed
        needs_retrieval, confidence = self._assess_retrieval_need(query)
        
        if needs_retrieval:
            # Step 2a: Standard RAG path
            docs = self.retriever.retrieve(query, top_k)
            context_text = self._format_context(docs)
            prompt = self.prompt_builder.build_rag(query, context_text)
            answer = self.model.generate(prompt)
            
            # Optional: Verify answer is supported
            if self.verify_answer:
                is_supported = self._verify_answer_support(answer, docs)
                if not is_supported:
                    # Fall back to direct answer
                    return self._generate_direct(query)
            
            return RAGResponse(
                answer=answer,
                prompt=prompt,
                context=docs,
                metadata={"used_retrieval": True, "confidence": confidence}
            )
        else:
            # Step 2b: Direct answer path
            return self._generate_direct(query)
    
    def _assess_retrieval_need(self, query: str) -> Tuple[bool, float]:
        """Ask model if retrieval would help."""
        assessment_prompt = f"""Given this question, decide if you need external information to answer it accurately.

Question: {query}

Consider:
- Is this asking for factual information you might not know?
- Is this asking about recent events or specific data?
- Could you answer this from general reasoning?

Respond with: RETRIEVE or DIRECT, followed by confidence 0-1.
Example: RETRIEVE 0.8"""
        
        response = self.model.generate(assessment_prompt)
        needs_retrieval, confidence = self._parse_assessment(response)
        return needs_retrieval, confidence
    
    def _generate_direct(self, query: str) -> RAGResponse:
        """Generate answer without retrieval."""
        prompt = self.prompt_builder.build_direct(query)
        answer = self.model.generate(prompt)
        return RAGResponse(
            answer=answer,
            prompt=prompt,
            context=[],
            metadata={"used_retrieval": False}
        )
```

**Config format**:
```yaml
experiments:
  - name: self_rag_test
    agent_type: self_rag  # NEW: agent type selection
    model: hf:meta-llama/Llama-3.2-3B-Instruct
    retriever: dense_bge
    self_rag:
      retrieval_threshold: 0.6
      verify_answer: true
```

---

#### Task 4.3: Context Summarization (as pipeline component)

**File**: NEW `src/ragicamp/rag/context_processor.py`

**Pattern**: Strategy pattern, composable with pipeline.

```python
class ContextProcessor(ABC):
    """Process retrieved context before passing to LLM."""
    
    @abstractmethod
    def process(
        self, 
        query: str, 
        documents: List[Document],
        model: LanguageModel,
    ) -> str:
        """Process documents into context string."""
        pass


class SummarizingProcessor(ContextProcessor):
    """Summarize context if too long."""
    
    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens
    
    def process(self, query, documents, model) -> str:
        context = ContextFormatter.format_numbered(documents)
        
        if model.count_tokens(context) <= self.max_tokens:
            return context
        
        # Summarize
        summary_prompt = f"""Summarize the following context, keeping information relevant to the question.

Question: {query}

Context:
{context}

Provide a concise summary with key facts:"""
        
        return model.generate(summary_prompt)


class PassthroughProcessor(ContextProcessor):
    """No processing, just format."""
    
    def process(self, query, documents, model) -> str:
        return ContextFormatter.format_numbered(documents)
```

---

### Phase 5: Efficiency Improvements

#### Task 5.1: Cache predictions.json

**File**: `src/ragicamp/execution/phases/`

Load once, share across phases via `ExecutionContext`.

---

#### Task 5.2: Parallelize CPU metrics

**File**: `src/ragicamp/metrics/__init__.py`

Use `ThreadPoolExecutor` for CPU-only metrics (exact_match, f1) while GPU metrics run.

---

## Part 4: Implementation Order

| Order | Phase | Task | Effort | Impact | Dependencies |
|-------|-------|------|--------|--------|--------------|
| 1 | 1.1 | Singleton experiments | Medium | HIGH | None |
| 2 | 1.2 | Inline retriever handling | Medium | HIGH | 1.1 |
| 3 | 2.1 | Chunking strategy | Low | HIGH | None |
| 4 | 3.1 | Fetch-K support | Low | MEDIUM | None |
| 5 | 4.1 | Iterative RAG agent | Medium | MEDIUM | None |
| 6 | 4.2 | Self-RAG agent | Medium | HIGH | None |
| 7 | 4.3 | Context summarization | Low | MEDIUM | None |
| 8 | 5.1 | Cache predictions | Low | LOW | None |
| 9 | 5.2 | Parallel metrics | Low | LOW | None |

---

## Part 5: Maintaining Code Quality

### Before Adding a Feature

1. **Identify the layer**: Where does this belong?
   - New experiment type → Agent layer
   - New retrieval strategy → RAG pipeline layer
   - New config option → Spec layer
   
2. **Check existing patterns**: Is there a similar feature?
   - Copy the pattern, don't invent new ones
   
3. **Define the interface first**: What's the minimal contract?
   ```python
   class NewAgent(RAGAgent):
       def answer(self, query: str, top_k: int = 5) -> RAGResponse: ...
   ```

4. **Keep it simple**: Start with the minimal implementation
   - Add complexity only when needed
   - Prefer configuration over code changes

### Code Review Checklist

- [ ] Follows existing patterns in the codebase
- [ ] No new dependencies on unrelated layers
- [ ] Has docstrings explaining purpose and usage
- [ ] Configuration is validated in spec layer
- [ ] Errors are handled gracefully with helpful messages
- [ ] GPU memory is cleaned up after use
- [ ] Can be tested in isolation

---

## Part 6: Example Config After Implementation

```yaml
# rag_hypotheses.yaml

name: rag_hypotheses
description: "Hypothesis-driven RAG experiments"

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

  # Singleton experiments - each tests one hypothesis
  experiments:
    # Baseline
    - name: baseline
      hypothesis: "Standard dense retrieval"
      model: *model
      retriever: dense_bge_c512
      top_k: 5
      prompt: concise
      dataset: nq

    # Hypothesis: Larger chunks help
    - name: h1_large_chunks
      hypothesis: "Larger chunks provide better context"
      model: *model
      retriever:
        type: dense
        embedding_model: BAAI/bge-large-en-v1.5
        chunk_size: 1024
        chunking_strategy: paragraph
      top_k: 3
      prompt: concise
      dataset: nq

    # Hypothesis: Fetch more, rerank to fewer
    - name: h2_fetch_rerank
      hypothesis: "Overfetch and rerank reduces noise"
      model: *model
      retriever: dense_bge_c512
      top_k: 3
      fetch_k: 15
      reranker: bge
      prompt: concise
      dataset: nq

    # Hypothesis: Self-RAG avoids bad retrieval
    - name: h3_self_rag
      hypothesis: "Let model decide when to use retrieval"
      model: *model
      agent_type: self_rag
      retriever: dense_bge_c512
      top_k: 5
      self_rag:
        retrieval_threshold: 0.6
      prompt: concise
      dataset: nq

    # Hypothesis: Iterative refinement finds better docs
    - name: h4_iterative
      hypothesis: "Refine query after initial retrieval"
      model: *model
      agent_type: iterative_rag
      retriever: dense_bge_c512
      top_k: 5
      iterative:
        max_iterations: 2
      prompt: concise
      dataset: nq

metrics:
  - exact_match
  - f1
  - llm_judge_qa

llm_judge:
  model: openai:gpt-4o-mini

output_dir: outputs/rag_hypotheses
```
