"""Fixed RAG Agent - Standard RAG with batch-optimized resource management.

This agent implements the classic RAG pipeline with GPU-optimal execution:
1. Batch retrieve all documents (embedder uses full GPU)
2. Unload embedder, free GPU memory
3. Batch generate all answers (generator uses full GPU)

Supports optional enhancements:
- Query transformation (HyDE, multi-query)
- Reranking (cross-encoder)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tqdm import tqdm

from ragicamp.agents.base import Agent, AgentResult, Query, Step, StepTimer
from ragicamp.core.logging import get_logger
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder
from ragicamp.utils.resource_manager import ResourceManager

if TYPE_CHECKING:
    from ragicamp.rag.pipeline import RAGPipeline
    from ragicamp.rag.query_transform.base import QueryTransformer
    from ragicamp.rag.rerankers.base import Reranker

logger = get_logger(__name__)


class FixedRAGAgent(Agent):
    """Standard RAG agent with batch-optimized GPU usage.
    
    Execution strategy (for simple RAG without query transformers):
    1. Phase: RETRIEVE - batch encode all queries, search index
       - Embedder loaded with full GPU
    2. Phase: UNLOAD - free embedder from GPU
    3. Phase: GENERATE - batch generate all answers
       - Generator loaded with full GPU
    
    For RAG with query transformers (HyDE, etc.):
    - Both models loaded with reduced GPU fractions
    - Per-query processing (generation influences retrieval)
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        top_k: int = 5,
        prompt_builder: PromptBuilder | None = None,
        query_transformer: "QueryTransformer | None" = None,
        reranker: "Reranker | None" = None,
        top_k_retrieve: int | None = None,
        **config,
    ):
        super().__init__(name, **config)
        
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.query_transformer = query_transformer
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve or (top_k * 4 if reranker else top_k)
        
        # Prompt builder
        if prompt_builder:
            self.prompt_builder = prompt_builder
        else:
            from ragicamp.utils.prompts import PromptConfig
            self.prompt_builder = PromptBuilder(PromptConfig())
        
        # Build pipeline if advanced features used
        self._pipeline: "RAGPipeline | None" = None
        if query_transformer or reranker:
            from ragicamp.rag.pipeline import RAGPipeline
            self._pipeline = RAGPipeline(
                retriever=retriever,
                query_transformer=query_transformer,
                reranker=reranker,
                top_k_retrieve=self.top_k_retrieve,
                top_k_final=top_k,
            )
    
    @property
    def uses_interleaved_pattern(self) -> bool:
        """Check if this agent needs both models loaded simultaneously."""
        # Query transformers (HyDE, MultiQuery) generate before retrieval
        return self.query_transformer is not None

    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process all queries with GPU-optimal batching.
        
        For simple RAG: batch_retrieve → unload_embedder → batch_generate
        For HyDE/MultiQuery: per-query interleaved (both models loaded)
        """
        # Load checkpoint if resuming
        results, completed_idx = [], set()
        if checkpoint_path:
            results, completed_idx = self._load_checkpoint(checkpoint_path)
        
        pending = [q for q in queries if q.idx not in completed_idx]
        
        if not pending:
            logger.info("All queries already completed")
            return results
        
        logger.info("Processing %d queries (strategy: %s)", 
                   len(pending),
                   "interleaved" if self.uses_interleaved_pattern else "batch")
        
        if self.uses_interleaved_pattern:
            # Both models needed - process per-query
            new_results = self._run_interleaved(pending, show_progress)
        else:
            # Optimal: batch retrieve → unload → batch generate
            new_results = self._run_batch_optimized(pending, show_progress)
        
        # Stream results and checkpoint
        for result in new_results:
            results.append(result)
            if on_result:
                on_result(result)
            if checkpoint_path and len(results) % 10 == 0:
                self._save_checkpoint(results, checkpoint_path)
        
        if checkpoint_path:
            self._save_checkpoint(results, checkpoint_path)
        
        return results

    def _run_batch_optimized(
        self, 
        queries: list[Query], 
        show_progress: bool,
    ) -> list[AgentResult]:
        """Batch-optimized execution: retrieve_all → unload → generate_all."""
        
        # Phase 1: Batch retrieve (embedder gets full GPU)
        logger.info("Phase 1: Batch retrieval (%d queries)", len(queries))
        retrievals = self._batch_retrieve(queries, show_progress)
        
        # Phase 2: Unload embedder
        logger.info("Phase 2: Unloading embedder")
        self._unload_embedder()
        
        # Phase 3: Batch generate (generator gets full GPU)  
        logger.info("Phase 3: Batch generation (%d queries)", len(queries))
        results = self._batch_generate(queries, retrievals, show_progress)
        
        return results

    def _run_interleaved(
        self,
        queries: list[Query],
        show_progress: bool,
    ) -> list[AgentResult]:
        """Per-query execution for patterns that need both models."""
        results = []
        
        iterator = tqdm(queries, desc="Processing") if show_progress else queries
        for query in iterator:
            result = self._process_single(query)
            results.append(result)
        
        return results

    def _batch_retrieve(
        self, 
        queries: list[Query],
        show_progress: bool,
    ) -> dict[int, tuple[list[Document], list[Step]]]:
        """Retrieve documents for all queries, return with steps."""
        retrievals: dict[int, tuple[list[Document], list[Step]]] = {}
        
        iterator = tqdm(queries, desc="Retrieving") if show_progress else queries
        for query in iterator:
            steps: list[Step] = []
            
            with StepTimer("retrieve", model=self._get_embedder_name()) as step:
                step.input = query.text
                if self._pipeline:
                    result = self._pipeline.retrieve_with_log(query.text)
                    docs = result.documents
                    step.metadata["pipeline_log"] = result.pipeline_log
                else:
                    docs = self.retriever.retrieve(query.text, top_k=self.top_k)
                step.output = [{"id": d.id, "text": d.text[:100]} for d in docs]
            
            steps.append(step)
            retrievals[query.idx] = (docs, steps)
        
        return retrievals

    def _batch_generate(
        self,
        queries: list[Query],
        retrievals: dict[int, tuple[list[Document], list[Step]]],
        show_progress: bool,
    ) -> list[AgentResult]:
        """Generate answers for all queries using cached retrievals."""
        results: list[AgentResult] = []
        
        iterator = tqdm(queries, desc="Generating") if show_progress else queries
        for query in iterator:
            docs, retrieve_steps = retrievals[query.idx]
            
            # Format context
            context_text = ContextFormatter.format_numbered(docs)
            prompt = self.prompt_builder.build_rag(query.text, context_text)
            
            # Generate
            with StepTimer("generate", model=self._get_generator_name()) as step:
                step.input = {"query": query.text, "context_length": len(context_text)}
                answer = self.model.generate(prompt)
                step.output = answer
            
            # Build result with all steps
            all_steps = retrieve_steps + [step]
            
            result = AgentResult(
                query=query,
                answer=answer,
                steps=all_steps,
                prompt=prompt,
                metadata={
                    "num_docs": len(docs),
                    "top_k": self.top_k,
                },
            )
            results.append(result)
        
        return results

    def _process_single(self, query: Query) -> AgentResult:
        """Process a single query (for interleaved patterns)."""
        steps: list[Step] = []
        
        # Retrieve
        with StepTimer("retrieve", model=self._get_embedder_name()) as step:
            step.input = query.text
            if self._pipeline:
                result = self._pipeline.retrieve_with_log(query.text)
                docs = result.documents
            else:
                docs = self.retriever.retrieve(query.text, top_k=self.top_k)
            step.output = [{"id": d.id, "text": d.text[:100]} for d in docs]
        steps.append(step)
        
        # Format and generate
        context_text = ContextFormatter.format_numbered(docs)
        prompt = self.prompt_builder.build_rag(query.text, context_text)
        
        with StepTimer("generate", model=self._get_generator_name()) as step:
            step.input = {"query": query.text, "context_length": len(context_text)}
            answer = self.model.generate(prompt)
            step.output = answer
        steps.append(step)
        
        return AgentResult(
            query=query,
            answer=answer,
            steps=steps,
            prompt=prompt,
            metadata={"num_docs": len(docs), "top_k": self.top_k},
        )

    def _unload_embedder(self) -> None:
        """Unload embedder to free GPU for generator."""
        if hasattr(self.retriever, 'index') and self.retriever.index:
            index = self.retriever.index
            if hasattr(index, '_encoder') and index._encoder:
                if hasattr(index._encoder, 'unload'):
                    index._encoder.unload()
                index._encoder = None
        
        ResourceManager.clear_gpu_memory()
        logger.info("Embedder unloaded, GPU memory cleared")

    def _get_embedder_name(self) -> str | None:
        """Get embedding model name for logging."""
        if hasattr(self.retriever, 'index') and self.retriever.index:
            return getattr(self.retriever.index, 'embedding_model_name', None)
        return None

    def _get_generator_name(self) -> str | None:
        """Get generator model name for logging."""
        return getattr(self.model, 'model_name', None)
