"""Fixed RAG agent - Baseline 2: Standard RAG with fixed parameters."""

from typing import TYPE_CHECKING, Any, Optional

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.core.schemas import AgentType, PipelineLog, RAGResponseMeta, RetrievedDoc
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Retriever
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.utils.artifacts import get_artifact_manager
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder

if TYPE_CHECKING:
    from ragicamp.rag.pipeline import RAGPipeline
    from ragicamp.rag.query_transform.base import QueryTransformer
    from ragicamp.rag.rerankers.base import Reranker


class FixedRAGAgent(RAGAgent):
    """Baseline RAG agent with fixed retrieval parameters.

    This agent implements the standard RAG pipeline:
    1. Retrieve top-k documents (optionally with query transformation)
    2. Optionally rerank retrieved documents
    3. Format context with retrieved documents
    4. Generate answer using LLM with context

    Supports advanced RAG features:
    - Query transformation (HyDE, multi-query)
    - Reranking (cross-encoder)
    - Various retriever types (dense, hybrid, hierarchical)
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        top_k: int = 5,
        prompt_builder: Optional[PromptBuilder] = None,
        # Advanced RAG pipeline options
        query_transformer: Optional["QueryTransformer"] = None,
        reranker: Optional["Reranker"] = None,
        top_k_retrieve: Optional[int] = None,
        # Legacy parameters for backwards compatibility
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
        context_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the fixed RAG agent.

        Args:
            name: Agent identifier
            model: The language model to use
            retriever: The retriever for finding relevant documents
            top_k: Number of documents to use for generation (after reranking)
            prompt_builder: PromptBuilder instance for building prompts.
                          If not provided, creates default.
            query_transformer: Optional query transformer (HyDE, multi-query)
            reranker: Optional reranker (cross-encoder)
            top_k_retrieve: Number of docs to retrieve before reranking.
                           If None, uses top_k * 4 when reranker is present.
            system_prompt: (Legacy) System prompt for the LLM
            context_template: (Legacy) Template with {context} and {query} placeholders
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.top_k = top_k

        # Advanced pipeline components
        self.query_transformer = query_transformer
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve or (top_k * 4 if reranker else top_k)

        # Build pipeline if advanced features are used
        self._pipeline: "Optional[RAGPipeline]" = None  # noqa: UP037
        if query_transformer is not None or reranker is not None:
            from ragicamp.rag.pipeline import RAGPipeline

            self._pipeline = RAGPipeline(
                retriever=retriever,
                query_transformer=query_transformer,
                reranker=reranker,
                top_k_retrieve=self.top_k_retrieve,
                top_k_final=top_k,
            )

        # Use provided prompt_builder or create from legacy params
        if prompt_builder is not None:
            self.prompt_builder = prompt_builder
        else:
            from ragicamp.utils.prompts import PromptConfig

            self.prompt_builder = PromptBuilder(PromptConfig(style=system_prompt))

        # Legacy template support (for backwards compat with study.py)
        self._legacy_template = context_template
        self._system_prompt = system_prompt

    def _build_prompt(self, query: str, context_text: str) -> str:
        """Build prompt from template with context and query."""
        # If legacy template provided, use it directly
        if self._legacy_template:
            formatted = self._legacy_template.format(context=context_text, query=query)
            return f"{self._system_prompt}\n\n{formatted}"

        # Use prompt builder
        return self.prompt_builder.build_rag(query, context_text)

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using fixed RAG pipeline.

        Args:
            query: The input question
            **kwargs: Additional generation parameters

        Returns:
            RAGResponse with the answer and retrieved context
        """
        # Retrieve documents (use pipeline if available, otherwise direct retrieval)
        pipeline_log: Optional[PipelineLog] = None

        if self._pipeline is not None:
            # Use pipeline with full logging
            result = self._pipeline.retrieve_with_log(query)
            retrieved_docs = result.documents
            pipeline_log = result.pipeline_log
        else:
            retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)

        # Format context using utility
        context_text = ContextFormatter.format_numbered(retrieved_docs)

        # Build prompt
        prompt = self._build_prompt(query, context_text)

        # Create context with retrieved docs info
        context = RAGContext(
            query=query,
            retrieved_docs=retrieved_docs,
            metadata={
                "top_k": self.top_k,
                "context_text": context_text,
            },
        )

        # Generate answer
        answer = self.model.generate(prompt, **kwargs)

        # Build structured retrieved docs for typed metadata
        # Include both retrieval and rerank scores if available
        retrieved_structured = [
            RetrievedDoc(
                rank=i + 1,
                content=doc.text,
                score=getattr(doc, "score", None),
                retrieval_score=getattr(doc, "_retrieval_score", None),
                rerank_score=getattr(doc, "_rerank_score", None),
                retrieval_rank=getattr(doc, "_retrieval_rank", None),
                doc_id=getattr(doc, "id", None),
            )
            for i, doc in enumerate(retrieved_docs)
        ]

        # Return response with typed metadata including pipeline log
        return RAGResponse(
            answer=answer,
            context=context,
            prompt=prompt,
            metadata=RAGResponseMeta(
                agent_type=AgentType.FIXED_RAG,
                num_docs_used=len(retrieved_docs),
                retrieved_docs=retrieved_structured,
                pipeline_log=pipeline_log,
            ),
        )

    def batch_answer(self, queries: list[str], **kwargs: Any) -> list[RAGResponse]:
        """Generate answers for multiple queries using batch processing.

        Optimized for speed:
        - Batch query encoding (10-50x faster than sequential)
        - Batch FAISS search (5-10x faster)
        - Batch LLM generation

        Note: Batch mode currently doesn't capture per-query pipeline logs
        for performance. Use single-query mode if detailed logging is needed.

        Args:
            queries: List of input questions
            **kwargs: Additional generation parameters

        Returns:
            List of RAGResponse objects, one per query
        """
        # Retrieve documents for all queries using batch retrieval if available
        # Note: batch_retrieve doesn't return per-query pipeline logs yet
        if self._pipeline is not None:
            all_docs = self._pipeline.batch_retrieve(queries)
        elif hasattr(self.retriever, "batch_retrieve"):
            # Use optimized batch retrieval
            all_docs = self.retriever.batch_retrieve(queries, top_k=self.top_k)
        else:
            # Fallback to sequential
            all_docs = [self.retriever.retrieve(q, top_k=self.top_k) for q in queries]

        # Format contexts and build prompts
        prompts = []
        contexts = []
        context_texts = []
        for query, docs in zip(queries, all_docs):
            context_text = ContextFormatter.format_numbered(docs)
            context_texts.append(context_text)
            prompt = self._build_prompt(query, context_text)
            prompts.append(prompt)
            contexts.append(
                RAGContext(
                    query=query,
                    retrieved_docs=docs,
                    metadata={"top_k": self.top_k, "context_text": context_text},
                )
            )

        # Batch generate (single forward pass for all prompts!)
        answers = self.model.generate(prompts, **kwargs)

        # Create responses with prompts for debugging/analysis
        responses = []
        for _query, prompt, answer, context, docs, _ctx_text in zip(
            queries, prompts, answers, contexts, all_docs, context_texts
        ):
            # Build structured retrieved docs for typed metadata
            # Include both retrieval and rerank scores if available
            retrieved_structured = [
                RetrievedDoc(
                    rank=i + 1,
                    content=doc.text,
                    score=getattr(doc, "score", None),
                    retrieval_score=getattr(doc, "_retrieval_score", None),
                    rerank_score=getattr(doc, "_rerank_score", None),
                    retrieval_rank=getattr(doc, "_retrieval_rank", None),
                    doc_id=getattr(doc, "id", None),
                )
                for i, doc in enumerate(docs)
            ]

            response = RAGResponse(
                answer=answer,
                context=context,
                prompt=prompt,
                metadata=RAGResponseMeta(
                    agent_type=AgentType.FIXED_RAG,
                    num_docs_used=len(docs),
                    batch_processing=True,
                    retrieved_docs=retrieved_structured,
                    # pipeline_log not available in batch mode for performance
                ),
            )
            responses.append(response)

        return responses

    def save(self, artifact_name: str, retriever_artifact_name: str) -> str:
        """Save agent configuration.

        Args:
            artifact_name: Name for this agent artifact (e.g., 'fixed_rag_wikipedia_v1')
            retriever_artifact_name: Name of the retriever artifact this agent uses

        Returns:
            Path where the artifact was saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_agent_path(artifact_name)

        # Save agent config
        config = {
            "agent_type": "fixed_rag",
            "name": self.name,
            "top_k": self.top_k,
            "retriever_artifact": retriever_artifact_name,
            "system_prompt": self._system_prompt,
        }
        manager.save_json(config, artifact_path / "config.json")

        print(f"✓ Saved agent to: {artifact_path}")
        return str(artifact_path)

    @classmethod
    def load(
        cls, artifact_name: str, model: LanguageModel, load_retriever: bool = True
    ) -> "FixedRAGAgent":
        """Load a previously saved agent.

        Args:
            artifact_name: Name of the agent artifact to load
            model: Language model to use (not saved, must provide)
            load_retriever: Whether to load the associated retriever

        Returns:
            Loaded FixedRAGAgent instance
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_agent_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Load retriever if requested
        retriever = None
        if load_retriever:
            retriever_name = config["retriever_artifact"]
            retriever = DenseRetriever.load_index(retriever_name)

        # Create agent
        agent = cls(
            name=config["name"],
            model=model,
            retriever=retriever,
            top_k=config["top_k"],
            system_prompt=config["system_prompt"],
        )

        print(f"✓ Loaded agent from: {artifact_path}")
        return agent
