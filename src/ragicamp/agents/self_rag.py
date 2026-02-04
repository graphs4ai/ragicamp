"""Self-RAG Agent - Adaptive retrieval based on query analysis.

Uses model providers with proper lifecycle management:
1. Load generator → assess if retrieval needed → unload
2. If needed: load embedder → retrieve → unload
3. Load generator → generate answer → unload
4. Optionally: load generator → verify answer → unload

Interleaved pattern with clean resource management.
"""

from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ragicamp.agents.base import Agent, AgentResult, Query, Step, StepTimer
from ragicamp.core.logging import get_logger
from ragicamp.indexes.vector_index import VectorIndex, SearchResult, Document
from ragicamp.models.providers import EmbedderProvider, GeneratorProvider
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)


# Prompts for self-RAG decision making
RETRIEVAL_DECISION_PROMPT = """Analyze this question and decide if you need to retrieve external information to answer it accurately.

Question: {query}

Consider:
- Is this asking for factual information that could be outdated or specific?
- Is this a general knowledge question you're confident about?
- Would external sources improve the answer's accuracy?

Respond with a confidence score from 0.0 to 1.0 indicating how confident you are that you can answer WITHOUT retrieval.
- 1.0 = Completely confident, no retrieval needed
- 0.5 = Uncertain, retrieval would help
- 0.0 = Definitely need retrieval

Format: CONFIDENCE: [score]
Then briefly explain your reasoning.

Response:"""

VERIFICATION_PROMPT = """Verify if the following answer is supported by the provided context.

Context:
{context}

Question: {query}

Answer: {answer}

Is this answer:
1. SUPPORTED - The answer is directly supported by information in the context
2. PARTIALLY_SUPPORTED - The answer is somewhat related but adds information not in context
3. NOT_SUPPORTED - The answer contradicts or is unrelated to the context

Respond with one of: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED
Then briefly explain.

Verification:"""


class SelfRAGAgent(Agent):
    """Self-RAG agent with adaptive retrieval decision.

    This agent dynamically decides whether to use retrieval based on
    the query and optionally verifies that answers are grounded in context.

    Flow per query:
    1. Load generator → assess retrieval need → unload
    2. If confidence <= threshold:
       - Load embedder → retrieve → unload
       - Load generator → generate with context → unload
    3. Else:
       - Load generator → generate directly → unload
    4. If verify_answer:
       - Load generator → verify grounding → unload

    Note: Inherently sequential per query due to the decision-making.
    """

    def __init__(
        self,
        name: str,
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: VectorIndex,
        top_k: int = 5,
        retrieval_threshold: float = 0.5,
        verify_answer: bool = False,
        fallback_to_direct: bool = True,
        prompt_builder: PromptBuilder | None = None,
        **config,
    ):
        """Initialize agent with providers.

        Args:
            name: Agent identifier
            embedder_provider: Provides embedder with lazy loading
            generator_provider: Provides generator with lazy loading
            index: Vector index (just data)
            top_k: Number of documents to retrieve
            retrieval_threshold: Confidence threshold for skipping retrieval
            verify_answer: Whether to verify answer is grounded
            fallback_to_direct: If verification fails, fall back to direct
            prompt_builder: For building prompts
        """
        super().__init__(name, **config)

        self.embedder_provider = embedder_provider
        self.generator_provider = generator_provider
        self.index = index
        self.top_k = top_k
        self.retrieval_threshold = retrieval_threshold
        self.verify_answer = verify_answer
        self.fallback_to_direct = fallback_to_direct
        self.prompt_builder = prompt_builder or PromptBuilder(PromptConfig())

    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process queries with adaptive retrieval."""
        # Load checkpoint if resuming
        results, completed_idx = [], set()
        if checkpoint_path:
            results, completed_idx = self._load_checkpoint(checkpoint_path)

        pending = [q for q in queries if q.idx not in completed_idx]

        if not pending:
            logger.info("All queries already completed")
            return results

        logger.info("SelfRAG: Processing %d queries (threshold=%.2f)",
                   len(pending), self.retrieval_threshold)

        iterator = pending
        if show_progress:
            iterator = tqdm(pending, desc="Processing queries")

        for query in iterator:
            result = self._process_single_query(query)
            results.append(result)

            if on_result:
                on_result(result)

            if checkpoint_path:
                self._save_checkpoint(results, checkpoint_path)

        logger.info("Completed %d queries", len(pending))
        return results

    def _process_single_query(self, query: Query) -> AgentResult:
        """Process a single query with adaptive retrieval."""
        steps: list[Step] = []

        # Step 1: Assess retrieval need
        with self.generator_provider.load() as generator:
            with StepTimer("assess_retrieval", model=self.generator_provider.model_name) as step:
                confidence = self._assess_retrieval_need(generator, query.text)
                step.input = {"query": query.text}
                step.output = {"confidence": confidence}
            steps.append(step)

        used_retrieval = confidence <= self.retrieval_threshold
        logger.debug("SelfRAG: confidence=%.2f, threshold=%.2f -> %s",
                    confidence, self.retrieval_threshold,
                    "retrieve" if used_retrieval else "direct")

        retrieved_docs: list[Document] = []
        context_text = ""
        verification_result = None

        if used_retrieval:
            # Step 2a: Retrieve
            with self.embedder_provider.load() as embedder:
                with StepTimer("encode", model=self.embedder_provider.model_name) as step:
                    query_embedding = embedder.batch_encode([query.text])
                    step.input = {"query": query.text}
                    step.output = {"embedding_shape": query_embedding.shape}
                steps.append(step)

            with StepTimer("search") as step:
                search_results = self.index.batch_search(query_embedding, top_k=self.top_k)[0]
                retrieved_docs = [r.document for r in search_results]
                step.input = {"top_k": self.top_k}
                step.output = {"n_docs": len(retrieved_docs)}
            steps.append(step)

            context_text = ContextFormatter.format_numbered_from_docs(retrieved_docs)
            prompt = self.prompt_builder.build_rag(query.text, context_text)

            # Step 3a: Generate with context
            with self.generator_provider.load() as generator:
                with StepTimer("generate", model=self.generator_provider.model_name) as step:
                    answer = generator.generate(prompt)
                    step.input = {"query": query.text, "with_context": True}
                    step.output = answer
                steps.append(step)

                # Step 4: Optionally verify
                if self.verify_answer:
                    with StepTimer("verify", model=self.generator_provider.model_name) as step:
                        verification_result = self._verify_grounding(
                            generator, query.text, answer, context_text
                        )
                        step.input = {"answer": answer[:100]}
                        step.output = {"verification": verification_result}
                    steps.append(step)

                    # Fallback if not supported
                    if verification_result == "NOT_SUPPORTED" and self.fallback_to_direct:
                        logger.debug("Answer not supported, falling back to direct")
                        with StepTimer("fallback_generate", model=self.generator_provider.model_name) as step:
                            prompt = self.prompt_builder.build_direct(query.text)
                            answer = generator.generate(prompt)
                            step.input = {"query": query.text, "fallback": True}
                            step.output = answer
                        steps.append(step)
                        retrieved_docs = []
                        context_text = ""
        else:
            # Step 2b: Generate directly without retrieval
            prompt = self.prompt_builder.build_direct(query.text)

            with self.generator_provider.load() as generator:
                with StepTimer("generate", model=self.generator_provider.model_name) as step:
                    answer = generator.generate(prompt)
                    step.input = {"query": query.text, "with_context": False}
                    step.output = answer
                steps.append(step)

        return AgentResult(
            query=query,
            answer=answer,
            steps=steps,
            prompt=prompt,
            metadata={
                "used_retrieval": used_retrieval,
                "confidence": confidence,
                "num_docs": len(retrieved_docs),
                "verification": verification_result,
            },
        )

    def _assess_retrieval_need(self, generator, query: str) -> float:
        """Assess whether retrieval is needed for this query."""
        prompt = RETRIEVAL_DECISION_PROMPT.format(query=query)
        response = generator.generate(prompt, max_tokens=150)

        confidence = 0.3  # Default: uncertain, should retrieve

        response_upper = response.upper()
        if "CONFIDENCE:" in response_upper:
            try:
                idx = response_upper.index("CONFIDENCE:")
                rest = response[idx + 11:idx + 20]
                num_str = ""
                for char in rest:
                    if char.isdigit() or char == ".":
                        num_str += char
                    elif num_str:
                        break
                if num_str:
                    confidence = float(num_str)
                    confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                pass

        return confidence

    def _verify_grounding(
        self, generator, query: str, answer: str, context: str
    ) -> str:
        """Verify if the answer is grounded in the context."""
        prompt = VERIFICATION_PROMPT.format(
            context=context[:3000],
            query=query,
            answer=answer,
        )
        response = generator.generate(prompt, max_tokens=100)

        response_upper = response.upper()
        if "NOT_SUPPORTED" in response_upper:
            return "NOT_SUPPORTED"
        elif "PARTIALLY_SUPPORTED" in response_upper:
            return "PARTIALLY_SUPPORTED"
        elif "SUPPORTED" in response_upper:
            return "SUPPORTED"
        else:
            return "UNKNOWN"
