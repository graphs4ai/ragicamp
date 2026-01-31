"""Self-RAG agent - Adaptive retrieval based on query analysis.

This agent decides whether to use retrieval based on query characteristics:
1. Assess query: "Do I need external information?"
2. If confident (above threshold): generate directly (no retrieval)
3. If unsure: use RAG path
4. Optionally verify answer is supported by context

Based on the Self-RAG paper concept of adaptive retrieval.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.core.logging import get_logger
from ragicamp.core.schemas import RAGResponseMeta, RetrievedDoc, AgentType
from ragicamp.factory import AgentFactory
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder

if TYPE_CHECKING:
    from ragicamp.rag.query_transform.base import QueryTransformer
    from ragicamp.rag.rerankers.base import Reranker

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


@AgentFactory.register("self_rag")
class SelfRAGAgent(RAGAgent):
    """Self-RAG agent with adaptive retrieval decision.
    
    This agent dynamically decides whether to use retrieval based on
    the query and optionally verifies that answers are grounded in context.
    
    Flow:
    1. Analyze query to decide if retrieval is needed
    2. If confidence > threshold: generate directly
    3. If confidence <= threshold: retrieve and generate with context
    4. Optionally verify answer is supported by context
    5. If not supported and fallback enabled: regenerate or use direct answer
    
    Example:
        >>> agent = SelfRAGAgent(
        ...     name="self_rag",
        ...     model=model,
        ...     retriever=retriever,
        ...     retrieval_threshold=0.5,
        ...     verify_answer=True,
        ... )
        >>> response = agent.answer("What is 2 + 2?")  # Likely no retrieval
        >>> response = agent.answer("Who won the 2024 election?")  # Will retrieve
    """
    
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        top_k: int = 5,
        retrieval_threshold: float = 0.5,
        verify_answer: bool = False,
        fallback_to_direct: bool = True,
        prompt_builder: Optional[PromptBuilder] = None,
        # Advanced options
        query_transformer: Optional["QueryTransformer"] = None,
        reranker: Optional["Reranker"] = None,
        top_k_retrieve: Optional[int] = None,
        # Legacy parameters
        system_prompt: str = "You are a helpful assistant. Answer questions accurately and concisely.",
        **kwargs: Any,
    ):
        """Initialize the self-RAG agent.
        
        Args:
            name: Agent identifier
            model: Language model for generation and decision making
            retriever: Document retriever
            top_k: Number of documents to retrieve
            retrieval_threshold: Confidence threshold for skipping retrieval.
                                 Higher = more likely to skip retrieval.
                                 Default 0.5 means retrieve unless very confident.
            verify_answer: Whether to verify answer is grounded in context
            fallback_to_direct: If verification fails, fall back to direct answer
            prompt_builder: PromptBuilder for answer generation
            query_transformer: Optional query transformer
            reranker: Optional reranker
            top_k_retrieve: Documents to retrieve before reranking
            system_prompt: System prompt for direct answer generation
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.retrieval_threshold = retrieval_threshold
        self.verify_answer = verify_answer
        self.fallback_to_direct = fallback_to_direct
        
        # Optional components
        self.query_transformer = query_transformer
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve or (top_k * 4 if reranker else top_k)
        
        # Prompt builders
        if prompt_builder is not None:
            self.prompt_builder = prompt_builder
        else:
            from ragicamp.utils.prompts import PromptConfig
            self.prompt_builder = PromptBuilder(PromptConfig(style=system_prompt))
        
        self._system_prompt = system_prompt
    
    def _assess_retrieval_need(self, query: str) -> float:
        """Assess whether retrieval is needed for this query.
        
        Returns:
            Confidence score (0-1) that query can be answered without retrieval.
            Higher = more confident, less likely to retrieve.
        """
        prompt = RETRIEVAL_DECISION_PROMPT.format(query=query)
        response = self.model.generate(prompt, max_new_tokens=150)
        
        # Parse confidence from response
        confidence = 0.3  # Default to "uncertain, should retrieve"
        
        # Look for "CONFIDENCE: X.X" pattern
        response_upper = response.upper()
        if "CONFIDENCE:" in response_upper:
            try:
                # Find the number after CONFIDENCE:
                idx = response_upper.index("CONFIDENCE:")
                rest = response[idx + 11:idx + 20]  # Get chars after "CONFIDENCE:"
                # Extract first number-like substring
                num_str = ""
                for char in rest:
                    if char.isdigit() or char == ".":
                        num_str += char
                    elif num_str:
                        break
                if num_str:
                    confidence = float(num_str)
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except (ValueError, IndexError):
                pass
        
        logger.debug("Retrieval need assessment for '%s': confidence=%.2f", query[:50], confidence)
        return confidence
    
    def _retrieve(self, query: str) -> List[Document]:
        """Retrieve documents for a query."""
        # Apply query transformer if present
        search_query = query
        if self.query_transformer:
            search_query = self.query_transformer.transform(query)
        
        # Retrieve
        docs = self.retriever.retrieve(search_query, top_k=self.top_k_retrieve)
        
        # Rerank if present
        if self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=self.top_k)
        else:
            docs = docs[:self.top_k]
        
        return docs
    
    def _verify_grounding(self, query: str, answer: str, context: str) -> str:
        """Verify if the answer is grounded in the context.
        
        Returns:
            One of: "SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"
        """
        prompt = VERIFICATION_PROMPT.format(
            context=context[:3000],  # Limit context
            query=query,
            answer=answer,
        )
        response = self.model.generate(prompt, max_new_tokens=100)
        
        # Parse verification result
        response_upper = response.upper()
        if "NOT_SUPPORTED" in response_upper:
            return "NOT_SUPPORTED"
        elif "PARTIALLY_SUPPORTED" in response_upper:
            return "PARTIALLY_SUPPORTED"
        elif "SUPPORTED" in response_upper:
            return "SUPPORTED"
        else:
            return "UNKNOWN"
    
    def _generate_direct(self, query: str, **kwargs: Any) -> str:
        """Generate answer directly without retrieval."""
        prompt = self.prompt_builder.build_direct(query)
        return self.model.generate(prompt, **kwargs)
    
    def _generate_with_context(
        self, 
        query: str, 
        context: str, 
        **kwargs: Any,
    ) -> str:
        """Generate answer using retrieved context."""
        prompt = self.prompt_builder.build_rag(query, context)
        return self.model.generate(prompt, **kwargs)
    
    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer with adaptive retrieval decision.
        
        Args:
            query: The input question
            **kwargs: Additional generation parameters
            
        Returns:
            RAGResponse with answer and decision metadata
        """
        # Step 1: Assess if retrieval is needed
        confidence = self._assess_retrieval_need(query)
        used_retrieval = confidence <= self.retrieval_threshold
        
        decision_info = {
            "confidence": confidence,
            "threshold": self.retrieval_threshold,
            "used_retrieval": used_retrieval,
        }
        
        retrieved_docs: List[Document] = []
        context_text = ""
        verification_result = None
        
        if used_retrieval:
            # Step 2a: Retrieve and generate with context
            retrieved_docs = self._retrieve(query)
            context_text = ContextFormatter.format_numbered(retrieved_docs)
            prompt = self.prompt_builder.build_rag(query, context_text)
            answer = self.model.generate(prompt, **kwargs)
            
            # Step 3: Optionally verify grounding
            if self.verify_answer:
                verification_result = self._verify_grounding(query, answer, context_text)
                decision_info["verification"] = verification_result
                
                # If not supported and fallback enabled, try direct answer
                if verification_result == "NOT_SUPPORTED" and self.fallback_to_direct:
                    logger.debug("Answer not supported by context, falling back to direct")
                    direct_answer = self._generate_direct(query, **kwargs)
                    decision_info["fallback_used"] = True
                    answer = direct_answer
                    # Clear retrieval info since we didn't use it
                    retrieved_docs = []
                    context_text = ""
                    prompt = self.prompt_builder.build_direct(query)
        else:
            # Step 2b: Generate directly without retrieval
            prompt = self.prompt_builder.build_direct(query)
            answer = self.model.generate(prompt, **kwargs)
        
        # Build context object
        context = RAGContext(
            query=query,
            retrieved_docs=retrieved_docs,
            intermediate_steps=[decision_info],
            metadata={
                "used_retrieval": used_retrieval,
                "confidence": confidence,
                "num_docs": len(retrieved_docs),
            },
        )
        
        # Build structured retrieved docs
        retrieved_structured = [
            RetrievedDoc(
                rank=i + 1,
                content=doc.text,
                score=getattr(doc, "score", None),
            )
            for i, doc in enumerate(retrieved_docs)
        ]
        
        return RAGResponse(
            answer=answer,
            context=context,
            prompt=prompt,
            metadata=RAGResponseMeta(
                agent_type=AgentType.SELF_RAG,
                num_docs_used=len(retrieved_docs),
                retrieved_docs=retrieved_structured if retrieved_structured else None,
            ),
        )
    
    def batch_answer(self, queries: List[str], **kwargs: Any) -> List[RAGResponse]:
        """Generate answers for multiple queries.
        
        Note: Each query requires its own retrieval decision, so this
        processes queries sequentially. Future optimization could batch
        the decision-making step.
        
        Args:
            queries: List of input questions
            **kwargs: Additional generation parameters
            
        Returns:
            List of RAGResponse objects
        """
        return [self.answer(q, **kwargs) for q in queries]
