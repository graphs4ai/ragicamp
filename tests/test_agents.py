"""Tests for RAG agents."""

import pytest

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.agents.iterative_rag import IterativeRAGAgent
from ragicamp.agents.self_rag import SelfRAGAgent
from ragicamp.retrievers.base import Document


class MockModel:
    """Mock language model for testing."""

    def __init__(self, responses=None):
        self.model_name = "mock_model"
        self._responses = responses or {}
        self._call_count = 0

    def generate(self, prompt, **kwargs):
        self._call_count += 1
        # Check for specific response patterns
        prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
        
        if "confidence:" in prompt_lower or "retrieval" in prompt_lower:
            return "CONFIDENCE: 0.3\nThis needs external information."
        if "sufficient" in prompt_lower:
            return "SUFFICIENT\nThe context answers the question."
        if "verify" in prompt_lower or "verification" in prompt_lower:
            return "SUPPORTED\nThe answer matches the context."
        if "refined query" in prompt_lower:
            return "What are the specific details?"
        
        return "This is a mock answer."

    def get_embeddings(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, docs=None):
        self._docs = docs or [
            Document(id="doc1", text="Document 1 content about the topic.", metadata={"source": "test"}),
            Document(id="doc2", text="Document 2 with more information.", metadata={"source": "test"}),
            Document(id="doc3", text="Document 3 additional context.", metadata={"source": "test"}),
        ]
    
    def retrieve(self, query, top_k=5):
        return self._docs[:top_k]
    
    def batch_retrieve(self, queries, top_k=5):
        return [self._docs[:top_k] for _ in queries]


def test_direct_llm_agent():
    """Test DirectLLMAgent."""
    model = MockModel()
    agent = DirectLLMAgent(name="test_agent", model=model, system_prompt="Test prompt")

    # Test answer generation
    response = agent.answer("What is the capital of France?")

    assert isinstance(response, RAGResponse)
    assert isinstance(response.context, RAGContext)
    assert response.answer == "This is a mock answer."
    assert response.context.query == "What is the capital of France?"
    assert len(response.context.retrieved_docs) == 0  # No retrieval for direct LLM


def test_rag_context():
    """Test RAGContext dataclass."""
    context = RAGContext(
        query="Test query",
        retrieved_docs=[{"text": "doc1"}, {"text": "doc2"}],
        metadata={"key": "value"},
    )

    assert context.query == "Test query"
    assert len(context.retrieved_docs) == 2
    assert context.metadata["key"] == "value"
    assert len(context.intermediate_steps) == 0


def test_rag_response():
    """Test RAGResponse dataclass."""
    context = RAGContext(query="Test")
    response = RAGResponse(
        answer="Test answer", context=context, confidence=0.95, metadata={"tokens": 10}
    )

    assert response.answer == "Test answer"
    assert response.confidence == 0.95
    assert response.metadata["tokens"] == 10


class TestIterativeRAGAgent:
    """Tests for IterativeRAGAgent."""
    
    def test_iterative_rag_initialization(self):
        """Test IterativeRAGAgent can be instantiated."""
        model = MockModel()
        retriever = MockRetriever()
        
        agent = IterativeRAGAgent(
            name="iterative_test",
            model=model,
            retriever=retriever,
            top_k=3,
            max_iterations=2,
        )
        
        assert agent.name == "iterative_test"
        assert agent.max_iterations == 2
        assert agent.top_k == 3
        assert agent.stop_on_sufficient is True  # Default
    
    def test_iterative_rag_answer(self):
        """Test IterativeRAGAgent generates answers."""
        model = MockModel()
        retriever = MockRetriever()
        
        agent = IterativeRAGAgent(
            name="iterative_test",
            model=model,
            retriever=retriever,
            max_iterations=1,
        )
        
        response = agent.answer("What is the topic?")
        
        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.context.retrieved_docs) > 0
        # Check metadata has iteration info
        assert "iterations" in response.context.metadata
    
    def test_iterative_rag_batch_answer(self):
        """Test IterativeRAGAgent batch processing."""
        model = MockModel()
        retriever = MockRetriever()
        
        agent = IterativeRAGAgent(
            name="iterative_test",
            model=model,
            retriever=retriever,
            max_iterations=1,
        )
        
        responses = agent.batch_answer(["Q1?", "Q2?"])
        
        assert len(responses) == 2
        assert all(isinstance(r, RAGResponse) for r in responses)
    
    def test_iterative_rag_registered(self):
        """Test IterativeRAGAgent is registered with factory."""
        from ragicamp.factory import AgentFactory
        
        assert "iterative_rag" in AgentFactory.get_available_agents()


class TestSelfRAGAgent:
    """Tests for SelfRAGAgent."""
    
    def test_self_rag_initialization(self):
        """Test SelfRAGAgent can be instantiated."""
        model = MockModel()
        retriever = MockRetriever()
        
        agent = SelfRAGAgent(
            name="self_rag_test",
            model=model,
            retriever=retriever,
            retrieval_threshold=0.5,
            verify_answer=True,
        )
        
        assert agent.name == "self_rag_test"
        assert agent.retrieval_threshold == 0.5
        assert agent.verify_answer is True
        assert agent.fallback_to_direct is True  # Default
    
    def test_self_rag_answer_with_retrieval(self):
        """Test SelfRAGAgent with low confidence (uses retrieval)."""
        model = MockModel()  # Returns low confidence
        retriever = MockRetriever()
        
        agent = SelfRAGAgent(
            name="self_rag_test",
            model=model,
            retriever=retriever,
            retrieval_threshold=0.5,
        )
        
        response = agent.answer("What happened in 2024?")
        
        assert isinstance(response, RAGResponse)
        # With mock returning 0.3 confidence, should use retrieval
        assert response.context.metadata.get("used_retrieval") is True
    
    def test_self_rag_high_threshold_skips_retrieval(self):
        """Test SelfRAGAgent skips retrieval with very low threshold."""
        model = MockModel()  # Returns 0.3 confidence
        retriever = MockRetriever()
        
        # With threshold of 0.2, confidence 0.3 > 0.2 means skip retrieval
        agent = SelfRAGAgent(
            name="self_rag_test",
            model=model,
            retriever=retriever,
            retrieval_threshold=0.2,  # Lower threshold
        )
        
        response = agent.answer("What is 2+2?")
        
        assert isinstance(response, RAGResponse)
        # Confidence 0.3 > threshold 0.2, should skip retrieval
        assert response.context.metadata.get("used_retrieval") is False
    
    def test_self_rag_batch_answer(self):
        """Test SelfRAGAgent batch processing."""
        model = MockModel()
        retriever = MockRetriever()
        
        agent = SelfRAGAgent(
            name="self_rag_test",
            model=model,
            retriever=retriever,
        )
        
        responses = agent.batch_answer(["Q1?", "Q2?"])
        
        assert len(responses) == 2
        assert all(isinstance(r, RAGResponse) for r in responses)
    
    def test_self_rag_registered(self):
        """Test SelfRAGAgent is registered with factory."""
        from ragicamp.factory import AgentFactory
        
        assert "self_rag" in AgentFactory.get_available_agents()


class TestAgentFactoryFromSpec:
    """Tests for AgentFactory.from_spec()."""
    
    def test_from_spec_direct_agent(self):
        """Test creating direct agent from spec."""
        from ragicamp.factory import AgentFactory
        from ragicamp.spec import ExperimentSpec
        
        spec = ExperimentSpec(
            name="test_direct",
            exp_type="direct",
            model="mock",
            dataset="nq",
            prompt="concise",
        )
        
        model = MockModel()
        agent = AgentFactory.from_spec(spec, model)
        
        assert agent.name == "test_direct"
        assert isinstance(agent, DirectLLMAgent)
    
    def test_from_spec_iterative_rag(self):
        """Test creating iterative RAG agent from spec."""
        from ragicamp.factory import AgentFactory
        from ragicamp.spec import ExperimentSpec
        
        spec = ExperimentSpec(
            name="test_iterative",
            exp_type="rag",
            model="mock",
            dataset="nq",
            prompt="concise",
            retriever="test_retriever",
            agent_type="iterative_rag",
            agent_params=(("max_iterations", 3), ("stop_on_sufficient", False)),
        )
        
        model = MockModel()
        retriever = MockRetriever()
        agent = AgentFactory.from_spec(spec, model, retriever)
        
        assert agent.name == "test_iterative"
        assert isinstance(agent, IterativeRAGAgent)
        assert agent.max_iterations == 3
        assert agent.stop_on_sufficient is False
    
    def test_from_spec_self_rag(self):
        """Test creating self-RAG agent from spec."""
        from ragicamp.factory import AgentFactory
        from ragicamp.spec import ExperimentSpec
        
        spec = ExperimentSpec(
            name="test_selfrag",
            exp_type="rag",
            model="mock",
            dataset="nq",
            prompt="concise",
            retriever="test_retriever",
            agent_type="self_rag",
            agent_params=(("retrieval_threshold", 0.7), ("verify_answer", True)),
        )
        
        model = MockModel()
        retriever = MockRetriever()
        agent = AgentFactory.from_spec(spec, model, retriever)
        
        assert agent.name == "test_selfrag"
        assert isinstance(agent, SelfRAGAgent)
        assert agent.retrieval_threshold == 0.7
        assert agent.verify_answer is True


if __name__ == "__main__":
    pytest.main([__file__])
