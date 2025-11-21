"""Tests for RAG agents."""

import pytest

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.agents.direct_llm import DirectLLMAgent


class MockModel:
    """Mock language model for testing."""

    def __init__(self):
        self.model_name = "mock_model"

    def generate(self, prompt, **kwargs):
        return "This is a mock answer."

    def get_embeddings(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


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


if __name__ == "__main__":
    pytest.main([__file__])
