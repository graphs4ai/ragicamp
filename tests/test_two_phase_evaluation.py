"""Tests for two-phase evaluation system."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.base import Metric


class MockModel:
    """Mock language model."""
    
    def __init__(self):
        self.model_name = "mock_model"
    
    def generate(self, prompt, **kwargs):
        """Generate mock responses."""
        if isinstance(prompt, list):
            return [f"Answer {i}" for i in range(len(prompt))]
        return "Mock answer"


class MockAgent(RAGAgent):
    """Mock agent for testing."""
    
    def __init__(self, name="test_agent"):
        super().__init__(name)
        self.model = MockModel()
    
    def answer(self, query, **kwargs):
        return RAGResponse(
            answer=f"Answer to: {query}",
            context=RAGContext(query=query),
            metadata={"agent": self.name}
        )
    
    def batch_answer(self, queries, **kwargs):
        return [self.answer(q, **kwargs) for q in queries]


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, num_examples=5):
        self.name = "mock_dataset"
        self.examples = [
            Mock(id=f"q{i}", question=f"Question {i}?", answers=[f"Answer {i}"])
            for i in range(num_examples)
        ]
    
    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        return iter(self.examples)


class MockMetric(Metric):
    """Mock metric for testing."""
    
    def __init__(self, name="mock_metric"):
        super().__init__(name)
    
    def compute(self, predictions, references, **kwargs):
        return {self.name: 0.75}
    
    def compute_single(self, prediction, reference, **kwargs):
        return {self.name: 0.8}


class TestTwoPhaseEvaluation:
    """Test two-phase evaluation system."""
    
    def test_generate_predictions_phase(self):
        """Test Phase 1: Generate predictions."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=5)
        evaluator = Evaluator(agent, dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            # Generate predictions
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path),
                num_examples=5
            )
            
            # Verify predictions file was created
            assert Path(predictions_file).exists()
            assert predictions_file.endswith("_predictions_raw.json")
            
            # Load and verify content
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            assert data["agent_name"] == "test_agent"
            assert data["dataset_name"] == "mock_dataset"
            assert data["num_examples"] == 5
            assert data["status"] == "predictions_only"
            assert len(data["predictions"]) == 5
            
            # Verify prediction structure
            pred = data["predictions"][0]
            assert "question_id" in pred
            assert "question" in pred
            assert "prediction" in pred
            assert "expected_answers" in pred
            assert pred["prediction"].startswith("Answer to:")
    
    def test_generate_predictions_with_batch(self):
        """Test Phase 1 with batch processing."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=10)
        evaluator = Evaluator(agent, dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            # Generate predictions with batching
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path),
                num_examples=10,
                batch_size=4
            )
            
            # Verify predictions
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            assert data["num_examples"] == 10
            assert len(data["predictions"]) == 10
    
    def test_generate_predictions_saves_questions(self):
        """Test that questions file is also saved."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=3)
        evaluator = Evaluator(agent, dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path)
            )
            
            # Check questions file exists
            questions_file = Path(predictions_file).parent / "mock_dataset_questions.json"
            assert questions_file.exists()
            
            # Verify questions content
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
            
            assert questions_data["dataset_name"] == "mock_dataset"
            assert questions_data["num_questions"] == 3
            assert len(questions_data["questions"]) == 3
            
            # Check question structure
            q = questions_data["questions"][0]
            assert "id" in q
            assert "question" in q
            assert "expected_answer" in q
            assert "all_acceptable_answers" in q
    
    def test_generate_predictions_with_limit(self):
        """Test generating predictions with num_examples limit."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=10)
        evaluator = Evaluator(agent, dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            # Generate only 5 predictions from 10 examples
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path),
                num_examples=5
            )
            
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            assert data["num_examples"] == 5
            assert len(data["predictions"]) == 5


class TestEvaluateMode:
    """Test evaluate mode (compute metrics on saved predictions)."""
    
    def test_compute_metrics_on_saved_predictions(self):
        """Test computing metrics on previously saved predictions."""
        # First generate predictions
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=5)
        evaluator = Evaluator(agent, dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            # Phase 1: Generate
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path)
            )
            
            # Phase 2: Load and compute metrics
            with open(predictions_file, 'r') as f:
                predictions_data = json.load(f)
            
            predictions = [p["prediction"] for p in predictions_data["predictions"]]
            references = [p["expected_answers"] for p in predictions_data["predictions"]]
            
            # Create metrics
            metric = MockMetric("test_metric")
            
            # Compute metrics
            results = metric.compute(predictions, references)
            
            assert "test_metric" in results
            assert results["test_metric"] == 0.75


class TestBothMode:
    """Test both mode (generate + evaluate)."""
    
    def test_sequential_execution(self):
        """Test that both phases work sequentially."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=5)
        metrics = [MockMetric("metric1"), MockMetric("metric2")]
        evaluator = Evaluator(agent, dataset, metrics)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            # Generate predictions
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path)
            )
            
            # Verify predictions exist
            assert Path(predictions_file).exists()
            
            # Now we could compute metrics on these predictions
            # This simulates the "both" mode
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            assert data["status"] == "predictions_only"


class TestRobustness:
    """Test robustness features."""
    
    def test_predictions_saved_before_metrics_failure(self):
        """Test that predictions are saved even if metrics computation would fail."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=3)
        
        # Metric that will fail
        class FailingMetric(Metric):
            def __init__(self):
                super().__init__("failing_metric")
            
            def compute(self, predictions, references, **kwargs):
                raise Exception("Simulated API failure!")
        
        evaluator = Evaluator(agent, dataset, [FailingMetric()])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            # Generate predictions (should succeed)
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path)
            )
            
            # Predictions should be saved
            assert Path(predictions_file).exists()
            
            # Verify we can load them
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            assert data["num_examples"] == 3
            # Metrics computation would fail, but predictions are safe!
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=0)
        evaluator = Evaluator(agent, dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            
            predictions_file = evaluator.generate_predictions(
                output_path=str(output_path)
            )
            
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            assert data["num_examples"] == 0
            assert len(data["predictions"]) == 0


class TestBackwardCompatibility:
    """Test backward compatibility."""
    
    def test_evaluator_with_no_metrics(self):
        """Test that evaluator can be created without metrics."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=3)
        
        # Should not raise error
        evaluator = Evaluator(agent, dataset)
        
        assert evaluator.metrics == []
    
    def test_evaluator_with_metrics(self):
        """Test that evaluator still works with metrics."""
        agent = MockAgent("test_agent")
        dataset = MockDataset(num_examples=3)
        metrics = [MockMetric("m1"), MockMetric("m2")]
        
        evaluator = Evaluator(agent, dataset, metrics)
        
        assert len(evaluator.metrics) == 2
        assert evaluator.metrics[0].name == "m1"
        assert evaluator.metrics[1].name == "m2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

