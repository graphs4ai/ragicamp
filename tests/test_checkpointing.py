"""Tests for LLM judge checkpointing system."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric


class MockJudgeModel:
    """Mock LLM judge model for testing."""
    
    def __init__(self, fail_at_batch=None):
        self.model_name = "mock_judge"
        self.call_count = 0
        self.fail_at_batch = fail_at_batch
    
    def generate(self, prompts, **kwargs):
        """Generate mock judgments."""
        self.call_count += 1
        
        # Simulate failure at specific batch
        if self.fail_at_batch is not None and self.call_count == self.fail_at_batch:
            raise Exception("Simulated API failure (403)")
        
        # Return mock judgments
        if isinstance(prompts, list):
            return [f"JUDGMENT: CORRECT\nThe answer is right." for _ in prompts]
        return "JUDGMENT: CORRECT\nThe answer is right."


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""
    
    def test_checkpoint_created_on_failure(self):
        """Test that checkpoint is created when LLM judge fails."""
        judge_model = MockJudgeModel(fail_at_batch=3)  # Fail at 3rd batch
        metric = LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="binary",
            batch_size=4
        )
        
        predictions = [f"Answer {i}" for i in range(16)]  # 4 batches of 4
        references = [["Ref"] for _ in range(16)]
        questions = [f"Q{i}" for i in range(16)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            # Should fail at batch 3 but save checkpoint
            with pytest.raises(Exception) as exc_info:
                metric.compute(
                    predictions=predictions,
                    references=references,
                    questions=questions,
                    checkpoint_path=str(checkpoint_path)
                )
            
            assert "Simulated API failure" in str(exc_info.value)
            
            # Checkpoint should exist
            assert checkpoint_path.exists()
            
            # Load and verify checkpoint
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            assert "last_batch" in checkpoint_data
            assert "cache" in checkpoint_data
            assert checkpoint_data["judgment_type"] == "binary"
            assert checkpoint_data["batch_size"] == 4
            
            # Should have cached judgments from batches 1 and 2
            assert len(checkpoint_data["cache"]) > 0
    
    def test_checkpoint_resume(self):
        """Test resuming from checkpoint after failure."""
        # First run: fail at batch 2
        judge_model1 = MockJudgeModel(fail_at_batch=2)
        metric1 = LLMJudgeQAMetric(
            judge_model=judge_model1,
            judgment_type="binary",
            batch_size=4
        )
        
        predictions = [f"Answer {i}" for i in range(12)]  # 3 batches
        references = [["Ref"] for _ in range(12)]
        questions = [f"Q{i}" for i in range(12)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            # First attempt - fails and saves checkpoint
            with pytest.raises(Exception):
                metric1.compute(
                    predictions=predictions,
                    references=references,
                    questions=questions,
                    checkpoint_path=str(checkpoint_path)
                )
            
            assert checkpoint_path.exists()
            assert judge_model1.call_count == 2  # Called twice before failing
            
            # Second attempt - resume from checkpoint
            judge_model2 = MockJudgeModel()  # No failure this time
            metric2 = LLMJudgeQAMetric(
                judge_model=judge_model2,
                judgment_type="binary",
                batch_size=4
            )
            
            results = metric2.compute(
                predictions=predictions,
                references=references,
                questions=questions,
                checkpoint_path=str(checkpoint_path)
            )
            
            # Should have resumed from batch 2 (skipped batch 0 and 1)
            # Only batch 2 should be processed
            assert judge_model2.call_count == 1  # Only called once for remaining batch
            
            # Checkpoint should be deleted on success
            assert not checkpoint_path.exists()
            
            # Results should be complete
            assert "llm_judge_qa" in results
            assert results["llm_judge_qa"] >= 0.0


class TestCheckpointContent:
    """Test checkpoint content and structure."""
    
    def test_checkpoint_saves_progress_every_5_batches(self):
        """Test that checkpoint is saved every 5 batches."""
        judge_model = MockJudgeModel()
        metric = LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="binary",
            batch_size=4
        )
        
        # Mock the _save_checkpoint method to track calls
        save_calls = []
        original_save = metric._save_checkpoint
        
        def mock_save(checkpoint_path, last_batch):
            save_calls.append(last_batch)
            original_save(checkpoint_path, last_batch)
        
        metric._save_checkpoint = mock_save
        
        predictions = [f"Answer {i}" for i in range(40)]  # 10 batches
        references = [["Ref"] for _ in range(40)]
        questions = [f"Q{i}" for i in range(40)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            results = metric.compute(
                predictions=predictions,
                references=references,
                questions=questions,
                checkpoint_path=str(checkpoint_path)
            )
            
            # Should save at batches 4, 9 (every 5 batches)
            assert len(save_calls) >= 2
    
    def test_checkpoint_cache_format(self):
        """Test checkpoint cache key format."""
        judge_model = MockJudgeModel(fail_at_batch=2)
        metric = LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="binary",
            batch_size=4
        )
        
        predictions = ["Answer 1", "Answer 2"]
        references = [["Ref 1"], ["Ref 2"]]
        questions = ["Q1", "Q2"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            try:
                metric.compute(
                    predictions=predictions,
                    references=references,
                    questions=questions,
                    checkpoint_path=str(checkpoint_path)
                )
            except Exception:
                pass
            
            # Load checkpoint
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Cache keys should be in format: prediction:::reference:::question
            cache_keys = list(checkpoint_data["cache"].keys())
            if cache_keys:
                key = cache_keys[0]
                assert ":::" in key
                parts = key.split(":::")
                assert len(parts) == 3  # prediction, reference, question


class TestCheckpointCleanup:
    """Test checkpoint cleanup after successful completion."""
    
    def test_checkpoint_deleted_on_success(self):
        """Test that checkpoint is deleted after successful completion."""
        judge_model = MockJudgeModel()
        metric = LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="binary",
            batch_size=4
        )
        
        predictions = [f"Answer {i}" for i in range(8)]
        references = [["Ref"] for _ in range(8)]
        questions = [f"Q{i}" for i in range(8)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            # Create a dummy checkpoint file
            checkpoint_path.write_text('{"test": "data"}')
            assert checkpoint_path.exists()
            
            # Compute successfully
            results = metric.compute(
                predictions=predictions,
                references=references,
                questions=questions,
                checkpoint_path=str(checkpoint_path)
            )
            
            # Checkpoint should be cleaned up
            assert not checkpoint_path.exists()
    
    def test_no_checkpoint_file_on_first_run(self):
        """Test that no checkpoint is needed on first successful run."""
        judge_model = MockJudgeModel()
        metric = LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="binary",
            batch_size=4
        )
        
        predictions = ["Answer 1", "Answer 2"]
        references = [["Ref 1"], ["Ref 2"]]
        questions = ["Q1", "Q2"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            # First run - should succeed without checkpoint
            results = metric.compute(
                predictions=predictions,
                references=references,
                questions=questions,
                checkpoint_path=str(checkpoint_path)
            )
            
            assert "llm_judge_qa" in results
            # No checkpoint file should exist
            assert not checkpoint_path.exists()


class TestBinaryVsTernaryJudgment:
    """Test binary vs ternary judgment types."""
    
    def test_binary_judgment_checkpoint(self):
        """Test checkpoint with binary judgment type."""
        judge_model = MockJudgeModel(fail_at_batch=2)
        metric = LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="binary",
            batch_size=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            try:
                metric.compute(
                    predictions=["A1", "A2", "A3", "A4"],
                    references=[["R1"], ["R2"], ["R3"], ["R4"]],
                    questions=["Q1", "Q2", "Q3", "Q4"],
                    checkpoint_path=str(checkpoint_path)
                )
            except Exception:
                pass
            
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            assert checkpoint_data["judgment_type"] == "binary"
    
    def test_ternary_judgment_checkpoint(self):
        """Test checkpoint with ternary judgment type."""
        judge_model = MockJudgeModel(fail_at_batch=2)
        metric = LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="ternary",
            batch_size=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            try:
                metric.compute(
                    predictions=["A1", "A2", "A3", "A4"],
                    references=[["R1"], ["R2"], ["R3"], ["R4"]],
                    questions=["Q1", "Q2", "Q3", "Q4"],
                    checkpoint_path=str(checkpoint_path)
                )
            except Exception:
                pass
            
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            assert checkpoint_data["judgment_type"] == "ternary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

