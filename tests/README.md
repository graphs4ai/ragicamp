# RAGiCamp Tests

Unit tests for RAGiCamp framework, with a focus on the robust two-phase evaluation system.

## Running Tests

### Run All Tests

```bash
# Using make (recommended)
make test

# Or using pytest directly
uv run pytest

# With verbose output
uv run pytest -v

# With coverage report
uv run pytest --cov=ragicamp --cov-report=html
```

### Run Specific Test Files

```bash
# Two-phase evaluation tests
uv run pytest tests/test_two_phase_evaluation.py -v

# Config validation tests
uv run pytest tests/test_config.py -v

# Checkpointing tests
uv run pytest tests/test_checkpointing.py -v

# Metrics tests
uv run pytest tests/test_metrics.py -v

# Factory tests
uv run pytest tests/test_factory.py -v

# Agent tests
uv run pytest tests/test_agents.py -v
```

### Run Specific Test Classes

```bash
# Test a specific class
uv run pytest tests/test_two_phase_evaluation.py::TestTwoPhaseEvaluation -v

# Test a specific method
uv run pytest tests/test_two_phase_evaluation.py::TestTwoPhaseEvaluation::test_generate_predictions_phase -v
```

### Run Tests by Marker

```bash
# Run only unit tests
uv run pytest -m unit

# Run only checkpoint tests
uv run pytest -m checkpoint

# Run only config tests
uv run pytest -m config

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Structure

```
tests/
├── README.md                          # This file
├── test_agents.py                     # Agent functionality tests
├── test_two_phase_evaluation.py       # Two-phase evaluation system tests ⭐ NEW
├── test_config.py                     # Config validation tests
├── test_checkpointing.py              # LLM judge checkpointing tests ⭐ NEW
├── test_metrics.py                    # Metrics computation tests
└── test_factory.py                    # Component factory tests
```

## Test Categories

### 1. Two-Phase Evaluation (`test_two_phase_evaluation.py`)

Tests the robust two-phase evaluation system:

- ✅ **Phase 1: Generate predictions**
  - `test_generate_predictions_phase()` - Basic generation
  - `test_generate_predictions_with_batch()` - Batch processing
  - `test_generate_predictions_saves_questions()` - Questions file
  - `test_generate_predictions_with_limit()` - Example limits

- ✅ **Phase 2: Compute metrics**
  - `test_compute_metrics_on_saved_predictions()` - Metrics computation

- ✅ **Robustness**
  - `test_predictions_saved_before_metrics_failure()` - Failure handling
  - `test_empty_dataset()` - Edge cases

**Key Features Tested:**
- Predictions saved immediately
- Metrics computed separately
- Failure recovery
- Batch processing

### 2. Checkpointing (`test_checkpointing.py`)

Tests LLM judge checkpoint system:

- ✅ **Checkpoint Creation**
  - `test_checkpoint_created_on_failure()` - Checkpoint on failure
  - `test_checkpoint_resume()` - Resume from checkpoint
  - `test_checkpoint_saves_progress_every_5_batches()` - Regular saves

- ✅ **Checkpoint Content**
  - `test_checkpoint_cache_format()` - Cache key format
  - `test_binary_judgment_checkpoint()` - Binary judgments
  - `test_ternary_judgment_checkpoint()` - Ternary judgments

- ✅ **Checkpoint Cleanup**
  - `test_checkpoint_deleted_on_success()` - Cleanup after success
  - `test_no_checkpoint_file_on_first_run()` - No checkpoint needed initially

**Key Features Tested:**
- Checkpoint saves every 5 batches
- Resume from checkpoint after failure
- Checkpoint cleanup on success
- Binary/ternary judgment types

### 3. Config Validation (`test_config.py`)

Tests configuration schemas and validation:

- ✅ **Evaluation Modes**
  - `test_generate_mode()` - Generate only mode
  - `test_evaluate_mode_requires_predictions_file()` - Evaluate requirements
  - `test_both_mode()` - Both modes
  - `test_invalid_mode_raises_error()` - Invalid mode handling

- ✅ **Config Schemas**
  - `test_huggingface_model_config()` - Model configs
  - `test_direct_llm_agent_config()` - Agent configs
  - `test_dataset_config()` - Dataset configs
  - `test_rag_agent_requires_retriever()` - Retriever requirements

**Key Features Tested:**
- Three evaluation modes (generate, evaluate, both)
- Config validation with Pydantic
- Required field checking
- Type safety

### 4. Metrics (`test_metrics.py`)

Tests metrics computation:

- ✅ **Exact Match**
  - `test_exact_match_perfect()` - Perfect matches
  - `test_exact_match_partial()` - Partial matches
  - `test_exact_match_multiple_references()` - Multiple refs

- ✅ **F1 Score**
  - `test_f1_perfect()` - Perfect F1
  - `test_f1_partial_overlap()` - Partial overlap
  - `test_f1_no_overlap()` - No overlap

- ✅ **Edge Cases**
  - `test_empty_predictions()` - Empty inputs
  - `test_whitespace_handling()` - Whitespace normalization
  - `test_punctuation_handling()` - Punctuation handling

**Key Features Tested:**
- Exact match computation
- F1 score computation
- Normalization
- Edge case handling

### 5. Factory (`test_factory.py`)

Tests component factory pattern:

- ✅ **Model Factory**
  - `test_create_huggingface_model()` - HF model creation
  - `test_create_openai_model()` - OpenAI model creation
  - `test_create_model_invalid_type()` - Error handling

- ✅ **Agent Factory**
  - `test_create_direct_llm_agent()` - DirectLLM creation
  - `test_create_fixed_rag_agent()` - FixedRAG creation
  - `test_create_rag_agent_without_retriever()` - Error handling

- ✅ **Dataset & Metrics Factory**
  - `test_create_natural_questions_dataset()` - Dataset creation
  - `test_create_multiple_metrics()` - Multiple metrics

**Key Features Tested:**
- Component creation from configs
- Type field removal
- Error handling
- Parameter passing

### 6. Agents (`test_agents.py`)

Tests agent functionality:

- ✅ **DirectLLM Agent**
  - `test_direct_llm_agent()` - Basic agent functionality
  - `test_rag_context()` - Context creation
  - `test_rag_response()` - Response structure

**Key Features Tested:**
- Agent answer generation
- Context and response structures
- Mock model integration

## Writing New Tests

### Template for New Test File

```python
"""Tests for [component name]."""

import pytest
from unittest.mock import Mock

from ragicamp.[module] import [Component]


class Test[ComponentName]:
    """Test [component] functionality."""
    
    def test_[feature_name](self):
        """Test [specific feature]."""
        # Arrange
        component = [Component]()
        
        # Act
        result = component.method()
        
        # Assert
        assert result == expected
```

### Guidelines

1. **Use descriptive test names**: `test_generate_predictions_with_batch()` not `test_batch()`
2. **One assertion per test**: Test one thing at a time
3. **Use mocks for external dependencies**: Don't make real API calls
4. **Test edge cases**: Empty inputs, failures, edge conditions
5. **Add docstrings**: Explain what the test does

### Mock Pattern

```python
from unittest.mock import Mock, patch

# Mock a model
mock_model = Mock()
mock_model.generate.return_value = "mock answer"

# Patch a class
with patch('ragicamp.models.HuggingFaceModel') as mock_hf:
    model = ComponentFactory.create_model(config)
    mock_hf.assert_called_once()
```

## Coverage

Check test coverage:

```bash
# Generate coverage report
uv run pytest --cov=ragicamp --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

Target: **>= 80% coverage** for core components

## Continuous Integration

Tests run automatically on:
- Every commit (if CI is configured)
- Pull requests
- Before deployment

## Common Issues

### Import Errors

```bash
# Make sure you're in the project root
cd /path/to/ragicamp

# Make sure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Or use uv run
uv run pytest
```

### Slow Tests

```bash
# Skip slow tests
uv run pytest -m "not slow"

# Run only fast tests
uv run pytest -m "not integration"
```

### Test Failures

```bash
# Run with more verbose output
uv run pytest -vv

# Stop on first failure
uv run pytest -x

# Run last failed tests
uv run pytest --lf
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Test both happy path and edge cases**
3. **Test error handling**
4. **Add docstrings to test functions**
5. **Run full test suite before committing**

```bash
# Before committing
make test
# Or
uv run pytest -v
```

## Test Philosophy

Our tests follow these principles:

1. **Fast**: Use mocks to avoid slow operations
2. **Isolated**: Each test is independent
3. **Deterministic**: Same input → same output
4. **Comprehensive**: Cover happy paths, edge cases, and errors
5. **Maintainable**: Clear names and good documentation

## Questions?

- Check existing tests for examples
- See [pytest documentation](https://docs.pytest.org/)
- Ask in the team chat

---

**Happy testing!** 🧪✨

