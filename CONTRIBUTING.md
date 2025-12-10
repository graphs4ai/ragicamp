# Contributing to RAGiCamp

Thanks for your interest in contributing! This guide covers everything you need to know.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USER/ragicamp.git
cd ragicamp
make setup

# Run tests
make test

# Format code
make format
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/my-bugfix
```

### 2. Make Changes

- Write code
- Add tests for new features
- Update documentation if needed

### 3. Test Locally

```bash
# Format code (required)
make format

# Run tests
make test

# Validate configs
make validate-all-configs
```

### 4. Submit PR

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/my-feature
```

Then open a Pull Request on GitHub.

---

## Code Style

### Formatting

We use **Black** and **isort**:

```bash
# Auto-format everything
make format
```

### Type Hints

Use type hints for all public functions:

```python
def compute_score(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """Short description.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Example:
        >>> my_function("hello", 42)
        True
    """
```

---

## Testing

### Running Tests

```bash
# All tests
make test

# Specific file
uv run pytest tests/test_metrics.py -v

# With coverage
make test-coverage
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_agents.py           # Agent tests
├── test_config.py           # Config validation
├── test_factory.py          # Factory pattern
├── test_metrics.py          # Metrics computation
├── test_checkpointing.py    # Checkpoint system
└── test_two_phase_evaluation.py  # Two-phase eval
```

### Writing Tests

```python
"""Tests for my feature."""

import pytest
from ragicamp.my_module import MyClass


class TestMyClass:
    """Test MyClass functionality."""

    def test_basic_functionality(self):
        """Test basic case."""
        obj = MyClass()
        result = obj.method()
        assert result == expected

    def test_edge_case(self):
        """Test edge case."""
        obj = MyClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

### Using Fixtures

We provide reusable fixtures in `tests/conftest.py`:

```python
def test_with_mock_model(mock_model):
    """Fixtures are auto-injected."""
    agent = DirectLLMAgent(model=mock_model)
    response = agent.answer("test question")
    assert response.answer == "Mock answer"

def test_with_sample_data(sample_dataset):
    """Use sample QA examples."""
    assert len(sample_dataset) == 5
```

Available fixtures:
- `mock_model` - Mock language model
- `mock_retriever` - Mock retriever
- `sample_dataset` - Sample QA examples
- `temp_dir` - Temporary directory
- `temp_config_file` - Temp config YAML

---

## Project Structure

```
ragicamp/
├── src/ragicamp/        # Main library
│   ├── core/            # Exceptions, logging, constants
│   ├── agents/          # RAG agents
│   ├── models/          # LLM wrappers
│   ├── metrics/         # Evaluation metrics
│   ├── datasets/        # Dataset loaders
│   ├── retrievers/      # Document retrievers
│   ├── config/          # Pydantic schemas
│   └── evaluation/      # Evaluator
├── conf/                # Hydra configs
├── scripts/             # Utility scripts
├── tests/               # Test suite
└── docs/                # Documentation
```

---

## Adding New Components

### New Metric

1. Create `src/ragicamp/metrics/my_metric.py`:

```python
from ragicamp.metrics.base import Metric

class MyMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(name="my_metric", **kwargs)

    def compute(self, predictions, references, **kwargs):
        # Your logic here
        return {"my_metric": score}
```

2. Export in `src/ragicamp/metrics/__init__.py`
3. Add to factory in `src/ragicamp/factory.py`
4. Add tests in `tests/test_metrics.py`

### New Agent

1. Create `src/ragicamp/agents/my_agent.py`
2. Inherit from `RAGAgent`
3. Implement `answer()` method
4. Add to factory and registry

### New Config

1. Create YAML in `conf/` appropriate subdirectory
2. Update `conf/config.yaml` defaults if needed
3. Test with `python -m ragicamp.cli.run --cfg job`

---

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new metric for faithfulness
fix: handle empty predictions in F1
docs: update contributing guide
test: add edge case tests for exact match
refactor: simplify evaluator code
chore: update dependencies
```

---

## Pull Request Guidelines

### PR Title

Use conventional commit format:
- `feat: add X` - New feature
- `fix: resolve Y` - Bug fix
- `docs: update Z` - Documentation
- `test: add tests for W` - Tests

### PR Description

Include:
- **What** - What does this PR do?
- **Why** - Why is this change needed?
- **How** - How was it implemented?
- **Testing** - How was it tested?

### PR Checklist

- [ ] Code follows style guide (`make format`)
- [ ] Tests pass (`make test`)
- [ ] New features have tests
- [ ] Documentation updated if needed
- [ ] Configs validated (`make validate-all-configs`)

---

## Getting Help

- **Questions**: Open a GitHub issue
- **Bugs**: Include reproduction steps
- **Features**: Describe the use case

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Happy contributing!** 🎉
