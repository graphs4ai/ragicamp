# RAGiCamp Development Guidelines

This document describes the repository philosophy, patterns, and guidelines for both contributors and users.

## Repository Goals

RAGiCamp is a **research framework** for experimenting with RAG approaches. It prioritizes:

1. **Simplicity** - Easy to understand, easy to modify
2. **Modularity** - Components can be swapped independently
3. **Reproducibility** - Experiments should be reproducible via configs
4. **Performance** - Efficient use of GPU resources

This is NOT:
- A production RAG system
- A comprehensive evaluation suite
- An MLOps platform

## Architecture Overview

```
Experiment (with ExperimentCallbacks)
    ├── Agent (DirectLLM or FixedRAG)
    │   ├── Model (HuggingFace or OpenAI or custom)
    │   └── Retriever (optional, for RAG)
    ├── Dataset (NQ, TriviaQA, HotpotQA)
    └── Metrics (F1, EM, BERTScore, etc.)
```

Key classes:
- `Experiment` - Unified entry point for running evaluations
- `ExperimentCallbacks` - Hooks for monitoring progress (on_batch_start, on_complete, etc.)
- `Evaluator` - Lower-level evaluation with checkpointing
- `ComponentFactory` - Creates components from config dicts (supports plugin registration)

Interfaces:
- All base classes in `{module}/base.py`
- Protocols in `core/protocols.py` for duck-typing
- Exception hierarchy: `RAGiCampError` → `ConfigError`, `ModelError`, `EvaluationError`

## Do's

### Code Style
- **Keep functions small** - If a function is >50 lines, split it
- **Use type hints** - All public APIs should have type hints
- **Document with docstrings** - Google style, include examples
- **Fail fast** - Validate inputs early, not deep in the stack

### Patterns to Follow
- **Factory pattern** - Use `ComponentFactory` for creating components from configs
- **Dataclasses** - Use for simple data containers (e.g., `ExperimentResult`)
- **Context managers** - Use for resource cleanup (GPU memory)
- **Checkpointing** - Long operations should save progress

### Adding New Components

New Model (with plugin registration):
```python
from ragicamp import ComponentFactory
from ragicamp.models.base import LanguageModel

@ComponentFactory.register_model("anthropic")
class AnthropicModel(LanguageModel):
    def generate(self, prompt: str, **kwargs) -> str:
        ...
    
    def unload(self) -> None:
        # Clean up GPU memory
        ...

# Now usable in configs: {"type": "anthropic", "model_name": "claude-3"}
```

New Metric:
```python
class MyMetric(Metric):
    @property
    def name(self) -> str:
        return "my_metric"
    
    def compute(self, predictions: List[str], references: List[Any]) -> Dict[str, float]:
        ...
```

New Dataset:
```python
class MyDataset(QADataset):
    def load(self) -> None:
        # Load data into self.examples
        ...
```

### Using Callbacks for Monitoring

```python
from ragicamp import Experiment, ExperimentCallbacks

callbacks = ExperimentCallbacks(
    on_batch_start=lambda i, n: print(f"Batch {i}/{n}"),
    on_complete=lambda r: send_slack_notification(f"Done! F1={r.f1:.3f}"),
)

result = exp.run(batch_size=8, callbacks=callbacks)
```

### Configuration
- **YAML for study configs** - Human readable, easy to version control
- **Dict for component configs** - Passed to factories
- **Use defaults** - Components should work with minimal config

## Don'ts

### Anti-patterns to Avoid

1. **Don't over-abstract**
   ```python
   # Bad: AbstractFactoryBuilder pattern
   # Good: Simple function that creates a thing
   ```

2. **Don't catch broad exceptions**
   ```python
   # Bad
   except Exception as e:
       print(f"Error: {e}")
   
   # Good
   except FileNotFoundError:
       logger.error("Index not found: %s", path)
       raise
   ```

3. **Don't add dependencies for small features**
   ```python
   # Bad: Add rich for slightly nicer progress bars
   # Good: Use tqdm which is already a dependency
   ```

4. **Don't duplicate code across scripts**
   ```python
   # Bad: Copy model loading code to every script
   # Good: Use ComponentFactory.create_model()
   ```

5. **Don't optimize prematurely**
   ```python
   # Bad: Async everything "for performance"
   # Good: Batch processing with existing sync code
   ```

### Things to Avoid
- **Complex inheritance hierarchies** - Prefer composition
- **Singleton patterns** - Use dependency injection instead
- **Magic strings everywhere** - Use constants or enums
- **Mutable default arguments** - `def f(x=[])`  is a bug waiting to happen

## Logging vs Print

**Use print() for:**
- CLI output meant for users
- Progress messages in scripts
- Interactive feedback

**Use logger for:**
- Library code in `src/ragicamp/`
- Debug information
- Error context that should be in log files

```python
# In CLI/scripts
print(f"Running {len(experiments)} experiments...")

# In library code
logger = get_logger(__name__)
logger.debug("Loading model: %s", model_name)
```

## Resource Management

Always clean up GPU memory:

```python
from ragicamp.utils.resource_manager import ResourceManager

# After model use
if hasattr(model, "unload"):
    model.unload()
ResourceManager.clear_gpu_memory()

# Or use context manager
from ragicamp.utils.resource_manager import managed_model

with managed_model(lambda: HuggingFaceModel(...)) as model:
    result = model.generate(prompt)
# Automatically cleaned up
```

## Error Handling

```python
# Specific exceptions
from ragicamp.core.exceptions import ModelLoadError, EvaluationError

try:
    model = load_model(path)
except FileNotFoundError:
    raise ModelLoadError(f"Model not found: {path}")

# Let unexpected errors propagate - don't silence them
```

## Testing

- **Unit tests** - For individual components
- **Skip GPU tests** - Use `@pytest.mark.skipif` for GPU-dependent tests
- **Mock external services** - Don't hit real OpenAI API in tests

```python
@pytest.fixture
def mock_model():
    model = Mock(spec=LanguageModel)
    model.generate.return_value = "answer"
    return model
```

## File Organization

```
src/ragicamp/
├── agents/          # RAG agents (base, direct_llm, fixed_rag)
├── models/          # LLM backends (base, huggingface, openai)
├── retrievers/      # Dense/Sparse retrieval
├── datasets/        # QA datasets (base, nq, triviaqa, hotpotqa)
├── metrics/         # Evaluation metrics (F1, EM, BERTScore, LLM-judge)
├── corpus/          # Document corpus and chunking
├── evaluation/      # Evaluator class (two-phase evaluation)
├── experiment.py    # Experiment + ExperimentCallbacks + ExperimentResult
├── factory.py       # ComponentFactory with plugin registration
├── cli/             # Command-line interface (main.py, study.py)
├── core/            # Logging, exceptions (3 classes), protocols
├── config/          # Pydantic schemas only
└── utils/           # ResourceManager, paths, prompts, formatting

scripts/
└── experiments/     # Standalone experiment scripts

conf/
├── study/           # Study configurations (comprehensive_baseline, etc.)
├── prompts/         # Few-shot examples
└── retriever/       # Retriever configs (pre-built indexes)
```

## Prompt Engineering

### Key Principles

1. **Explicit stop instructions** - Tell the model to give ONE answer only
2. **Use "Question/Answer" format** - Avoid "Q:/A:" which models continue
3. **Stop sequences in code** - Truncate at `\nQuestion:` etc.
4. **Short, concrete examples** - Show exact answer format expected

### What NOT to do in prompts

```
# BAD: Model will continue generating Q&A pairs
Q: when did ww1 end
A: 1918

Q: who was president
A: Wilson
...

# GOOD: Explicit instruction to stop
Question: when did ww1 end
Answer: 1918
```

### Current prompt structure

```python
# Direct (no context)
"{style}
{stop_instruction}

{examples}
Question: {question}
Answer:"

# RAG (with context)  
"Use the context to answer. {style}
{stop_instruction}

{examples}
Context: {context}

Question: {query}
Answer:"
```

Few-shot examples are in `conf/prompts/fewshot_examples.yaml`.

## Adding a New Feature

1. **Ask: Is this core functionality?**
   - Yes → Add to `src/ragicamp/`
   - No → Add as a script or separate tool

2. **Ask: Does this need a new dependency?**
   - If possible, use existing dependencies
   - If new dep needed, make it optional

3. **Ask: Can users do this themselves easily?**
   - Yes → Don't build it, document how
   - No → Build the minimum viable version

4. **Implementation:**
   - Write the simplest thing that works
   - Add a test
   - Add a docstring with example
   - Update CHANGELOG

## Running Experiments

### Quick test
```bash
uv run python scripts/experiments/run_study.py conf/study/simple_hf.yaml --dry-run
```

### Full study
```bash
uv run python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --skip-existing
```

### With MLflow tracking
```bash
uv run python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --mlflow
```

## Analyzing Results

### CLI comparison
```bash
# Basic comparison by model
uv run ragicamp compare outputs/comprehensive_baseline

# Compare by different dimensions
uv run ragicamp compare outputs/comprehensive_baseline --group-by retriever

# Pivot table: model performance across datasets
uv run ragicamp compare outputs/comprehensive_baseline --pivot model dataset

# Export to CSV
uv run python scripts/analysis/compare_baseline.py outputs/comprehensive_baseline --csv results.csv
```

### Python API
```python
from ragicamp.analysis import ResultsLoader, compare_results, best_by, pivot_results

# Load results
loader = ResultsLoader("outputs/comprehensive_baseline")
results = loader.load_all()

# Compare by model
stats = compare_results(results, group_by="model", metric="f1")

# Get top 10 by exact match
top = best_by(results, metric="exact_match", n=10)

# Pivot table: rows=model, cols=dataset
pivot = pivot_results(results, rows="model", cols="dataset", metric="f1")
```

### MLflow tracking
```python
from ragicamp.analysis import MLflowTracker

# Backfill existing results
tracker = MLflowTracker("my_study")
tracker.backfill_from_results(results)

# View in MLflow UI
# $ mlflow ui
# Open http://localhost:5000
```

### Using Python API
```python
from ragicamp import Experiment
from ragicamp.agents import DirectLLMAgent
from ragicamp.models import HuggingFaceModel
from ragicamp.datasets import NaturalQuestionsDataset
from ragicamp.metrics import F1Metric, ExactMatchMetric

model = HuggingFaceModel("google/gemma-2b-it", load_in_4bit=True)
agent = DirectLLMAgent("baseline", model)
dataset = NaturalQuestionsDataset(split="validation")

exp = Experiment(
    name="my_exp",
    agent=agent,
    dataset=dataset,
    metrics=[F1Metric(), ExactMatchMetric()],
)

result = exp.run(batch_size=8)
print(f"F1: {result.f1:.3f}")
```

## Common Issues

### OOM (Out of Memory)
- Reduce `batch_size`
- Use `load_in_4bit=True`
- Clear cache between experiments: `ResourceManager.clear_gpu_memory()`

### Slow experiments
- Use batch processing: `batch_size=8`
- Use `--skip-existing` to resume
- Reduce `num_questions` for testing

### Config errors
- Run with `--dry-run` first
- Check model spec format: `hf:model/name` or `openai:model-name`

