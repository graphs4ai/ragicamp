# Output Structure

## Overview

RAGiCamp uses a **clean, modular 3-file structure** for evaluation results:

1. **`{dataset}_questions.json`** - Dataset questions and expected answers (reusable)
2. **`{agent}_predictions.json`** - Predictions with per-question metrics (one per agent/run)
3. **`{agent}_summary.json`** - Overall metrics summary with statistics

This structure is:
- ✅ **Modular** - Data and results are separated
- ✅ **Reusable** - Questions file shared across runs
- ✅ **Async-friendly** - Each agent can write independently
- ✅ **Easy to maintain** - Clear separation of concerns
- ✅ **Easy to expand** - Add new agents without conflicts

---

## File Descriptions

### 1. `{dataset}_questions.json`

**Purpose:** Contains dataset questions and expected answers. Reusable across all runs on this dataset.

**Location:** `outputs/{dataset}_questions.json`

**Format:**
```json
{
  "dataset_name": "natural_questions",
  "num_questions": 100,
  "questions": [
    {
      "id": "0",
      "question": "when was the last time anyone was on the moon",
      "expected_answer": "14 December 1972 UTC",
      "all_acceptable_answers": [
        "14 December 1972 UTC",
        "December 1972",
        "1972"
      ]
    },
    ...
  ]
}
```

**Fields:**
- `dataset_name`: Name of the dataset
- `num_questions`: Total number of questions
- `questions`: Array of question objects
  - `id`: Question ID
  - `question`: Question text
  - `expected_answer`: Primary expected answer
  - `all_acceptable_answers`: All acceptable answer variants

**Usage:**
- Created once per dataset evaluation
- Shared across all agent runs
- No need to re-save for each agent
- Can be used for analysis without predictions

---

### 2. `{agent}_predictions.json`

**Purpose:** Contains predictions and per-question metrics for a specific agent.

**Location:** `outputs/{agent}_predictions.json`

**Format:**
```json
{
  "agent_name": "gemma_2b_baseline",
  "dataset_name": "natural_questions",
  "timestamp": "2024-10-30T15:30:45.123456",
  "num_examples": 100,
  "predictions": [
    {
      "question_id": "0",
      "question": "when was the last time anyone was on the moon",
      "prediction": "December 1972",
      "metrics": {
        "exact_match": 1.0,
        "f1": 0.8571,
        "bertscore_precision": 0.8234,
        "bertscore_recall": 0.8567,
        "bertscore_f1": 0.8397
      },
      "metadata": {}
    },
    ...
  ]
}
```

**Fields:**
- `agent_name`: Name of the agent
- `dataset_name`: Dataset used
- `timestamp`: When evaluation was run
- `num_examples`: Number of examples evaluated
- `predictions`: Array of prediction objects
  - `question_id`: Links to question in `{dataset}_questions.json`
  - `question`: Question text (for convenience)
  - `prediction`: Agent's predicted answer
  - `metrics`: All metrics computed for this prediction
  - `metadata`: Additional agent-specific metadata

**Benefits:**
- One file per agent/run - **async-friendly!**
- Can run multiple agents in parallel
- Per-question metrics included directly
- Easy to compare across agents

---

### 3. `{agent}_summary.json`

**Purpose:** Overall metrics summary with statistics across all questions.

**Location:** `outputs/{agent}_summary.json`

**Format:**
```json
{
  "agent_name": "gemma_2b_baseline",
  "dataset_name": "natural_questions",
  "timestamp": "2024-10-30T15:30:45.123456",
  "num_examples": 100,
  "overall_metrics": {
    "exact_match": 0.6667,
    "f1": 0.6190,
    "bertscore_f1": 0.7630
  },
  "metric_statistics": {
    "exact_match": {
      "mean": 0.6667,
      "min": 0.0,
      "max": 1.0,
      "std": 0.4714
    },
    "f1": {
      "mean": 0.6190,
      "min": 0.0,
      "max": 1.0,
      "std": 0.4055
    },
    "bertscore_f1": {
      "mean": 0.7630,
      "min": 0.4615,
      "max": 0.9878,
      "std": 0.2151
    }
  }
}
```

**Fields:**
- `agent_name`: Name of the agent
- `dataset_name`: Dataset used
- `timestamp`: When evaluation was run
- `num_examples`: Number of examples evaluated
- `overall_metrics`: Mean scores for each metric
- `metric_statistics`: Statistics for each metric
  - `mean`: Average score across all questions
  - `min`: Minimum score
  - `max`: Maximum score
  - `std`: Standard deviation

**Benefits:**
- Quick overview of performance
- Statistics show score distribution
- Easy to compare agents at a glance
- Publication-ready numbers

---

## Comparison with Old Structure

### Old Structure (5 files) ❌

```
outputs/
  gemma2b_baseline_results.json       # Full results
  gemma2b_baseline_results_metrics.json  # Metrics summary
  gemma2b_baseline_results_metrics.txt   # Text summary
  gemma2b_baseline_results_per_question.json  # Per-question
  gemma2b_baseline_results_per_question.csv  # CSV
```

**Problems:**
- Too many files (5 per run)
- Redundant information
- Hard to maintain
- Not async-friendly (single output path)
- Mixed data (questions + predictions)

### New Structure (3 files) ✅

```
outputs/
  natural_questions_questions.json    # Questions (reusable)
  gemma_2b_baseline_predictions.json  # Predictions + metrics
  gemma_2b_baseline_summary.json      # Summary + stats
```

**Benefits:**
- Clean separation of concerns
- Reusable questions file
- One predictions file per agent
- Async-friendly
- Easy to expand
- Less redundancy

---

## Usage Examples

### Running Evaluation

```bash
# Run evaluation (same command as before)
make run-gemma2b-full

# Creates:
# - outputs/natural_questions_questions.json
# - outputs/gemma_2b_baseline_predictions.json
# - outputs/gemma_2b_baseline_summary.json
```

### Loading Results in Python

```python
import json

# Load questions (reusable)
with open('outputs/natural_questions_questions.json') as f:
    questions = json.load(f)

# Load agent predictions
with open('outputs/gemma_2b_baseline_predictions.json') as f:
    predictions = json.load(f)

# Load summary
with open('outputs/gemma_2b_baseline_summary.json') as f:
    summary = json.load(f)

# Quick metrics check
print(f"Overall F1: {summary['overall_metrics']['f1']:.4f}")
print(f"F1 range: {summary['metric_statistics']['f1']['min']:.2f} - "
      f"{summary['metric_statistics']['f1']['max']:.2f}")

# Analyze per-question
for pred in predictions['predictions']:
    if pred['metrics']['f1'] < 0.3:  # Find poor predictions
        print(f"Low F1: {pred['question']}")
        print(f"  Predicted: {pred['prediction']}")
```

### Comparing Multiple Agents

```python
import json
import pandas as pd

# Load summaries for multiple agents
agents = ['gemma_2b_baseline', 'gpt4_baseline', 'rag_model']
summaries = []

for agent in agents:
    with open(f'outputs/{agent}_summary.json') as f:
        summaries.append(json.load(f))

# Create comparison table
df = pd.DataFrame([
    {
        'Agent': s['agent_name'],
        'Exact Match': s['overall_metrics']['exact_match'],
        'F1': s['overall_metrics']['f1'],
        'BERTScore': s['overall_metrics'].get('bertscore_f1', 0)
    }
    for s in summaries
])

print(df)
```

### Async Evaluation

Since each agent writes to its own files, you can run multiple evaluations in parallel:

```python
import asyncio
from ragicamp.evaluation import Evaluator

async def evaluate_agent(agent, dataset):
    """Evaluate one agent (async-safe)."""
    evaluator = Evaluator(
        agent=agent,
        dataset=dataset,
        metrics=metrics,
        output_path=f"outputs/{agent.name}_results.json"
    )
    return await evaluator.evaluate_async()

# Run multiple agents in parallel
agents = [agent1, agent2, agent3]
results = await asyncio.gather(*[
    evaluate_agent(agent, dataset) for agent in agents
])

# Each creates its own files:
# - agent1_predictions.json, agent1_summary.json
# - agent2_predictions.json, agent2_summary.json
# - agent3_predictions.json, agent3_summary.json
# All share: natural_questions_questions.json
```

---

## Migration from Old Format

If you have old results, you can convert them:

```python
import json

# Load old format
with open('outputs/gemma2b_baseline_results.json') as f:
    old_data = json.load(f)

# Convert to new format
# (See examples/convert_old_format.py for full script)
```

---

## Benefits Summary

✅ **Modular**: Data and results separated  
✅ **Reusable**: Questions file shared across runs  
✅ **Async-friendly**: Independent files per agent  
✅ **Maintainable**: Clear structure, easy to update  
✅ **Expandable**: Add new agents without conflicts  
✅ **Efficient**: No redundancy, smaller files  
✅ **Analyzable**: Easy to load and compare  
✅ **Professional**: Publication-ready structure

---

## Example Files

See `examples/new_output_structure/` for:
- `natural_questions_questions.json`
- `gemma_2b_baseline_predictions.json`
- `gemma_2b_baseline_summary.json`

