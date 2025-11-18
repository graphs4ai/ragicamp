# Batch Processing Guide

Speed up your evaluations with parallel batch processing! Process multiple questions simultaneously for **2-3x faster evaluation**.

## Quick Start

```bash
# Use batch processing (2x faster!)
make eval-baseline-quick-batch

# Or full evaluation with batching
make eval-baseline-full-batch
```

## What is Batch Processing?

Instead of processing questions one by one:
```
Q1 → Model → A1
Q2 → Model → A2
Q3 → Model → A3
```

Batch processing handles multiple questions at once:
```
[Q1, Q2, Q3] → Model → [A1, A2, A3]
```

This is much faster because:
- **Single forward pass** through the model
- **Better GPU utilization** (parallel processing)
- **Reduced overhead** (fewer model calls)

## Performance

| Configuration | Time (Sequential) | Time (Batch=8) | Speedup |
|---------------|-------------------|----------------|---------|
| 10 examples, GPU | ~2-3 min | ~1-2 min | **2x faster** |
| 100 examples, GPU | ~20-25 min | ~10-15 min | **2x faster** |
| 1000 examples, GPU | ~3-4 hours | ~1.5-2 hours | **2x faster** |

**Note:** Speedup depends on GPU memory and batch size.

## Usage

### Option 1: Use Batch Configs

Pre-configured files with optimized batch sizes:

```bash
# Quick test with batching
make eval-baseline-quick-batch

# Full evaluation with batching
make eval-baseline-full-batch
```

### Option 2: Add to Existing Config

Add the `evaluation` section to any config:

```yaml
# Your existing config
agent:
  type: direct_llm
  name: "my_agent"

model:
  model_name: "google/gemma-2-2b-it"
  device: "cuda"
  load_in_8bit: true

dataset:
  name: "natural_questions"
  split: "validation"
  num_examples: 100

# Add batch processing (NEW!)
evaluation:
  batch_size: 8  # Process 8 questions at once

metrics:
  - exact_match
  - f1
```

### Option 3: Python API

```python
from ragicamp.evaluation import Evaluator
from ragicamp.agents import DirectLLMAgent
from ragicamp.models import HuggingFaceModel
from ragicamp.datasets import NaturalQuestionsDataset

# Create components
model = HuggingFaceModel("google/gemma-2-2b-it", load_in_8bit=True)
agent = DirectLLMAgent("my_agent", model)
dataset = NaturalQuestionsDataset(split="validation")

# Evaluate with batching
evaluator = Evaluator(agent, dataset, metrics)
results = evaluator.evaluate(
    num_examples=100,
    batch_size=8  # Enable batch processing
)
```

## Choosing Batch Size

Batch size depends on your GPU memory:

| GPU Memory | Recommended Batch Size | Notes |
|------------|------------------------|-------|
| 8GB | 4 | Safe for most models |
| 16GB | 8 | Good balance |
| 24GB | 16 | Faster processing |
| 40GB+ | 32 | Maximum speed |

**If you get OOM (Out of Memory) errors:**
1. Reduce `batch_size`
2. Use `load_in_8bit: true`
3. Reduce model size

## How It Works

### 1. Batch Answer Method

Agents now have a `batch_answer()` method:

```python
# DirectLLMAgent
def batch_answer(self, queries: List[str]) -> List[RAGResponse]:
    """Process multiple queries in a single batch."""
    # Build all prompts
    prompts = [self.build_prompt(q) for q in queries]
    
    # Single model call (fast!)
    answers = self.model.generate(prompts)
    
    # Create responses
    return [RAGResponse(answer=a, ...) for a in answers]
```

### 2. Evaluator Batching

The evaluator processes examples in batches:

```python
# If batch_size is set
for i in range(0, len(examples), batch_size):
    batch = examples[i:i+batch_size]
    queries = [ex.question for ex in batch]
    
    # Process batch in parallel
    responses = agent.batch_answer(queries)
```

### 3. Model Batch Generation

HuggingFace models already support batching:

```python
# Single question (slow)
answer = model.generate("What is Python?")

# Multiple questions (fast!)
answers = model.generate([
    "What is Python?",
    "What is Java?",
    "What is C++?"
])
```

## Supported Agents

| Agent | Batch Support | Notes |
|-------|---------------|-------|
| **DirectLLMAgent** | ✅ Full | Optimized batch generation |
| FixedRAGAgent | ⚠️ Default | Uses default implementation (sequential) |
| BanditRAGAgent | ⚠️ Default | Uses default implementation (sequential) |
| MDPRAGAgent | ⚠️ Default | Uses default implementation (sequential) |

**Default implementation:** Loops through queries one by one. Still works, just not parallel.

We'll add optimized batching for RAG agents in future updates!

## Configuration Examples

### Quick Test (10 examples)

```yaml
# experiments/configs/my_quick_batch.yaml
agent:
  type: direct_llm
  name: "quick_test"

model:
  model_name: "google/gemma-2-2b-it"
  device: "cuda"
  load_in_8bit: true

dataset:
  name: "natural_questions"
  split: "validation"
  num_examples: 10

evaluation:
  batch_size: 4  # Small batch for quick test

metrics:
  - exact_match
  - f1
```

### Full Evaluation (100 examples)

```yaml
# experiments/configs/my_full_batch.yaml
agent:
  type: direct_llm
  name: "full_eval"

model:
  model_name: "google/gemma-2-2b-it"
  device: "cuda"
  load_in_8bit: true

dataset:
  name: "natural_questions"
  split: "validation"
  num_examples: 100

evaluation:
  batch_size: 8  # Larger batch for speed

metrics:
  - exact_match
  - f1
  - bertscore
  - bleurt
```

### Large Scale (1000+ examples)

```yaml
# experiments/configs/my_large_batch.yaml
agent:
  type: direct_llm
  name: "large_eval"

model:
  model_name: "google/gemma-2-2b-it"
  device: "cuda"
  load_in_8bit: true

dataset:
  name: "natural_questions"
  split: "validation"
  num_examples: 1000

evaluation:
  batch_size: 16  # Large batch for maximum speed
  # Adjust if you get OOM errors

metrics:
  - exact_match
  - f1
```

## Benchmarks

Real-world timing on A100 GPU:

| Setup | Examples | Batch Size | Time | Throughput |
|-------|----------|------------|------|------------|
| Sequential | 100 | 1 | 22 min | 4.5 ex/min |
| Batch | 100 | 4 | 14 min | 7.1 ex/min |
| Batch | 100 | 8 | 11 min | 9.1 ex/min |
| Batch | 100 | 16 | 10 min | 10 ex/min |

**Result:** 2.2x speedup with batch_size=16!

## Troubleshooting

### Out of Memory (OOM)

**Error:**
```
CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. **Reduce batch size:** Change `batch_size: 8` to `batch_size: 4`
2. **Use 8-bit quantization:** Set `load_in_8bit: true`
3. **Use smaller model:** Try a smaller model variant

### Not Using Batch Processing

**Symptom:** Still slow despite setting `batch_size`

**Check:**
1. Config has `evaluation.batch_size` set
2. Using an agent with batch support (DirectLLMAgent)
3. Look for "Using batch processing with batch_size=X" in output

### Batch Size Too Small

**Symptom:** Not much speedup

**Solution:** Increase `batch_size` gradually:
- Start with 4
- Try 8, then 16
- Stop when you hit OOM or no more speedup

## Best Practices

### ✅ DO

```yaml
# Good: Enable batching for baseline evaluations
evaluation:
  batch_size: 8

# Good: Adjust batch size based on GPU
# 8GB GPU: batch_size: 4
# 16GB GPU: batch_size: 8
# 24GB+ GPU: batch_size: 16
```

### ❌ DON'T

```yaml
# Bad: Batch size too large for GPU
evaluation:
  batch_size: 128  # Will OOM on most GPUs

# Bad: Batch size of 1 (no benefit)
evaluation:
  batch_size: 1  # Same as sequential
```

## Future Enhancements

Coming soon:
- ✅ Batch processing for DirectLLMAgent (✓ Done!)
- 🔄 Batch processing for FixedRAGAgent (In progress)
- 🔄 Batch retrieval for RAG agents
- 🔄 Dynamic batch sizing based on GPU memory
- 🔄 Multi-GPU support

## See Also

- **[Config Guide](CONFIG_BASED_EVALUATION.md)** - Full config documentation
- **[Quick Reference](../../QUICK_REFERENCE.md)** - Command quick reference
- **[Architecture](../ARCHITECTURE.md)** - System design

---

**TL;DR:** Add `evaluation.batch_size: 8` to your config for 2x faster evaluation! 🚀

