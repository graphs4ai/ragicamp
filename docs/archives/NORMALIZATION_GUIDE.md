# Text Normalization for QA Metrics

## Overview

Text normalization is crucial for fair evaluation in question answering. Without it, trivial differences like capitalization or punctuation can cause incorrect scoring.

## What We Implement

### Standard Normalization (Default)

Following **SQuAD evaluation best practices**:

1. **Lowercase** - "Paris" → "paris"
2. **Remove articles** - "the capital" → "capital"
3. **Remove punctuation** - "Paris, France" → "paris france"
4. **Normalize whitespace** - "paris  france" → "paris france"

This is the **industry standard** used in most QA benchmarks.

### Optional: Stemming (Advanced)

For more aggressive normalization:
- "running" → "run"
- "cities" → "citi"
- "better" → "better" (not "good" - that's lemmatization)

**When to use:**
- When you want to be more lenient with word variations
- When testing models with known word form issues

**When NOT to use:**
- Standard benchmarking (not part of SQuAD/NQ official metrics)
- When word form matters (e.g., "running" vs "run")

## Usage

### Default (Recommended)

```python
from ragicamp.metrics import ExactMatchMetric, F1Metric

# Uses standard normalization (lowercase, articles, punctuation)
em = ExactMatchMetric()
f1 = F1Metric()
```

### With Stemming

```python
# Install NLTK first: pip install nltk
em = ExactMatchMetric(use_stemming=True)
f1 = F1Metric(use_stemming=True)
```

### No Normalization (Not Recommended)

```python
# Exact string matching - very strict!
em = ExactMatchMetric(normalize=False)
f1 = F1Metric(normalize=False)
```

## Examples

### Standard Normalization

```python
Prediction: "The capital of France is Paris."
Reference:  "paris"

Without normalization:
  Exact Match: 0.0 ❌
  F1:          0.0 ❌

With normalization (default):
  Both become: "capital of france is paris" vs "paris"
  Exact Match: 0.0 (not exact match)
  F1:          0.4 (1 token match out of 5 and 1)
```

### With Stemming

```python
Prediction: "The cities are running"
Reference:  "city is run"

Standard normalization:
  "cities are running" vs "city is run"
  F1: 0.0 (no common tokens)

With stemming:
  "citi are run" vs "citi is run"
  F1: 0.4 (2/3 tokens match)
```

## Best Practices

### ✅ DO

1. **Always use normalization** for QA evaluation
   ```python
   em = ExactMatchMetric()  # normalize=True by default
   ```

2. **Use standard normalization** for benchmarking
   ```python
   # This matches SQuAD, NaturalQuestions, TriviaQA
   em = ExactMatchMetric(use_stemming=False)
   ```

3. **Document which normalization** you use in papers
   ```
   "We follow SQuAD evaluation: lowercase, article removal, punctuation removal"
   ```

### ❌ DON'T

1. **Don't skip normalization** unless testing exact string matching
   
2. **Don't use stemming** for official benchmarks (not standard)

3. **Don't use different normalization** when comparing to published results

## Why These Choices?

### Lowercase
- "Paris" and "paris" are semantically identical
- Models shouldn't be penalized for capitalization

### Remove Articles
- "a", "an", "the" don't add semantic meaning for answers
- "the capital" and "capital" mean the same thing

### Remove Punctuation
- "Paris!" and "Paris" are the same answer
- Punctuation is often inconsistent

### Whitespace
- "paris  france" and "paris france" should match
- Tokenization artifact

### Why NOT Lemmatization by Default?

Lemmatization is more accurate than stemming but:
- **Slower** - requires POS tagging
- **Not standard** - SQuAD doesn't use it
- **Overkill** - normalization + stemming handles most cases
- **Dependencies** - requires language-specific resources

## Comparison: No Norm vs Standard vs Stemming

```python
Prediction: "The United States' capital is Washington, D.C."
Reference:  "washington"

No normalization:
  EM: 0.0  F1: 0.0  ❌ Too strict

Standard normalization (recommended):
  Pred: "united states capital is washington d c"
  Ref:  "washington"
  EM: 0.0  F1: 0.29  ✅ Partial credit for partial match

With stemming:
  Pred: "unit state capit is washington d c"
  Ref:  "washington"
  EM: 0.0  F1: 0.29  ✅ Same as standard (no stems in this case)
```

## When to Use Each

| Scenario | Normalization | Stemming | Reasoning |
|----------|--------------|----------|-----------|
| **Benchmarking** | ✅ Yes | ❌ No | Match published results |
| **Development** | ✅ Yes | ❌ No | Standard is sufficient |
| **Debugging** | ❌ No | ❌ No | See exact issues |
| **Lenient eval** | ✅ Yes | ✅ Yes | Accept more variations |
| **Multilingual** | ✅ Yes | ⚠️ Maybe | Language-specific |

## Implementation Details

Our implementation (`normalize_answer` function):

```python
def normalize_answer(text, 
                     lowercase=True,
                     remove_articles=True,
                     remove_punctuation=True,
                     remove_extra_whitespace=True,
                     stemmer=None):
    """Normalize following SQuAD practices."""
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    if remove_articles:
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
    if remove_extra_whitespace:
        text = ' '.join(text.split())
    if stemmer:
        tokens = text.split()
        tokens = [stemmer.stem(token) for token in tokens]
        text = ' '.join(tokens)
    return text.strip()
```

This matches the official SQuAD evaluation script.

## References

- **SQuAD**: https://rajpurkar.github.io/SQuAD-explorer/
- **Natural Questions**: https://ai.google.com/research/NaturalQuestions/
- **Best practices**: Normalization is standard in all major QA benchmarks

## Quick Start

For most users:
```python
# Just use defaults - they follow best practices!
from ragicamp.metrics import ExactMatchMetric, F1Metric

em = ExactMatchMetric()  # ✅ Normalized by default
f1 = F1Metric()          # ✅ Normalized by default
```

That's it! You're following best practices. 🎉
