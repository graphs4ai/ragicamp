# Matplotlib Backend Issues

## Problem

When running evaluations with BERTScore or BLEURT in non-interactive environments (scripts, Makefile), you might encounter:

```
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend
```

## Root Cause

The `MPLBACKEND` environment variable is set to a Jupyter/IPython-specific backend (`matplotlib_inline.backend_inline`), which is not available when running in regular Python scripts.

This happens when:
- Running in Google Colab
- Running after using Jupyter notebooks
- Environment variables from notebook sessions persist

## Solution

### Automatic Fix (Built-in)

RAGiCamp automatically handles this! The `bertscore.py` and `bleurt.py` metric modules automatically set the backend to `'Agg'` (non-interactive) when they detect the problematic environment variable.

**No action needed** - just run your evaluation:

```bash
make eval-baseline-full  # Works automatically
```

### Manual Fix (If Needed)

If you encounter this in other contexts:

```bash
# Option 1: Unset the variable
unset MPLBACKEND

# Option 2: Set to non-interactive backend
export MPLBACKEND=Agg

# Then run evaluation
make eval-baseline-full
```

### In Python Code

```python
import os

# Set matplotlib backend before any imports
os.environ['MPLBACKEND'] = 'Agg'

# Now import metrics
from ragicamp.metrics import BERTScoreMetric, BLEURTMetric
```

## Implementation

The fix is implemented at the module level in:
- `src/ragicamp/metrics/bertscore.py`
- `src/ragicamp/metrics/bleurt.py`

```python
# At the top of the module, before other imports
import os

if 'MPLBACKEND' in os.environ:
    # Change to non-interactive backend for scripts
    os.environ['MPLBACKEND'] = 'Agg'
```

This ensures the backend is set correctly **before** any matplotlib imports occur.

## Valid Matplotlib Backends

For non-interactive use (scripts):
- `Agg` - Anti-Grain Geometry (recommended for scripts)
- `Cairo` - Cairo graphics
- `PDF`, `PS`, `SVG` - Direct file output

For interactive use (notebooks):
- `notebook` - Jupyter notebook backend
- `nbagg` - Interactive figures in notebooks
- `module://matplotlib_inline.backend_inline` - IPython inline (Jupyter only)

## Testing

```bash
# This should now work without errors
make eval-baseline-full

# Or with Python
uv run python experiments/scripts/run_experiment.py \
    --config experiments/configs/nq_baseline_gemma2b_full.yaml \
    --mode eval
```

## See Also

- [BERTScore documentation](https://github.com/Tiiiger/bert_score)
- [Matplotlib backends](https://matplotlib.org/stable/users/explain/backends.html)
- [RAGiCamp Troubleshooting](../TROUBLESHOOTING.md)

---

**TL;DR:** The fix is automatic. Just run `make eval-baseline-full` and it works! ✅

