# Path Utilities Guide

Utilities to safely handle file paths and avoid `FileNotFoundError`.

## Problem

Python's `open()` and similar functions fail if the parent directory doesn't exist:

```python
# ❌ This fails if 'outputs/' doesn't exist
with open("outputs/results.json", "w") as f:
    json.dump(data, f)
# FileNotFoundError: [Errno 2] No such file or directory: 'outputs/results.json'
```

## Solution

RAGiCamp provides path utilities to handle this automatically.

## Quick Start

```python
from ragicamp.utils import ensure_dir, safe_write_json

# Option 1: Ensure directory exists first
ensure_dir("outputs/results.json")
with open("outputs/results.json", "w") as f:
    json.dump(data, f)

# Option 2: Use safe_write_json (recommended)
safe_write_json(data, "outputs/results.json", indent=2)
```

## Available Functions

### `ensure_dir(path)`

Ensures a directory exists, creating it if necessary.

```python
from ragicamp.utils import ensure_dir

# Works with file paths (creates parent directory)
ensure_dir("outputs/experiments/run1/results.json")
# Creates: outputs/experiments/run1/

# Works with directory paths
ensure_dir("artifacts/retrievers/wikipedia")
# Creates: artifacts/retrievers/wikipedia/
```

**Parameters:**
- `path` (str | Path): Path to file or directory

**Returns:**
- `Path`: The created/verified directory

**Features:**
- Creates all parent directories (`parents=True`)
- Doesn't fail if directory exists (`exist_ok=True`)
- Auto-detects files vs directories (by extension)

### `safe_write_json(data, path, **kwargs)`

Writes JSON to file, ensuring directory exists.

```python
from ragicamp.utils import safe_write_json

data = {"metric": "exact_match", "score": 0.85}

# Creates 'outputs/' if it doesn't exist, then writes JSON
safe_write_json(data, "outputs/results.json", indent=2)
```

**Parameters:**
- `data` (dict): Dictionary to write as JSON
- `path` (str | Path): Output file path
- `**kwargs`: Additional arguments for `json.dump()`

**Returns:**
- `Path`: Path to written file

### `ensure_output_dirs()`

Creates all standard RAGiCamp directories.

```python
from ragicamp.utils import ensure_output_dirs

# Creates all standard directories at once
ensure_output_dirs()
```

**Creates:**
- `outputs/`
- `outputs/experiments/`
- `outputs/comparisons/`
- `artifacts/`
- `artifacts/retrievers/`
- `artifacts/agents/`
- `data/`
- `data/datasets/`

**Use this** at the start of scripts to ensure all directories exist.

## Usage Examples

### Example 1: In Evaluation Scripts

```python
from ragicamp.utils import ensure_dir

def save_results(results, output_path):
    """Save evaluation results."""
    # Ensure output directory exists
    ensure_dir(output_path)
    
    # Now safe to write
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
```

### Example 2: In Experiment Setup

```python
from ragicamp.utils import ensure_output_dirs

def run_experiment(config):
    """Run an experiment."""
    # Ensure all standard directories exist
    ensure_output_dirs()
    
    # Now safe to use any output directory
    save_config(config, "outputs/experiments/exp1/config.yaml")
    save_results(results, "outputs/experiments/exp1/results.json")
```

### Example 3: Saving Multiple Files

```python
from ragicamp.utils import safe_write_json

# Save multiple related files
safe_write_json(config, "outputs/exp1/config.json", indent=2)
safe_write_json(results, "outputs/exp1/results.json", indent=2)
safe_write_json(metrics, "outputs/exp1/metrics.json", indent=2)

# All directories created automatically!
```

### Example 4: In Agent Save Methods

```python
from pathlib import Path
from ragicamp.utils import ensure_dir

class MyAgent(RAGAgent):
    def save(self, artifact_name: str) -> str:
        """Save agent configuration."""
        from ragicamp.utils.artifacts import get_artifact_manager
        
        manager = get_artifact_manager()
        path = manager.get_agent_path(artifact_name)
        
        # Path already ensured by manager, but to be explicit:
        ensure_dir(path / "config.json")
        
        with open(path / "config.json", 'w') as f:
            json.dump(self.get_config(), f, indent=2)
        
        return str(path)
```

## Comparison

### Before (Manual Directory Creation)

```python
import os
import json

output_path = "outputs/experiments/run1/results.json"

# Manual directory creation (verbose and error-prone)
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_path, 'w') as f:
    json.dump(data, f)
```

### After (With Utilities)

```python
from ragicamp.utils import safe_write_json

# One line, always works
safe_write_json(data, "outputs/experiments/run1/results.json", indent=2)
```

## When to Use Each Function

| Function | Use When |
|----------|----------|
| `ensure_dir()` | Writing files with standard `open()` |
| `safe_write_json()` | Writing JSON files (recommended) |
| `ensure_output_dirs()` | Script initialization |

## Integration with Existing Code

### In Evaluator

```python
from ragicamp.utils import ensure_dir

def _save_results(self, output_path):
    output_dir = Path(output_path).parent
    
    # Ensure directory exists
    ensure_dir(output_dir)
    
    # Now safe to write all files
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f)
```

### In OutputManager

The `OutputManager` already handles directory creation internally:

```python
from ragicamp.output import OutputManager

# OutputManager automatically creates directories
mgr = OutputManager()
exp_dir = mgr.create_experiment_dir("my_experiment")
# Creates: outputs/experiments/my_experiment/
```

### In ArtifactManager

Similarly, `ArtifactManager` handles directories:

```python
from ragicamp.utils.artifacts import get_artifact_manager

# ArtifactManager automatically creates directories
manager = get_artifact_manager()
path = manager.get_retriever_path("wikipedia_v1")
# Creates: artifacts/retrievers/wikipedia_v1/
```

## Best Practices

### ✅ DO

```python
from ragicamp.utils import ensure_dir, safe_write_json

# Use utilities for file operations
ensure_dir(output_path)
safe_write_json(data, output_path, indent=2)

# Call ensure_output_dirs() at script start
def main():
    ensure_output_dirs()
    # ... rest of script
```

### ❌ DON'T

```python
# Don't manually check and create directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Don't assume directories exist
with open("outputs/file.json", 'w') as f:  # ❌ May fail
    json.dump(data, f)
```

## Implementation Notes

The utilities use `Path.mkdir()` with:
- `parents=True` - Creates all parent directories
- `exist_ok=True` - Doesn't fail if directory exists

This is idempotent and thread-safe (in most cases).

## Example Script

See: `examples/path_utilities_example.py`

```bash
# Run the example
uv run python examples/path_utilities_example.py
```

## Testing

```python
import tempfile
from pathlib import Path
from ragicamp.utils import ensure_dir, safe_write_json

def test_ensure_dir():
    """Test ensure_dir creates directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with file path
        path = Path(tmpdir) / "deep" / "nested" / "file.txt"
        ensure_dir(path)
        assert path.parent.exists()
        
        # Test with directory path
        path = Path(tmpdir) / "another" / "dir"
        ensure_dir(path)
        assert path.exists()

def test_safe_write_json():
    """Test safe_write_json creates dirs and writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output" / "data.json"
        data = {"key": "value"}
        
        safe_write_json(data, path, indent=2)
        
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data
```

## See Also

- **[Quick Reference](../../QUICK_REFERENCE.md)** - Utility overview
- **[Architecture](../ARCHITECTURE.md)** - System design
- **Example**: `examples/path_utilities_example.py`

---

**Questions?** Check the main [README](../../README.md) or [TROUBLESHOOTING](../TROUBLESHOOTING.md) guide.

