# Troubleshooting Guide

Common issues and solutions for RAGiCamp.

## Quick Fixes

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` on save | **Fixed automatically** - Uses `ensure_dir()` |
| Matplotlib backend error | **Fixed automatically** - Sets backend to `'Agg'` |
| CUDA/GPU not found | Use CPU configs or set `device: cpu` |
| Out of memory | Use `load_in_8bit: true` or reduce batch size |

See detailed guides: [Matplotlib Issues](guides/TROUBLESHOOTING_MATPLOTLIB.md)

---

## NumPy Version Compatibility

**Symptom:**
```
AttributeError: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Cause:**  
TensorFlow and some other dependencies were compiled with NumPy 1.x and are incompatible with NumPy 2.x.

**Solution:**  
The dependencies in `pyproject.toml` are already pinned to `numpy>=1.21.0,<2.0.0`. If you still see this issue:

```bash
# Resync dependencies
uv sync

# Or manually downgrade numpy
uv pip install "numpy<2.0.0"
```

**Verify the fix:**
```bash
uv run python -c "import numpy; print(numpy.__version__)"
# Should show 1.x.x, not 2.x.x
```

---

## Build Errors

### Error: "Dependency cannot be a direct reference"

**Full error**:
```
ValueError: Dependency #2 of option `metrics` of field `project.optional-dependencies`
cannot be a direct reference unless field `tool.hatch.metadata.allow-direct-references` is
set to `true`
```

**Cause**: Hatchling build backend doesn't allow git-based dependencies by default.

**Solution**: Already fixed in latest version. If you see this:
```bash
git pull  # Get latest code
uv sync   # Re-sync dependencies
```

The fix adds this to `pyproject.toml`:
```toml
[tool.hatch.metadata]
allow-direct-references = true
```

### Error: "Failed to build ragicamp"

**Solution**:
```bash
# Clear UV cache and rebuild
rm -rf .venv
uv sync
```

## Metrics Errors

### Error: "No module named 'bert_score'"

**Cause**: BERTScore not installed.

**Solution**:
```bash
uv sync --extra metrics
```

### Error: "No module named 'bleurt'"

**Cause**: BLEURT not installed.

**Solution**:
```bash
uv sync --extra metrics
```

**Note**: BLEURT installation from git may take a few minutes.

### Error: "Failed to load BLEURT checkpoint"

**Cause**: Checkpoint download failed or wrong checkpoint name.

**Solution**:
```bash
# Use default checkpoint
--bleurt-checkpoint BLEURT-20

# Check internet connection - checkpoint is ~1.5GB
# Wait for download to complete
```

### BERTScore: "CUDA out of memory"

**Solutions**:
```bash
# 1. Use smaller model
--bertscore-model microsoft/deberta-base-mnli

# 2. Reduce batch size (automatic)
# 3. Use CPU (slower but works)
--device cpu

# 4. Reduce dataset size
--num-examples 10
```

## Dataset Errors

### Error: "Failed to load dataset"

**Cause**: Network issue or HuggingFace datasets not properly installed.

**Solution**:
```bash
# Check internet connection
ping huggingface.co

# Re-install dependencies
uv sync

# Try again with a smaller dataset first
--num-examples 10
```

### Warning: "Filtered out N examples without explicit answers"

**This is normal!** It means some questions don't have ground-truth answers.

**To disable filtering**:
```bash
# Remove --filter-no-answer flag
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100
```

## Model Errors

### Error: "You need to accept the Gemma license"

**Solution**:
1. Visit: https://huggingface.co/google/gemma-2-2b-it
2. Click "Agree and access repository"
3. Login: `uv run huggingface-cli login`

### Error: "We couldn't connect to huggingface.co"

**Cause**: Network issue or not logged in.

**Solutions**:
```bash
# 1. Check internet
ping huggingface.co

# 2. Login to HuggingFace
uv run huggingface-cli login

# 3. Check if you accepted the license (see above)
```

### Error: "CUDA out of memory"

**Solutions**:
```bash
# 1. Use 8-bit quantization
--load-in-8bit

# 2. Use CPU
--device cpu

# 3. Reduce batch size (automatic)

# 4. Reduce dataset size
--num-examples 10
```

## Performance Issues

### Evaluation is very slow

**Solutions**:

For GPU:
```bash
# Check GPU is being used
--device cuda

# Use smaller model
--load-in-8bit

# Reduce dataset size
--num-examples 10
```

For CPU:
```bash
# CPU is ~10x slower, this is expected
# Use very small dataset for testing
--num-examples 5
```

### BLEURT is extremely slow

**This is normal!** BLEURT is the slowest metric.

**Solutions**:
```bash
# 1. Use fewer examples
--num-examples 10

# 2. Use only BERTScore instead
--metrics exact_match f1 bertscore

# 3. Remove BLEURT for quick iteration
--metrics exact_match f1
```

## UV/Package Issues

### Error: "uv: command not found"

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (follow instructions from install script)
```

### Error: "Package not found in virtual environment"

**Solution**:
```bash
# Re-sync dependencies
uv sync

# Or with extras
uv sync --extra metrics
```

## Import Errors

### Error: "No module named 'ragicamp'"

**Cause**: Not running with `uv run` or virtual environment not activated.

**Solution**:
```bash
# Always use uv run
uv run python script.py

# Or activate venv manually
source .venv/bin/activate
python script.py
```

### Error: "ModuleNotFoundError: No module named 'X'"

**Solutions**:
```bash
# Re-sync all dependencies
uv sync

# Install specific extras
uv sync --extra metrics --extra viz

# Check what's installed
uv pip list
```

## Make Command Errors

### Error: "make: command not found"

**Cause**: Make not installed (rare on Linux/Mac, common on Windows).

**Solution**:

On Linux/Mac:
```bash
# Install make (usually pre-installed)
sudo apt-get install make  # Ubuntu/Debian
brew install make          # macOS
```

On Windows:
```bash
# Use the commands directly instead of make
uv run python experiments/scripts/run_gemma2b_baseline.py --dataset natural_questions --num-examples 10
```

## Git Issues

### Error: "Signing failed"

**Solution**: Use `--no-gpg-sign` flag
```bash
git commit --no-gpg-sign -m "message"
```

Or disable signing globally:
```bash
git config --global commit.gpgsign false
```

## General Tips

### Quick Diagnosis

```bash
# Check UV
uv --version

# Check Python
uv run python --version

# Check installed packages
uv pip list | grep ragicamp
uv pip list | grep bert-score
uv pip list | grep bleurt

# Check if in right directory
pwd
# Should show: .../ragicamp

# Check git status
git status
```

### Start Fresh

If nothing works, start from scratch:
```bash
# Remove virtual environment
rm -rf .venv

# Remove UV cache
rm -rf ~/.cache/uv/

# Re-install
uv sync

# Test basic import
uv run python -c "import ragicamp; print('✓ ragicamp works')"
```

### Get Help

1. Check the documentation:
   - `QUICK_REFERENCE.md` - Quick commands
   - `METRICS_GUIDE.md` - Metrics details
   - `GEMMA2B_QUICKSTART.md` - Gemma setup

2. Check error messages carefully - they usually tell you exactly what's wrong

3. Search for the error message online

4. Try the simplest possible command first:
```bash
uv run python -c "print('hello')"
```

## Common Gotchas

1. **Always use `uv run`** - Don't just run `python`, use `uv run python`

2. **Install metrics extras** - BERTScore and BLEURT need: `uv sync --extra metrics`

3. **Accept Gemma license** - Required for first-time use

4. **Network required** - For downloading models and datasets

5. **GPU vs CPU** - GPU is 10x faster but requires CUDA

6. **Filtering is default** - Use `--filter-no-answer` to only evaluate questions with answers

7. **BLEURT is slow** - This is normal, use fewer examples or skip BLEURT for quick iteration

## Still Having Issues?

Check these files for more info:
- `README.md` - Project overview
- `GETTING_STARTED.md` - Setup guide
- `USAGE.md` - Detailed usage
- `ARCHITECTURE.md` - System design

Or try the simplest possible workflow:
```bash
# 1. Fresh start
cd ragicamp
rm -rf .venv
uv sync

# 2. Test import
uv run python -c "from ragicamp.agents.direct_llm import DirectLLMAgent; print('✓ Works')"

# 3. Run tiny test
uv run python experiments/scripts/run_gemma2b_baseline.py --dataset natural_questions --num-examples 2 --device cpu
```

