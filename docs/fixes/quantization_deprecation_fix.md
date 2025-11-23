# Fix for Deprecation Warnings: `load_in_4bit` and `load_in_8bit`

**Date:** November 23, 2025  
**Issue:** Deprecation warnings when using `load_in_4bit` and `load_in_8bit` arguments

---

## 🐛 Problem

When using quantization in HuggingFace models, the following deprecation warning appeared:

```
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed 
in the future versions. Please, pass a `BitsAndBytesConfig` object in 
`quantization_config` argument instead.
```

---

## ✅ Solution

Updated the `HuggingFaceModel` class to use the new `BitsAndBytesConfig` approach instead of deprecated arguments.

### Changes Made

#### 1. Updated `src/ragicamp/models/huggingface.py`

**Before:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if load_in_8bit else None,
    load_in_8bit=load_in_8bit,  # ❌ Deprecated
    **kwargs,
)
```

**After:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Prepare quantization config if needed
quantization_config = None
use_quantization = load_in_8bit or load_in_4bit

if load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
elif load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if use_quantization else None,
    quantization_config=quantization_config,  # ✅ New approach
    **kwargs,
)
```

### Key Improvements

1. **No deprecation warnings**: Uses the new `quantization_config` parameter
2. **Support for both 4-bit and 8-bit**: Properly handles both quantization modes
3. **4-bit priority**: When both are specified, 4-bit takes precedence (more memory efficient)
4. **Backward compatible**: Existing code using `load_in_8bit` or `load_in_4bit` continues to work
5. **Best practices for 4-bit**: Uses recommended settings:
   - `bnb_4bit_compute_dtype=torch.float16` for computation
   - `bnb_4bit_use_double_quant=True` for better quality
   - `bnb_4bit_quant_type="nf4"` for Normal Float 4-bit quantization

---

## 🧪 Testing

### Added Tests

Created `tests/test_models.py` with comprehensive tests:

1. **test_8bit_quantization_config**: Verifies 8-bit uses `BitsAndBytesConfig`
2. **test_4bit_quantization_config**: Verifies 4-bit uses `BitsAndBytesConfig`
3. **test_no_quantization**: Verifies no quantization doesn't use config
4. **test_4bit_takes_precedence_over_8bit**: Verifies 4-bit priority

### Test Results

```bash
$ uv run pytest tests/ -v
============================= test session starts ==============================
...
tests/test_models.py::TestHuggingFaceModelQuantization::test_8bit_quantization_config PASSED
tests/test_models.py::TestHuggingFaceModelQuantization::test_4bit_quantization_config PASSED
tests/test_models.py::TestHuggingFaceModelQuantization::test_no_quantization PASSED
tests/test_models.py::TestHuggingFaceModelQuantization::test_4bit_takes_precedence_over_8bit PASSED
...
======================= 90 passed, 19 warnings in 7.13s ========================
```

All tests pass! ✅

---

## 📊 Impact

### Files Modified
- `src/ragicamp/models/huggingface.py` - Core implementation

### Files Added
- `tests/test_models.py` - Quantization tests

### Backward Compatibility
- ✅ **100% backward compatible** - All existing configs and code continue to work
- ✅ **No API changes** - The public interface remains the same
- ✅ **Config files unchanged** - YAML configs use the same `load_in_8bit` and `load_in_4bit` fields

---

## 💡 Usage Examples

### 8-bit Quantization (Config File)
```yaml
model:
  type: huggingface
  model_name: "google/gemma-2-2b-it"
  device: "cuda"
  load_in_8bit: true  # Still works!
```

### 4-bit Quantization (Programmatic)
```python
from ragicamp.models.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_name="google/gemma-2-2b-it",
    device="cuda",
    load_in_4bit=True  # Still works!
)
```

### No Quantization
```python
model = HuggingFaceModel(
    model_name="google/gemma-2-2b-it",
    device="cuda"
)
```

---

## 🔍 Verification

To verify no deprecation warnings are emitted:

```python
import warnings

warnings.simplefilter("always", DeprecationWarning)

from ragicamp.models.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_name="google/gemma-2-2b-it",
    device="cuda",
    load_in_8bit=True
)
# ✅ No warnings about load_in_8bit or load_in_4bit!
```

---

## 📚 References

- HuggingFace Transformers documentation on quantization: https://huggingface.co/docs/transformers/main_classes/quantization
- BitsAndBytes library: https://github.com/TimDettmers/bitsandbytes

---

## Summary

This fix eliminates deprecation warnings while maintaining full backward compatibility. Users can continue using `load_in_8bit` and `load_in_4bit` parameters, but internally the code now uses the modern `BitsAndBytesConfig` approach that HuggingFace Transformers expects. The fix is transparent to users and requires no changes to existing configurations or code.

