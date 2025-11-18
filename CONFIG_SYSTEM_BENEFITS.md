# Config System Benefits

How the new Pydantic-based config system improves the entire repository.

---

## 🎯 Core Benefits

### 1. **Type Safety & Validation**

**Before:**
```python
# Raw dict access - no validation, easy to typo
device = config["model"]["device"]  # KeyError if missing!
batch_size = config.get("evaluation", {}).get("batch_size")  # Verbose
```

**After:**
```python
# Validated, typed config object
config = ConfigLoader.load_and_validate("config.yaml")
device = config.model.device  # ✓ Validated, autocomplete works
batch_size = config.evaluation.batch_size  # ✓ Type-safe
```

### 2. **Better Error Messages**

**Before:**
```
KeyError: 'model_name'
# Where? What was expected? No idea!
```

**After:**
```
❌ Configuration validation failed:

Errors:
  • model -> model_name: field required
  • agent -> type: field required
  • dataset -> name: field required

Please fix the configuration and try again.
```

### 3. **Default Values**

**Before:**
```python
# Scattered defaults throughout code
device = config.get("model", {}).get("device", "cuda")
batch_size = config.get("evaluation", {}).get("batch_size", None)
filter_no_answer = config.get("dataset", {}).get("filter_no_answer", True)
```

**After:**
```python
# Centralized defaults in schema
class ModelConfig(BaseModel):
    device: str = Field(default="cuda")  # Single source of truth
```

### 4. **IDE Support**

With Pydantic models, you get:
- ✅ **Autocomplete** for all config fields
- ✅ **Type checking** in IDE
- ✅ **Inline documentation** from Field descriptions
- ✅ **Go-to-definition** for config schemas

---

## 🚀 Where It Helps

### 1. **run_experiment.py** (Already Improved)

```python
# Before: 426 LOC with scattered dict access
# After: 295 LOC with validated config objects

from ragicamp.config import ConfigLoader

config = ConfigLoader.load_and_validate(args.config)
model = ComponentFactory.create_model(config.model)  # Type-safe!
```

### 2. **Makefile Commands**

Add config validation:
```makefile
validate-config:
	@echo "🔍 Validating configuration..."
	uv run python -c "from ragicamp.config import ConfigLoader; \
		ConfigLoader.validate_file('$(CONFIG)')"

validate-all-configs:
	@echo "🔍 Validating all configs..."
	@for f in experiments/configs/*.yaml; do \
		uv run python -c "from ragicamp.config import ConfigLoader; \
			ConfigLoader.validate_file('$$f')" || exit 1; \
	done
	@echo "✓ All configs valid!"
```

### 3. **CLI Tool for Config Management**

```bash
# Validate a config
ragicamp validate config.yaml

# Create template
ragicamp create-config my_experiment.yaml

# Show config summary
ragicamp show-config config.yaml

# Merge configs
ragicamp merge base.yaml overrides.yaml -o final.yaml
```

### 4. **Testing**

Much easier to test with validated configs:

```python
def test_experiment_with_valid_config():
    config = ExperimentConfig(
        agent=AgentConfig(type="direct_llm", name="test"),
        model=ModelConfig(model_name="gpt-2"),
        dataset=DatasetConfig(name="natural_questions"),
        metrics=["exact_match", "f1"]
    )
    # Guaranteed to be valid!
    assert config.model.device == "cuda"  # Default value
```

### 5. **Documentation Auto-Generation**

Generate docs from schemas:

```python
# Generate markdown docs
from ragicamp.config.schemas import ExperimentConfig

schema = ExperimentConfig.schema()
# Now generate docs from schema['properties']
```

### 6. **Compare Baseline Script**

```python
# Before: Hardcoded configs
# After: Load validated configs

from ragicamp.config import ConfigLoader

config1 = ConfigLoader.load_and_validate("baseline1.yaml")
config2 = ConfigLoader.load_and_validate("baseline2.yaml")

# Compare with type safety
print(f"Model 1: {config1.model.model_name}")
print(f"Model 2: {config2.model.model_name}")
```

### 7. **Dataset Download Script**

```python
# Standardize dataset config access
from ragicamp.config.schemas import DatasetConfig

dataset_config = DatasetConfig(**config["dataset"])
# Validated: name, split, filter_no_answer all correct types
```

### 8. **Error Recovery**

```python
# Provide helpful suggestions
try:
    config = ConfigLoader.load_and_validate("config.yaml")
except ValidationError as e:
    # Show what's wrong + suggest fixes
    # e.g., "Did you mean 'model_name' instead of 'modelname'?"
```

---

## 📊 Improvements Across Repo

### Code Quality
```
Before: Dict[str, Any] everywhere
After:  Typed config objects
        ↓
Result: ✓ Type safety
        ✓ Better IDE support
        ✓ Fewer runtime errors
```

### User Experience
```
Before: Cryptic KeyError
After:  "field required: model.model_name"
        ↓
Result: ✓ Clear error messages
        ✓ Faster debugging
        ✓ Self-documenting
```

### Maintainability
```
Before: Defaults scattered across 10+ files
After:  Centralized in schemas.py
        ↓
Result: ✓ Single source of truth
        ✓ Easy to update
        ✓ Consistent behavior
```

---

## 🔧 Specific Use Cases

### Use Case 1: Experiment Tracking

```python
# Save validated config with results
config = ConfigLoader.load_and_validate("config.yaml")
results = run_experiment(config)

# Save for reproducibility
output = {
    "config": config.dict(),  # Pydantic -> dict
    "results": results,
    "timestamp": datetime.now()
}
```

### Use Case 2: Config Templates

```python
# Create templates for common experiments
from ragicamp.config import create_config_template

create_config_template("baseline_template.yaml")
create_config_template("rag_template.yaml")
```

### Use Case 3: Programmatic Config Creation

```python
# Create configs in code (not just YAML)
from ragicamp.config.schemas import ExperimentConfig, ModelConfig, DatasetConfig

config = ExperimentConfig(
    agent=AgentConfig(type="direct_llm", name="test"),
    model=ModelConfig(
        type="huggingface",
        model_name="google/gemma-2-2b-it",
        device="cuda"
    ),
    dataset=DatasetConfig(
        name="natural_questions",
        num_examples=10
    ),
    metrics=["exact_match", "f1"]
)

# Guaranteed valid!
run_experiment(config)
```

### Use Case 4: Config Inheritance

```python
# Load base config, override specific fields
base = ConfigLoader.load("base_config.yaml")
overrides = {"model": {"device": "cpu"}, "dataset": {"num_examples": 5}}

merged = ConfigLoader.merge_configs(base, overrides)
config = ConfigLoader.validate(merged)
```

### Use Case 5: Batch Experiments

```python
# Run multiple experiments with validated configs
configs = [
    ConfigLoader.load_and_validate(f"config_{i}.yaml")
    for i in range(10)
]

for i, config in enumerate(configs):
    print(f"Running experiment {i}: {config.agent.name}")
    run_experiment(config)
```

---

## 🎨 Future Enhancements

### 1. **Config Diffing**
```python
diff = ConfigLoader.diff(config1, config2)
# Show exactly what changed between runs
```

### 2. **Config Linting**
```python
warnings = ConfigLoader.lint(config)
# "Warning: batch_size=1 may be slow"
# "Suggestion: Use load_in_8bit=True to save memory"
```

### 3. **Config Versioning**
```python
class ExperimentConfigV2(ExperimentConfig):
    version: str = "2.0"
    # Add new fields while maintaining compatibility
```

### 4. **Config Search**
```python
# Find all configs using a specific model
configs = ConfigLoader.search_dir(
    "experiments/configs",
    filter_fn=lambda c: c.model.model_name == "gemma-2-2b-it"
)
```

### 5. **Auto-Fix Common Issues**
```python
config = ConfigLoader.load_and_validate(
    "config.yaml",
    auto_fix=True  # Fix common issues automatically
)
```

---

## 📝 Migration Guide

### For Existing Scripts

**Step 1:** Add config validation at the top
```python
from ragicamp.config import ConfigLoader

# Old
config = yaml.safe_load(open("config.yaml"))

# New  
config = ConfigLoader.load_and_validate("config.yaml")
```

**Step 2:** Use typed access
```python
# Old
model_name = config["model"]["model_name"]

# New
model_name = config.model.model_name
```

**Step 3:** Leverage defaults
```python
# Old (manual default handling)
device = config.get("model", {}).get("device", "cuda")

# New (automatic from schema)
device = config.model.device  # Always has a value
```

### For Existing Configs

No changes needed! YAML files remain the same:
```yaml
# This still works!
model:
  type: huggingface
  model_name: google/gemma-2-2b-it
  
# Now gets validated + defaults applied automatically
```

---

## ✅ Summary

The config system improves:

1. **Safety**: Type checking + validation
2. **UX**: Better errors + IDE support  
3. **Maintainability**: Centralized defaults
4. **Testing**: Easy to create test configs
5. **Documentation**: Self-documenting schemas
6. **Tools**: Config validation, templates, etc.
7. **Reliability**: Catch errors before running

**Impact:** Touches almost every part of the codebase in a positive way!

