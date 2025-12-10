# Hydra Configuration System

RAGiCamp uses [Hydra](https://hydra.cc/) for flexible, composable experiment configuration.

## Why Hydra?

**Before (old YAML configs):**
```yaml
# nq_baseline_gemma2b_quick.yaml - 53 lines
# nq_baseline_gemma2b_full.yaml - 53 lines (only 2 values different!)
# nq_fixed_rag_gemma2b.yaml - 68 lines (mostly copy-paste)
```

**After (Hydra):**
```bash
# Same experiments, zero duplication:
python -m ragicamp.cli.run experiment=baseline evaluation=quick
python -m ragicamp.cli.run experiment=baseline evaluation=full
python -m ragicamp.cli.run experiment=rag
```

## 📁 Structure

```
conf/
├── config.yaml              # Default config (can be overridden)
├── model/                   # Model configurations (7 options)
│   ├── gemma_2b.yaml       
│   ├── gemma_2b_4bit.yaml
│   ├── gemma_2b_8bit.yaml
│   ├── phi3.yaml
│   ├── llama3_8b.yaml
│   ├── openai_gpt4.yaml
│   └── cpu.yaml
├── dataset/                 # Dataset configurations (3 options)
│   ├── nq.yaml              # Natural Questions
│   ├── triviaqa.yaml
│   └── hotpotqa.yaml
├── agent/                   # Agent configurations (3 options)
│   ├── direct_llm.yaml
│   ├── fixed_rag.yaml
│   └── bandit_rag.yaml
├── retriever/               # Retriever configurations
│   ├── dense.yaml
│   └── sparse.yaml
├── metrics/                 # Metric presets (4 options)
│   ├── fast.yaml            # EM + F1 only
│   ├── standard.yaml        # + LLM judge
│   ├── full.yaml            # All metrics
│   └── rag.yaml             # RAG-specific (Ragas)
├── evaluation/              # Evaluation settings (5 options)
│   ├── quick.yaml           # 10 examples, fast
│   ├── standard.yaml        # 100 examples
│   ├── full.yaml            # All examples
│   ├── generate_only.yaml
│   └── evaluate_only.yaml
├── judge/                   # LLM judge models
│   ├── gpt4_mini.yaml
│   └── gpt4.yaml
├── mlflow/                  # MLflow tracking
│   ├── default.yaml
│   └── disabled.yaml
└── experiment/              # Complete experiment presets
    ├── baseline.yaml        # DirectLLM baseline
    ├── rag.yaml             # RAG experiments
    ├── quick_test.yaml      # Quick smoke test
    └── model_comparison.yaml
```

## 🚀 Quick Start

### Run with defaults
```bash
python -m ragicamp.cli.run
```

### Override single parameter
```bash
python -m ragicamp.cli.run model=gemma_2b_4bit
```

### Override multiple parameters
```bash
python -m ragicamp.cli.run model=phi3 dataset=triviaqa evaluation=quick
```

### Override specific values
```bash
python -m ragicamp.cli.run dataset.num_examples=50 model.load_in_4bit=true
```

### Multi-run (parameter sweep)
```bash
python -m ragicamp.cli.run --multirun \
  model=gemma_2b,phi3 \
  agent=direct_llm,fixed_rag \
  dataset.num_examples=10,50,100
```

## 📝 Examples

### Quick test with Gemma 2B
```bash
python -m ragicamp.cli.run \
  model=gemma_2b_4bit \
  dataset=nq \
  evaluation=quick \
  metrics=fast
```

### Full RAG experiment
```bash
python -m ragicamp.cli.run \
  experiment=rag \
  model=gemma_2b \
  dataset=nq \
  evaluation=standard
```

### Compare models (multi-run)
```bash
python -m ragicamp.cli.run --multirun \
  model=gemma_2b,phi3,llama3_8b \
  dataset=nq \
  evaluation=standard
```

### Sweep top_k values
```bash
python -m ragicamp.cli.run --multirun \
  agent=fixed_rag \
  agent.top_k=1,3,5,10 \
  dataset.num_examples=100
```

## 🔧 Config Composition

Hydra composes configs from multiple sources:

```yaml
# config.yaml - defaults that can be overridden
defaults:
  - model: gemma_2b           # Use conf/model/gemma_2b.yaml
  - dataset: nq               # Use conf/dataset/nq.yaml
  - agent: direct_llm         # Use conf/agent/direct_llm.yaml
  - metrics: fast             # Use conf/metrics/fast.yaml
  - evaluation: standard      # Use conf/evaluation/standard.yaml
  - _self_                    # This file's values override defaults

# Additional settings
output:
  save_predictions: true
  dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

## 🎯 Benefits

1. **No Duplication** - Define model once, reuse everywhere
2. **Easy Overrides** - Change any param from CLI
3. **Multi-run** - Sweep hyperparameters automatically
4. **Composition** - Mix and match components
5. **Reproducibility** - Hydra saves full config with outputs

---

## 🔄 Migration from Old Configs

Old config files in `experiments/configs/` still work! But new experiments should use Hydra.

| Old Way | New Way |
|---------|---------|
| `nq_baseline_gemma2b_quick.yaml` | `experiment=baseline evaluation=quick` |
| `nq_baseline_gemma2b_full.yaml` | `experiment=baseline evaluation=full` |
| `nq_fixed_rag_gemma2b.yaml` | `experiment=rag` |
| Copy YAML and edit | `model=phi3` override |

---

## 📊 Combinations Available

With Hydra's composition, you have:

- **7 models** × **3 datasets** × **3 agents** × **4 metric sets** × **5 eval modes** = **1,260 combinations**

All without writing a single new config file!

---

## 🧪 Example Workflows

### Research: Compare Models
```bash
# Run same experiment with 3 different models
python -m ragicamp.cli.run --multirun \
  model=gemma_2b_4bit,phi3,llama3_8b \
  dataset=nq \
  evaluation=standard
```

### Debug: Quick Test
```bash
# Fast smoke test (10 examples, no heavy metrics)
python -m ragicamp.cli.run experiment=quick_test
```

### Ablation: Vary top_k
```bash
# Sweep retrieval parameter
python -m ragicamp.cli.run --multirun \
  experiment=rag \
  agent.top_k=1,3,5,10,20
```

### Cross-Dataset Evaluation
```bash
# Same model on all datasets
python -m ragicamp.cli.run --multirun \
  model=gemma_2b_4bit \
  dataset=nq,triviaqa,hotpotqa
```
