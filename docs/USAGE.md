# RAGiCamp Usage Guide

This guide shows how to use RAGiCamp for your RAG experiments.

## Installation

```bash
# Clone the repository
cd ragicamp

# Install with uv (recommended - faster and better dependency resolution)
uv sync

# Install with optional dependencies
uv sync --extra dev --extra metrics --extra viz

# Or install with pip if you prefer
# pip install -e ".[dev,metrics,viz]"
```

## Quick Start

### 1. Using Agents Programmatically

```python
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.models.huggingface import HuggingFaceModel

# Create a model
model = HuggingFaceModel(
    model_name="google/flan-t5-base",
    device="cuda"
)

# Create an agent
agent = DirectLLMAgent(
    name="my_agent",
    model=model
)

# Ask a question
response = agent.answer("What is the capital of France?")
print(response.answer)
```

### 2. Using Configuration Files

```bash
# Run baseline experiment
uv run python experiments/scripts/run_experiment.py \
    --config experiments/configs/baseline_direct.yaml \
    --mode eval

# Train adaptive agent
uv run python experiments/scripts/run_experiment.py \
    --config experiments/configs/bandit_rag.yaml \
    --mode train

# Run Gemma 2B baseline (recommended starting point)
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100
```

## Components Guide

### Agents

#### DirectLLMAgent (Baseline 1)
No retrieval, just asks the LLM.

```python
from ragicamp.agents.direct_llm import DirectLLMAgent

agent = DirectLLMAgent(
    name="direct_llm",
    model=model,
    system_prompt="Answer questions accurately."
)
```

#### FixedRAGAgent (Baseline 2)
Standard RAG with fixed parameters.

```python
from ragicamp.agents.fixed_rag import FixedRAGAgent

agent = FixedRAGAgent(
    name="fixed_rag",
    model=model,
    retriever=retriever,
    top_k=5  # Always retrieve 5 documents
)
```

#### BanditRAGAgent
Adaptive parameter selection using bandits.

```python
from ragicamp.agents.bandit_rag import BanditRAGAgent
from ragicamp.policies.bandits import UCBBandit

# Define action space (different configurations)
actions = [
    {"top_k": 3},
    {"top_k": 5},
    {"top_k": 10}
]

# Create policy
policy = UCBBandit(name="ucb", actions=actions, c=2.0)

# Create agent
agent = BanditRAGAgent(
    name="bandit_rag",
    model=model,
    retriever=retriever,
    policy=policy
)

# After getting response and reward
agent.update_policy(
    query=query,
    params=selected_params,
    reward=reward
)
```

#### MDPRAGAgent
Iterative decision-making with reinforcement learning.

```python
from ragicamp.agents.mdp_rag import MDPRAGAgent
from ragicamp.policies.mdp import QLearningMDPPolicy

# Create MDP policy
policy = QLearningMDPPolicy(
    name="qlearning",
    action_types=["retrieve", "reformulate", "generate"],
    learning_rate=0.1,
    epsilon=0.1
)

# Create agent
agent = MDPRAGAgent(
    name="mdp_rag",
    model=model,
    retriever=retriever,
    policy=policy,
    max_steps=5  # Maximum steps before forcing answer
)
```

### Models

#### HuggingFace Models

```python
from ragicamp.models.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_name="google/flan-t5-base",
    device="cuda",
    load_in_8bit=False  # Enable for large models
)
```

#### OpenAI Models

```python
from ragicamp.models.openai import OpenAIModel

model = OpenAIModel(
    model_name="gpt-4.1-mini",
    api_key="your-api-key"  # Or set OPENAI_API_KEY env var
)
```

### Retrievers

#### Dense Retriever (Embeddings + FAISS)

```python
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.base import Document

# Create retriever
retriever = DenseRetriever(
    name="dense",
    embedding_model="all-MiniLM-L6-v2",
    index_type="flat"
)

# Index documents
documents = [
    Document(id="1", text="Paris is the capital of France.", metadata={}),
    Document(id="2", text="London is the capital of England.", metadata={}),
    # ... more documents
]
retriever.index_documents(documents)

# Retrieve
results = retriever.retrieve("What is the capital of France?", top_k=5)
```

#### Sparse Retriever (TF-IDF)

```python
from ragicamp.retrievers.sparse import SparseRetriever

retriever = SparseRetriever(
    name="sparse",
    max_features=10000
)
```

### Datasets

```python
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.hotpotqa import HotpotQADataset
from ragicamp.datasets.triviaqa import TriviaQADataset

# Load datasets
nq = NaturalQuestionsDataset(split="validation")
hotpot = HotpotQADataset(split="train")
trivia = TriviaQADataset(split="validation")

# Access examples
for example in nq:
    print(f"Q: {example.question}")
    print(f"A: {example.answers}")
```

### Metrics

```python
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric
from ragicamp.metrics.bertscore import BERTScoreMetric
from ragicamp.metrics.llm_judge import LLMJudgeMetric

# Simple metrics
em = ExactMatchMetric()
f1 = F1Metric()

# Advanced metrics
bert_score = BERTScoreMetric(model_type="microsoft/deberta-xlarge-mnli")
llm_judge = LLMJudgeMetric(judge_model=model, criteria="accuracy", scale=10)

# Compute
predictions = ["Paris", "London"]
references = [["Paris"], ["London", "Greater London"]]

em_score = em.compute(predictions, references)
f1_score = f1.compute(predictions, references)
bert_scores = bert_score.compute(predictions, references)
```

### Training

```python
from ragicamp.training.trainer import Trainer

trainer = Trainer(
    agent=adaptive_agent,
    dataset=train_dataset,
    metrics=[F1Metric()],
    reward_metric="f1"
)

results = trainer.train(
    num_epochs=1,
    eval_interval=100
)

# Save learned policy
agent.policy.save("policy.json")
```

### Evaluation

```python
from ragicamp.evaluation.evaluator import Evaluator

evaluator = Evaluator(
    agent=agent,
    dataset=eval_dataset,
    metrics=[ExactMatchMetric(), F1Metric()]
)

# Evaluate single agent
results = evaluator.evaluate(
    num_examples=100,
    save_predictions=True,
    output_path="results.json"
)

# Compare multiple agents
agents = [direct_agent, rag_agent, bandit_agent]
comparison = evaluator.compare_agents(agents, num_examples=100)
```

## Configuration File Format

Example configuration (YAML):

```yaml
agent:
  type: bandit_rag
  name: "my_bandit_agent"
  system_prompt: "Answer accurately using context."

model:
  type: huggingface
  model_name: "google/flan-t5-base"
  device: "cuda"

retriever:
  type: dense
  embedding_model: "all-MiniLM-L6-v2"

policy:
  type: ucb
  c: 2.0
  actions:
    - top_k: 3
    - top_k: 5
    - top_k: 10

dataset:
  name: natural_questions
  split: validation
  num_examples: 100

metrics:
  - exact_match
  - f1

output:
  save_predictions: true
  output_path: "outputs/results.json"
```

## Experiment Workflow

### 1. Run Baselines

```bash
# Baseline 1: Direct LLM
python experiments/scripts/run_experiment.py \
    --config experiments/configs/baseline_direct.yaml \
    --mode eval

# Baseline 2: Fixed RAG
python experiments/scripts/run_experiment.py \
    --config experiments/configs/baseline_rag.yaml \
    --mode eval
```

### 2. Train Adaptive Agents

```bash
# Train bandit-based RAG
python experiments/scripts/run_experiment.py \
    --config experiments/configs/bandit_rag.yaml \
    --mode train

# Train MDP-based RAG
python experiments/scripts/run_experiment.py \
    --config experiments/configs/mdp_rag.yaml \
    --mode train
```

### 3. Compare Results

```python
# Use the compare_baselines.py script
python experiments/scripts/compare_baselines.py
```

## Adding New Components

### Add a New Agent

```python
from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse

class MyCustomAgent(RAGAgent):
    def __init__(self, name, model, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model
    
    def answer(self, query, **kwargs):
        # Your custom logic here
        context = RAGContext(query=query)
        answer = self.model.generate(f"Question: {query}\nAnswer:")
        
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={"agent_type": "custom"}
        )
```

### Add a New Metric

```python
from ragicamp.metrics.base import Metric

class MyCustomMetric(Metric):
    def __init__(self):
        super().__init__(name="my_metric")
    
    def compute(self, predictions, references, **kwargs):
        # Your custom metric logic
        scores = []
        for pred, ref in zip(predictions, references):
            score = # ... compute score
            scores.append(score)
        return sum(scores) / len(scores)
```

## Tips and Best Practices

1. **Start Small**: Test on a small dataset subset first
2. **Use Configs**: Define experiments in YAML for reproducibility
3. **Monitor Training**: Use `eval_interval` to track learning progress
4. **Save Policies**: Save learned policies for later use
5. **Compare Fairly**: Use same data, model, and metrics for all comparisons
6. **Iterate**: Start with baselines, then add complexity

## Troubleshooting

### Out of Memory

- Use smaller models
- Enable `load_in_8bit=True` for HuggingFace models
- Reduce batch sizes
- Use CPU instead of GPU for small experiments

### Slow Retrieval

- Use smaller embedding models
- Reduce corpus size
- Use IVF index for large document collections

### Poor Performance

- Check that documents are properly indexed
- Verify metric implementations
- Ensure proper text normalization
- Try different reward metrics for training

## Next Steps

- Check `ARCHITECTURE.md` for design details
- Explore `notebooks/quickstart.ipynb` for interactive examples
- Read source code for implementation details
- Add your own datasets, models, and agents!

