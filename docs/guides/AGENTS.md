# RAGiCamp Agents Guide

Complete guide to understanding, using, and creating RAG agents in RAGiCamp.

## 📖 Table of Contents

- [What Are Agents?](#what-are-agents)
- [Agent Types](#agent-types)
- [Using Agents](#using-agents)
- [Saving & Loading](#saving--loading)
- [Creating Custom Agents](#creating-custom-agents)
- [Best Practices](#best-practices)

---

## What Are Agents?

**Agents** are the core decision-making components in RAGiCamp. An agent takes a question and produces an answer, potentially using retrieval, intermediate reasoning steps, or adaptive strategies.

### Base Abstraction

All agents inherit from `RAGAgent` and implement the `answer()` method:

```python
from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse

class MyAgent(RAGAgent):
    def answer(self, query: str, **kwargs) -> RAGResponse:
        """Generate an answer for the given query."""
        # Your implementation here
        pass
```

### Key Concepts

**RAGContext** - Information about the query:
```python
@dataclass
class RAGContext:
    query: str                           # The question
    retrieved_docs: List[Document]       # Retrieved documents
    intermediate_steps: List[Dict]       # Action history (for MDP)
    metadata: Dict[str, Any]             # Additional info
```

**RAGResponse** - The agent's answer:
```python
@dataclass
class RAGResponse:
    answer: str                          # Generated answer
    context: RAGContext                  # Context used
    confidence: Optional[float]          # Confidence score
    metadata: Dict[str, Any]             # Additional metadata
```

---

## Agent Types

### 1. DirectLLMAgent

**No retrieval** - Just queries the LLM directly.

```python
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.models.huggingface import HuggingFaceModel

model = HuggingFaceModel('google/gemma-2-2b-it')
agent = DirectLLMAgent(
    name="baseline_llm",
    model=model,
    system_prompt="You are a helpful assistant."
)

response = agent.answer("What is Python?")
print(response.answer)
```

**Use Cases:**
- Baseline comparison
- When documents aren't needed
- Testing LLM capabilities

**Training:** None required

---

### 2. FixedRAGAgent

**Standard RAG** with fixed parameters (most common for production).

```python
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.retrievers.dense import DenseRetriever

# Create retriever
retriever = DenseRetriever(
    name="wiki_retriever",
    embedding_model="all-MiniLM-L6-v2"
)
retriever.index_documents(documents)

# Create agent
model = HuggingFaceModel('google/gemma-2-2b-it')
agent = FixedRAGAgent(
    name="fixed_rag",
    model=model,
    retriever=retriever,
    top_k=5,  # Retrieve 5 documents
    system_prompt="Use the context to answer accurately."
)

response = agent.answer("What is machine learning?")
print(response.answer)
print(f"Used {len(response.context.retrieved_docs)} documents")
```

**Use Cases:**
- Production RAG systems
- When you know good parameters
- Stable, reliable performance

**Training:** Index documents once

**Save/Load:**
```python
# Save
agent.save("fixed_rag_v1", "wikipedia_nq_v1")

# Load
agent = FixedRAGAgent.load("fixed_rag_v1", model)
```

---

### 3. BanditRAGAgent

**Adaptive RAG** - Uses bandit algorithms to learn optimal parameters.

```python
from ragicamp.agents.bandit_rag import BanditRAGAgent
from ragicamp.policies.bandits import EpsilonGreedyBandit

# Define possible actions (parameter configurations)
actions = [
    {"top_k": 3},
    {"top_k": 5},
    {"top_k": 10},
]

# Create policy
policy = EpsilonGreedyBandit(
    name="param_selector",
    actions=actions,
    epsilon=0.1  # 10% exploration
)

# Create agent
agent = BanditRAGAgent(
    name="bandit_rag",
    model=model,
    retriever=retriever,
    policy=policy
)

# Use - policy selects best top_k automatically
response = agent.answer("What is deep learning?")

# Update policy based on reward
reward = compute_reward(response.answer, gold_answer)
agent.update_policy(query, params, reward)
```

**Use Cases:**
- Optimizing retrieval parameters
- A/B testing different strategies
- Learning from user feedback

**Training:** 
1. Index documents
2. Run training loop with rewards
3. Policy learns optimal parameters

---

### 4. MDPRAGAgent

**Sequential decision-making** - Takes multiple steps to answer.

```python
from ragicamp.agents.mdp_rag import MDPRAGAgent
from ragicamp.policies.mdp import QLearningMDPPolicy

# Create policy
policy = QLearningMDPPolicy(
    name="mdp_policy",
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
    max_steps=5  # Max 5 steps before forcing answer
)

# Use - agent decides sequence of actions
response = agent.answer("Complex multi-hop question?")

# See what it did
for step in response.context.intermediate_steps:
    print(f"Step {step['step']}: {step['action_type']}")
```

**Actions:**
- `retrieve` - Get more documents
- `reformulate` - Rephrase query
- `generate` - Produce answer

**Use Cases:**
- Complex reasoning tasks
- Multi-hop questions
- When simple retrieval isn't enough

**Training:**
1. Index documents
2. Run episodes with reward feedback
3. Policy learns action sequences

---

## Using Agents

### Basic Usage Pattern

```python
# 1. Create/load agent
agent = FixedRAGAgent.load("fixed_rag_v1", model)

# 2. Answer single question
response = agent.answer("What is RAG?")
print(response.answer)

# 3. Batch questions
questions = ["Q1?", "Q2?", "Q3?"]
for q in questions:
    response = agent.answer(q)
    print(f"Q: {q}\nA: {response.answer}\n")
```

### With Evaluation

```python
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

evaluator = Evaluator(
    agent=agent,
    dataset=dataset,
    metrics=[ExactMatchMetric(), F1Metric()]
)

results = evaluator.evaluate(
    num_examples=100,
    save_predictions=True,
    output_path="outputs/results.json"
)

print(f"Exact Match: {results['exact_match']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
```

### With Training (Adaptive Agents)

```python
from ragicamp.training.trainer import Trainer

trainer = Trainer(
    agent=bandit_agent,
    dataset=train_dataset,
    metrics=[F1Metric()],
    reward_metric="f1"  # Use F1 as reward
)

trainer.train(
    num_epochs=1,
    eval_interval=100
)

# Policy is now trained!
# Save for later use
bandit_agent.policy.save("trained_policy.json")
```

---

## Saving & Loading

### Why Save/Load?

- **Train once, use forever** - No need to recompute embeddings
- **Share artifacts** - Collaborate across team
- **Version control** - Track different model versions
- **Production ready** - Fast startup, no training overhead

### Workflow

```python
# ============================================
# TRAINING (do once)
# ============================================

# 1. Index documents
retriever = DenseRetriever(name="retriever", embedding_model="...")
retriever.index_documents(documents)
retriever.save_index("wikipedia_nq_v1")

# 2. Create and save agent
agent = FixedRAGAgent(name="agent", model=None, retriever=retriever, top_k=5)
agent.save("fixed_rag_v1", "wikipedia_nq_v1")

# ============================================
# INFERENCE (use many times)
# ============================================

# Load model at runtime
model = HuggingFaceModel('google/gemma-2-2b-it')

# Load agent (automatically loads retriever)
agent = FixedRAGAgent.load("fixed_rag_v1", model)

# Use immediately
response = agent.answer("Question?")
```

### Artifact Structure

```
artifacts/
├── retrievers/
│   └── wikipedia_nq_v1/
│       ├── index.faiss        # Vector index
│       ├── documents.pkl      # Document store
│       └── config.json        # Metadata
└── agents/
    └── fixed_rag_v1/
        └── config.json        # Agent config
```

### CLI Training

```bash
# Quick test (1000 docs)
make train-fixed-rag-small

# Full training
make train-fixed-rag

# List what you have
make list-artifacts

# Outputs:
# Retrievers:
#   wikipedia_nq_v1
#   wikipedia_nq_small
# Agents:
#   fixed_rag_v1
#   fixed_rag_small
```

---

## Creating Custom Agents

### Minimal Example

```python
from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.models.base import LanguageModel

class MyCustomAgent(RAGAgent):
    def __init__(self, name: str, model: LanguageModel, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model
    
    def answer(self, query: str, **kwargs) -> RAGResponse:
        # Your custom logic here
        context = RAGContext(query=query)
        
        # Do something interesting
        answer = self.model.generate(f"Answer: {query}")
        
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={"agent_type": "custom"}
        )
```

### With Retrieval

```python
from ragicamp.retrievers.base import Retriever
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder

class MyRAGAgent(RAGAgent):
    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.prompt_builder = PromptBuilder.create_default()
    
    def answer(self, query: str, **kwargs) -> RAGResponse:
        # Retrieve documents
        docs = self.retriever.retrieve(query, top_k=5)
        
        # Create context
        context = RAGContext(query=query, retrieved_docs=docs)
        
        # Format context
        context_text = ContextFormatter.format_numbered(docs)
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(query, context_text)
        
        # Generate
        answer = self.model.generate(prompt, **kwargs)
        
        return RAGResponse(answer=answer, context=context)
```

### With Custom State

```python
class StatefulAgent(RAGAgent):
    def __init__(self, name: str, model: LanguageModel):
        super().__init__(name)
        self.model = model
        self.conversation_history = []  # Maintain state
    
    def answer(self, query: str, **kwargs) -> RAGResponse:
        # Add to history
        self.conversation_history.append(query)
        
        # Use history in prompt
        history_text = "\n".join(self.conversation_history[-5:])  # Last 5
        prompt = f"History:\n{history_text}\n\nCurrent: {query}\n\nAnswer:"
        
        answer = self.model.generate(prompt)
        
        return RAGResponse(
            answer=answer,
            context=RAGContext(
                query=query,
                metadata={"history_length": len(self.conversation_history)}
            )
        )
    
    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []
```

---

## Best Practices

### 1. Use Utilities

✅ **DO** - Use formatting and prompt utilities:
```python
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder

context_text = ContextFormatter.format_numbered(docs)
prompt = PromptBuilder.create_default().build_prompt(query, context_text)
```

❌ **DON'T** - Duplicate formatting logic:
```python
# Don't reimplement this in every agent
context_text = "\n\n".join([f"[{i}] {doc.text}" for i, doc in enumerate(docs, 1)])
```

### 2. Return Proper Types

✅ **DO** - Return RAGResponse with context:
```python
def answer(self, query: str, **kwargs) -> RAGResponse:
    context = RAGContext(query=query, retrieved_docs=docs)
    return RAGResponse(answer=answer, context=context)
```

❌ **DON'T** - Return just strings:
```python
def answer(self, query: str) -> str:  # ❌ Wrong type
    return "answer"  # ❌ No context
```

### 3. Use Type Hints

✅ **DO**:
```python
from typing import Any
from ragicamp.retrievers.base import Document

def answer(self, query: str, **kwargs: Any) -> RAGResponse:
    docs: List[Document] = self.retriever.retrieve(query)
```

### 4. Document Metadata

✅ **DO** - Include useful metadata:
```python
return RAGResponse(
    answer=answer,
    context=context,
    metadata={
        "agent_type": "my_custom_agent",
        "num_docs_used": len(docs),
        "retrieval_time": elapsed_time,
        "model_name": self.model.model_name
    }
)
```

### 5. Implement Save/Load

If your agent has configuration (not models), implement save/load:

```python
def save(self, artifact_name: str, retriever_artifact: str) -> str:
    manager = get_artifact_manager()
    path = manager.get_agent_path(artifact_name)
    
    config = {
        "agent_type": "my_custom",
        "name": self.name,
        "retriever_artifact": retriever_artifact,
        # Your config here
    }
    manager.save_json(config, path / "config.json")
    return str(path)

@classmethod
def load(cls, artifact_name: str, model: LanguageModel) -> 'MyAgent':
    manager = get_artifact_manager()
    path = manager.get_agent_path(artifact_name)
    config = manager.load_json(path / "config.json")
    
    retriever = DenseRetriever.load_index(config["retriever_artifact"])
    return cls(name=config["name"], model=model, retriever=retriever)
```

---

## Comparison Table

| Feature | DirectLLM | FixedRAG | BanditRAG | MDPRAG |
|---------|-----------|----------|-----------|---------|
| **Retrieval** | ❌ | ✅ | ✅ | ✅ |
| **Training** | ❌ | Index only | RL training | RL training |
| **Parameters** | Fixed | Fixed | Adaptive | Adaptive |
| **Complexity** | Simple | Simple | Medium | Complex |
| **Use Case** | Baseline | Production | Optimization | Complex reasoning |
| **Speed** | Fast | Fast | Fast | Slower (multi-step) |
| **Setup Time** | None | 5-10 min | 30+ min | 1+ hour |
| **Performance** | Baseline | Good | Better | Best (with training) |

---

## Quick Reference

```python
# DirectLLM - No retrieval
agent = DirectLLMAgent(name="direct", model=model)

# FixedRAG - Standard RAG
agent = FixedRAGAgent(name="fixed", model=model, retriever=retriever, top_k=5)
agent.save("fixed_rag_v1", "wikipedia_v1")
agent = FixedRAGAgent.load("fixed_rag_v1", model)

# BanditRAG - Adaptive parameters
policy = EpsilonGreedyBandit(name="policy", actions=actions, epsilon=0.1)
agent = BanditRAGAgent(name="bandit", model=model, retriever=retriever, policy=policy)
agent.update_policy(query, params, reward)

# MDPRAG - Sequential decisions
policy = QLearningMDPPolicy(name="mdp", action_types=["retrieve", "generate"])
agent = MDPRAGAgent(name="mdp", model=model, retriever=retriever, policy=policy)

# All agents
response = agent.answer("Question?")
print(response.answer)
```

---

## 📊 Evaluating Agents

### Modern Approach: Hydra Configs ⭐ **RECOMMENDED**

The easiest way to evaluate agents is with the Hydra configuration system:

```bash
# Quick evaluation with defaults
python -m ragicamp.cli.run

# Override components
python -m ragicamp.cli.run model=phi3 dataset=triviaqa evaluation.num_examples=100

# Use experiment presets
python -m ragicamp.cli.run +experiment=baseline
```

**Available configs in `conf/`:**

```
conf/
├── model/          # gemma_2b, phi3, llama3_8b, openai_gpt4
├── dataset/        # nq, triviaqa, hotpotqa
├── agent/          # direct_llm, fixed_rag, bandit_rag
├── metrics/        # fast, standard, full, rag
├── evaluation/     # quick, standard, full
└── experiment/     # baseline, rag, quick_test
```

**Benefits:**
- ✅ **Composable** - Mix and match components
- ✅ **CLI overrides** - Change any parameter without editing files
- ✅ **Multi-run sweeps** - Test multiple configs at once
- ✅ **Reproducible** - Hydra saves full config with each run

### Parameter Sweeps

```bash
# Test multiple models
python -m ragicamp.cli.run -m model=gemma_2b,phi3

# Sweep datasets and models
python -m ragicamp.cli.run -m model=gemma_2b,phi3 dataset=nq,triviaqa

# Sweep top_k values
python -m ragicamp.cli.run -m agent=fixed_rag agent.top_k=3,5,10
```

### Programmatic API (Advanced)

For custom workflows, use the evaluator directly:

```python
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric
from ragicamp.metrics.f1_metric import F1Metric

# Phase 1: Generate predictions
evaluator = Evaluator(agent, dataset)
predictions_file = evaluator.generate_predictions(
    output_path="outputs/predictions.json",
    num_examples=100,
    batch_size=8
)

# Phase 2: Compute metrics (can retry if it fails!)
results = evaluator.compute_metrics(
    predictions_file=predictions_file,
    metrics=[ExactMatchMetric(), F1Metric()],
    output_path="outputs/results.json"
)

print(f"Exact Match: {results['exact_match']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
```

### LLM-as-a-Judge

```bash
# Use with judge model
python -m ragicamp.cli.run +judge=gpt4_mini metrics=full
```

Features:
- ✅ **Saves progress every 5 batches** automatically
- ✅ **Resumes from checkpoint** if API fails
- ✅ **Batch processing** for speed

---

## Next Steps

- **[Hydra Config Guide](HYDRA_CONFIG.md)** - Full configuration system docs
- **[CHEATSHEET](../../CHEATSHEET.md)** - Quick reference for common commands
- **[CONTRIBUTING](../../CONTRIBUTING.md)** - How to add new agents

**Ready to build?** Start with FixedRAGAgent for production, then explore adaptive agents for optimization! 🚀

