#!/usr/bin/env python3
"""Main experiment runner script."""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.agents.bandit_rag import BanditRAGAgent
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.agents.mdp_rag import MDPRAGAgent
from ragicamp.datasets.hotpotqa import HotpotQADataset
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.triviaqa import TriviaQADataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

# Optional metrics
try:
    from ragicamp.metrics.bertscore import BERTScoreMetric
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from ragicamp.metrics.bleurt import BLEURTMetric
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False

try:
    from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False

try:
    from ragicamp.metrics.faithfulness import FaithfulnessMetric
    FAITHFULNESS_AVAILABLE = True
except ImportError:
    FAITHFULNESS_AVAILABLE = False

try:
    from ragicamp.metrics.hallucination import HallucinationMetric
    HALLUCINATION_AVAILABLE = True
except ImportError:
    HALLUCINATION_AVAILABLE = False

from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.models.openai import OpenAIModel
from ragicamp.policies.bandits import EpsilonGreedyBandit, UCBBandit
from ragicamp.policies.mdp import QLearningMDPPolicy, RandomMDPPolicy
from ragicamp.retrievers.dense import DenseRetriever
from ragicamp.retrievers.sparse import SparseRetriever
from ragicamp.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: dict):
    """Create model from configuration."""
    model_type = config.get("type", "huggingface")
    
    if model_type == "huggingface":
        return HuggingFaceModel(
            model_name=config["model_name"],
            device=config.get("device", "cpu"),
            load_in_8bit=config.get("load_in_8bit", False),
            load_in_4bit=config.get("load_in_4bit", False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_retriever(config: dict):
    """Create retriever from configuration."""
    retriever_type = config.get("type", "dense")
    
    if retriever_type == "dense":
        return DenseRetriever(
            name=config.get("name", "dense_retriever"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            index_type=config.get("index_type", "flat")
        )
    elif retriever_type == "sparse":
        return SparseRetriever(
            name=config.get("name", "sparse_retriever"),
            max_features=config.get("max_features", 10000)
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def create_policy(config: dict):
    """Create policy from configuration."""
    policy_type = config.get("type", "random")
    
    if policy_type == "epsilon_greedy":
        return EpsilonGreedyBandit(
            name=config.get("name", "epsilon_greedy"),
            actions=config.get("actions", []),
            epsilon=config.get("epsilon", 0.1)
        )
    elif policy_type == "ucb":
        return UCBBandit(
            name=config.get("name", "ucb"),
            actions=config.get("actions", []),
            c=config.get("c", 2.0)
        )
    elif policy_type == "qlearning":
        return QLearningMDPPolicy(
            name=config.get("name", "qlearning"),
            action_types=config.get("action_types", ["retrieve", "reformulate", "generate"]),
            learning_rate=config.get("learning_rate", 0.1),
            discount_factor=config.get("discount_factor", 0.95),
            epsilon=config.get("epsilon", 0.1)
        )
    elif policy_type == "random":
        return RandomMDPPolicy(
            name=config.get("name", "random"),
            action_types=config.get("action_types", ["retrieve", "reformulate", "generate"])
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def create_agent(config: dict, model, retriever=None, policy=None):
    """Create agent from configuration."""
    agent_type = config.get("type", "direct_llm")
    
    if agent_type == "direct_llm":
        return DirectLLMAgent(
            name=config.get("name", "direct_llm"),
            model=model,
            system_prompt=config.get("system_prompt", "")
        )
    elif agent_type == "fixed_rag":
        return FixedRAGAgent(
            name=config.get("name", "fixed_rag"),
            model=model,
            retriever=retriever,
            top_k=config.get("top_k", 5),
            system_prompt=config.get("system_prompt", "")
        )
    elif agent_type == "bandit_rag":
        return BanditRAGAgent(
            name=config.get("name", "bandit_rag"),
            model=model,
            retriever=retriever,
            policy=policy,
            system_prompt=config.get("system_prompt", "")
        )
    elif agent_type == "mdp_rag":
        return MDPRAGAgent(
            name=config.get("name", "mdp_rag"),
            model=model,
            retriever=retriever,
            policy=policy,
            max_steps=config.get("max_steps", 5),
            system_prompt=config.get("system_prompt", "")
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_dataset(config: dict):
    """Create dataset from configuration."""
    dataset_name = config.get("name", "natural_questions")
    split = config.get("split", "validation")
    
    if dataset_name == "natural_questions":
        dataset = NaturalQuestionsDataset(split=split)
    elif dataset_name == "hotpotqa":
        dataset = HotpotQADataset(split=split)
    elif dataset_name == "triviaqa":
        dataset = TriviaQADataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Filter questions without answers if specified
    if config.get("filter_no_answer", False):
        print("Filtering questions without explicit answers...")
        original_size = len(dataset)
        dataset.filter_with_answers()
        print(f"Filtered: {original_size} → {len(dataset)} examples")
    
    # Limit number of examples if specified
    num_examples = config.get("num_examples")
    if num_examples and num_examples < len(dataset):
        dataset.examples = dataset.examples[:num_examples]
    
    return dataset


def create_metrics(config: list, judge_model_config: dict = None):
    """Create metrics from configuration.
    
    Args:
        config: List of metric names or dicts with metric config
        judge_model_config: Optional config for LLM judge model
        
    Returns:
        List of metric instances
    """
    metrics = []
    judge_model = None
    
    # Create judge model if needed and configured
    if judge_model_config:
        print("Creating LLM judge model...")
        model_type = judge_model_config.get("type", "openai")
        if model_type == "openai":
            import os
            if not os.getenv("OPENAI_API_KEY"):
                print("⚠️  OPENAI_API_KEY not set. LLM judge metrics will be skipped.")
                print("   Set it with: export OPENAI_API_KEY='your-key'")
            else:
                try:
                    judge_model = OpenAIModel(
                        model_name=judge_model_config.get("model_name", "gpt-4o"),
                        temperature=judge_model_config.get("temperature", 0.0)
                    )
                    print(f"✓ Created judge model: {judge_model_config.get('model_name', 'gpt-4o')}")
                except Exception as e:
                    print(f"⚠️  Failed to create judge model: {e}")
    
    for metric_config in config:
        # Handle simple string or dict format
        if isinstance(metric_config, str):
            metric_name = metric_config
            metric_params = {}
        else:
            metric_name = metric_config.get("name")
            metric_params = metric_config.get("params", {})
        
        # Create metric
        if metric_name == "exact_match":
            metrics.append(ExactMatchMetric(**metric_params))
        elif metric_name == "f1":
            metrics.append(F1Metric(**metric_params))
        elif metric_name == "bertscore":
            if not BERTSCORE_AVAILABLE:
                print(f"⚠️  Skipping BERTScore (not installed). Install with: uv sync")
                continue
            model_type = metric_params.get("model_type", "microsoft/deberta-base-mnli")
            metrics.append(BERTScoreMetric(model_type=model_type))
        elif metric_name == "bleurt":
            if not BLEURT_AVAILABLE:
                print(f"⚠️  Skipping BLEURT (not installed). Install with: uv sync")
                continue
            checkpoint = metric_params.get("checkpoint", "BLEURT-20-D3")
            metrics.append(BLEURTMetric(checkpoint=checkpoint))
        elif metric_name == "llm_judge_qa":
            if not LLM_JUDGE_AVAILABLE:
                print(f"⚠️  Skipping LLM Judge (not available)")
                continue
            if not judge_model:
                print(f"⚠️  Skipping LLM Judge (judge_model not configured or API key not set)")
                continue
            judgment_type = metric_params.get("judgment_type", "binary")
            metrics.append(LLMJudgeQAMetric(
                judge_model=judge_model,
                judgment_type=judgment_type
            ))
            print(f"✓ Added LLM Judge ({judgment_type} mode)")
        elif metric_name == "faithfulness":
            if not FAITHFULNESS_AVAILABLE:
                print(f"⚠️  Skipping Faithfulness (not installed). Install with: uv sync")
                continue
            method = metric_params.get("method", "nli")
            nli_model = metric_params.get("nli_model", "microsoft/deberta-base-mnli")
            threshold = metric_params.get("threshold", 0.5)
            # Check if LLM method requested
            if method == "llm":
                if not judge_model:
                    print(f"⚠️  Skipping Faithfulness (llm method requires judge_model)")
                    continue
                metrics.append(FaithfulnessMetric(
                    method=method,
                    judge_model=judge_model,
                    threshold=threshold
                ))
            else:
                metrics.append(FaithfulnessMetric(
                    method=method,
                    nli_model=nli_model,
                    threshold=threshold
                ))
            print(f"✓ Added Faithfulness ({method} method)")
        elif metric_name == "hallucination":
            if not HALLUCINATION_AVAILABLE:
                print(f"⚠️  Skipping Hallucination (not installed). Install with: uv sync")
                continue
            method = metric_params.get("method", "nli")
            nli_model = metric_params.get("nli_model", "microsoft/deberta-base-mnli")
            threshold = metric_params.get("threshold", 0.5)
            metrics.append(HallucinationMetric(
                method=method,
                nli_model=nli_model,
                threshold=threshold
            ))
            print(f"✓ Added Hallucination Detection ({method} method)")
        else:
            print(f"⚠️  Unknown metric: {metric_name}, skipping")
    
    if not metrics:
        print("⚠️  No valid metrics configured, using defaults: exact_match, f1")
        metrics = [ExactMatchMetric(), F1Metric()]
    
    return metrics


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run RAGiCamp experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="eval",
        help="Run mode: train or eval"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create components
    print("Creating model...")
    model = create_model(config["model"])
    
    retriever = None
    if "retriever" in config:
        print("Creating retriever...")
        retriever = create_retriever(config["retriever"])
        # TODO: Index documents from corpus
    
    policy = None
    if "policy" in config:
        print("Creating policy...")
        policy = create_policy(config["policy"])
    
    print("Creating agent...")
    agent = create_agent(config["agent"], model, retriever, policy)
    
    print("Loading dataset...")
    dataset = create_dataset(config["dataset"])
    print(f"Dataset size: {len(dataset)}")
    
    print("Creating metrics...")
    judge_model_config = config.get("judge_model")
    metrics = create_metrics(config["metrics"], judge_model_config)
    
    # Run experiment
    if args.mode == "train":
        print("\n=== Training Mode ===")
        trainer = Trainer(
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            reward_metric="f1"
        )
        
        training_config = config.get("training", {})
        results = trainer.train(
            num_epochs=training_config.get("num_epochs", 1),
            eval_interval=training_config.get("eval_interval", 100)
        )
        
        # Save policy if applicable
        output_config = config.get("output", {})
        if output_config.get("save_policy") and policy:
            policy_path = output_config.get("policy_path", "outputs/policy.json")
            os.makedirs(os.path.dirname(policy_path), exist_ok=True)
            policy.save(policy_path)
            print(f"Policy saved to {policy_path}")
    
    else:  # eval mode
        print("\n=== Evaluation Mode ===")
        evaluator = Evaluator(
            agent=agent,
            dataset=dataset,
            metrics=metrics
        )
        
        output_config = config.get("output", {})
        eval_config = config.get("evaluation", {})
        results = evaluator.evaluate(
            save_predictions=output_config.get("save_predictions", False),
            output_path=output_config.get("output_path"),
            batch_size=eval_config.get("batch_size")
        )
        
        # Print results
        print("\n=== Results ===")
        for metric_name, score in results.items():
            if metric_name not in ["num_examples", "agent_name", "dataset_name"]:
                if isinstance(score, float):
                    print(f"{metric_name}: {score:.4f}")
                else:
                    print(f"{metric_name}: {score}")


if __name__ == "__main__":
    main()

