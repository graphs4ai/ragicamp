"""Factory for creating RAGiCamp components from configuration dictionaries.

This module provides a centralized way to instantiate models, agents, datasets,
metrics, and retrievers from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ragicamp.agents import (
    RAGAgent,
    DirectLLMAgent,
    FixedRAGAgent,
    BanditRAGAgent,
    MDPRAGAgent,
)
from ragicamp.datasets import (
    QADataset,
    NaturalQuestionsDataset,
    TriviaQADataset,
    HotpotQADataset,
)
from ragicamp.metrics import Metric
from ragicamp.models import LanguageModel, HuggingFaceModel, OpenAIModel
from ragicamp.retrievers import Retriever, DenseRetriever


class ComponentFactory:
    """Factory for creating RAGiCamp components from config dictionaries."""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> LanguageModel:
        """Create a language model from configuration.

        Args:
            config: Model configuration dict with 'type' and model-specific params

        Returns:
            Instantiated LanguageModel

        Example:
            >>> config = {"type": "huggingface", "model_name": "google/gemma-2-2b-it"}
            >>> model = ComponentFactory.create_model(config)
        """
        model_type = config.get("type", "huggingface")
        config_copy = dict(config)
        config_copy.pop("type", None)

        # Remove generation-specific parameters (used in generate(), not __init__)
        generation_params = ["max_tokens", "temperature", "top_p", "stop"]
        for param in generation_params:
            config_copy.pop(param, None)

        if model_type == "huggingface":
            return HuggingFaceModel(**config_copy)
        elif model_type == "openai":
            return OpenAIModel(**config_copy)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. " f"Available: huggingface, openai"
            )

    @staticmethod
    def create_agent(
        config: Dict[str, Any],
        model: LanguageModel,
        retriever: Optional[Retriever] = None,
        **kwargs: Any,
    ) -> RAGAgent:
        """Create an agent from configuration.

        Args:
            config: Agent configuration dict with 'type' and agent-specific params
            model: Language model instance
            retriever: Optional retriever for RAG agents
            **kwargs: Additional arguments (e.g., policy for RL agents)

        Returns:
            Instantiated RAGAgent

        Example:
            >>> config = {"type": "direct_llm", "name": "baseline"}
            >>> agent = ComponentFactory.create_agent(config, model)
        """
        agent_type = config["type"]
        config_copy = dict(config)
        config_copy.pop("type", None)

        if agent_type == "direct_llm":
            return DirectLLMAgent(model=model, **config_copy)

        elif agent_type == "fixed_rag":
            if retriever is None:
                raise ValueError("fixed_rag agent requires a retriever")
            return FixedRAGAgent(model=model, retriever=retriever, **config_copy)

        elif agent_type == "bandit_rag":
            if retriever is None:
                raise ValueError("bandit_rag agent requires a retriever")
            policy = kwargs.get("policy")
            if policy is None:
                raise ValueError("bandit_rag agent requires a policy")
            return BanditRAGAgent(model=model, retriever=retriever, policy=policy, **config_copy)

        elif agent_type == "mdp_rag":
            if retriever is None:
                raise ValueError("mdp_rag agent requires a retriever")
            policy = kwargs.get("policy")
            if policy is None:
                raise ValueError("mdp_rag agent requires a policy")
            return MDPRAGAgent(model=model, retriever=retriever, policy=policy, **config_copy)

        else:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available: direct_llm, fixed_rag, bandit_rag, mdp_rag"
            )

    @staticmethod
    def create_dataset(config: Dict[str, Any]) -> QADataset:
        """Create a dataset from configuration.

        Args:
            config: Dataset configuration dict with 'name' and dataset-specific params

        Returns:
            Instantiated QADataset

        Example:
            >>> config = {"name": "natural_questions", "split": "validation"}
            >>> dataset = ComponentFactory.create_dataset(config)
        """
        dataset_name = config["name"]
        split = config.get("split", "validation")
        num_examples = config.get("num_examples")
        filter_no_answer = config.get("filter_no_answer", True)
        cache_dir = config.get("cache_dir", Path("data/datasets"))

        # Create appropriate dataset class
        if dataset_name == "natural_questions":
            dataset_class = NaturalQuestionsDataset
        elif dataset_name == "triviaqa":
            dataset_class = TriviaQADataset
        elif dataset_name == "hotpotqa":
            dataset_class = HotpotQADataset
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: natural_questions, triviaqa, hotpotqa"
            )

        # Load dataset
        dataset = dataset_class(split=split, cache_dir=Path(cache_dir))

        # Filter if needed
        if filter_no_answer:
            original_size = len(dataset)
            dataset.filter_with_answers()
            if len(dataset) < original_size:
                print(f"Filtered: {original_size} → {len(dataset)} examples")

        # Limit if requested
        if num_examples and len(dataset) > num_examples:
            dataset.examples = dataset.examples[:num_examples]
            print(f"Dataset size: {len(dataset)}")

        return dataset

    @staticmethod
    def create_metrics(
        config: List[Union[str, Dict[str, Any]]],
        judge_model: Optional[LanguageModel] = None,
    ) -> List[Metric]:
        """Create metrics from configuration.

        Args:
            config: List of metric names or dicts with name and params
            judge_model: Optional judge model for LLM-based metrics

        Returns:
            List of instantiated Metric objects

        Example:
            >>> config = ["exact_match", "f1", {"name": "bertscore", "params": {...}}]
            >>> metrics = ComponentFactory.create_metrics(config)
        """
        from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

        # Import optional metrics with guards
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

        metrics = []

        for metric_config in config:
            # Handle both string and dict formats
            if isinstance(metric_config, str):
                metric_name = metric_config
                metric_params = {}
            else:
                metric_name = metric_config["name"]
                metric_params = metric_config.get("params", {})

            # Create metric based on name
            if metric_name == "exact_match":
                metrics.append(ExactMatchMetric())

            elif metric_name == "f1":
                metrics.append(F1Metric())

            elif metric_name == "bertscore":
                if not BERTSCORE_AVAILABLE:
                    print(f"⚠️  Skipping BERTScore (not installed)")
                    continue
                metrics.append(BERTScoreMetric(**metric_params))

            elif metric_name == "bleurt":
                if not BLEURT_AVAILABLE:
                    print(f"⚠️  Skipping BLEURT (not installed)")
                    continue
                metrics.append(BLEURTMetric(**metric_params))

            elif metric_name == "llm_judge_qa":
                if not LLM_JUDGE_AVAILABLE:
                    print(f"⚠️  Skipping LLM Judge (not available)")
                    continue
                if not judge_model:
                    print(f"⚠️  Skipping LLM Judge (judge_model not configured)")
                    continue
                judgment_type = metric_params.get("judgment_type", "binary")
                batch_size = metric_params.get("batch_size", 8)
                metrics.append(
                    LLMJudgeQAMetric(
                        judge_model=judge_model,
                        judgment_type=judgment_type,
                        batch_size=batch_size,
                    )
                )

            elif metric_name == "faithfulness":
                if not FAITHFULNESS_AVAILABLE:
                    print(f"⚠️  Skipping Faithfulness (not installed)")
                    continue
                metrics.append(FaithfulnessMetric(**metric_params))

            elif metric_name == "hallucination":
                if not HALLUCINATION_AVAILABLE:
                    print(f"⚠️  Skipping Hallucination (not installed)")
                    continue
                metrics.append(HallucinationMetric(**metric_params))

            else:
                print(f"⚠️  Unknown metric: {metric_name}, skipping")

        return metrics

    @staticmethod
    def create_retriever(config: Dict[str, Any]) -> Retriever:
        """Create a retriever from configuration.

        Args:
            config: Retriever configuration dict with 'type' and retriever-specific params

        Returns:
            Instantiated Retriever

        Example:
            >>> config = {"type": "dense", "embedding_model": "all-MiniLM-L6-v2"}
            >>> retriever = ComponentFactory.create_retriever(config)
        """
        retriever_type = config.get("type", "dense")
        config_copy = dict(config)
        config_copy.pop("type", None)

        if retriever_type == "dense":
            return DenseRetriever(**config_copy)
        elif retriever_type == "sparse":
            from ragicamp.retrievers import SparseRetriever

            return SparseRetriever(**config_copy)
        else:
            raise ValueError(
                f"Unknown retriever type: {retriever_type}. " f"Available: dense, sparse"
            )
