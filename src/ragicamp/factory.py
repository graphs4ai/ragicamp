"""Factory for creating RAGiCamp components from configuration dictionaries.

This module provides a centralized way to instantiate models, agents, datasets,
metrics, and retrievers from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)

from ragicamp.agents import (
    DirectLLMAgent,
    FixedRAGAgent,
    RAGAgent,
)
from ragicamp.datasets import (
    HotpotQADataset,
    NaturalQuestionsDataset,
    QADataset,
    TriviaQADataset,
)
from ragicamp.metrics import Metric
from ragicamp.models import HuggingFaceModel, LanguageModel, OpenAIModel
from ragicamp.retrievers import DenseRetriever, Retriever


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration.

    Args:
        config: Model config dict with 'type' and model-specific params

    Raises:
        ValueError: If config is invalid
    """
    model_type = config.get("type", "huggingface")
    if model_type not in ("huggingface", "openai"):
        raise ValueError(f"Invalid model type: {model_type}. Valid: huggingface, openai")
    if model_type == "huggingface" and not config.get("model_name"):
        raise ValueError("HuggingFace model requires 'model_name'")
    if model_type == "openai" and not config.get("name"):
        raise ValueError("OpenAI model requires 'name'")


class ComponentFactory:
    """Factory for creating RAGiCamp components from config dictionaries.

    Supports extension via registration:
        @ComponentFactory.register_model("anthropic")
        class AnthropicModel(LanguageModel):
            ...
    """

    # Plugin registries
    _custom_models: Dict[str, type] = {}
    _custom_agents: Dict[str, type] = {}
    _custom_metrics: Dict[str, type] = {}
    _custom_retrievers: Dict[str, type] = {}

    @classmethod
    def register_model(cls, name: str):
        """Register a custom model type.

        Usage:
            @ComponentFactory.register_model("anthropic")
            class AnthropicModel(LanguageModel):
                ...
        """

        def decorator(model_class: type) -> type:
            cls._custom_models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def register_agent(cls, name: str):
        """Register a custom agent type."""

        def decorator(agent_class: type) -> type:
            cls._custom_agents[name] = agent_class
            return agent_class

        return decorator

    @classmethod
    def register_metric(cls, name: str):
        """Register a custom metric type."""

        def decorator(metric_class: type) -> type:
            cls._custom_metrics[name] = metric_class
            return metric_class

        return decorator

    @classmethod
    def register_retriever(cls, name: str):
        """Register a custom retriever type."""

        def decorator(retriever_class: type) -> type:
            cls._custom_retrievers[name] = retriever_class
            return retriever_class

        return decorator

    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> LanguageModel:
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

        # Check custom registry first
        if model_type in cls._custom_models:
            return cls._custom_models[model_type](**config_copy)

        # Built-in types
        if model_type == "huggingface":
            return HuggingFaceModel(**config_copy)
        elif model_type == "openai":
            return OpenAIModel(**config_copy)
        else:
            available = ["huggingface", "openai"] + list(cls._custom_models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

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

        else:
            raise ValueError(
                f"Unknown agent type: {agent_type}. " f"Available: direct_llm, fixed_rag"
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
                logger.debug("Filtered: %d → %d examples", original_size, len(dataset))

        # Limit if requested
        if num_examples and len(dataset) > num_examples:
            dataset.examples = dataset.examples[:num_examples]
            logger.debug("Dataset size: %d", len(dataset))

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
                    logger.warning("Skipping BERTScore (not installed)")
                    continue
                metrics.append(BERTScoreMetric(**metric_params))

            elif metric_name == "bleurt":
                if not BLEURT_AVAILABLE:
                    logger.warning("Skipping BLEURT (not installed)")
                    continue
                metrics.append(BLEURTMetric(**metric_params))

            elif metric_name == "llm_judge_qa":
                if not LLM_JUDGE_AVAILABLE:
                    logger.warning("Skipping LLM Judge (not available)")
                    continue
                if not judge_model:
                    logger.warning("Skipping LLM Judge (judge_model not configured)")
                    continue
                judgment_type = metric_params.get("judgment_type", "binary")
                # Support both batch_size (legacy) and max_concurrent (new)
                max_concurrent = metric_params.get(
                    "max_concurrent", metric_params.get("batch_size", 20)
                )
                metrics.append(
                    LLMJudgeQAMetric(
                        judge_model=judge_model,
                        judgment_type=judgment_type,
                        max_concurrent=max_concurrent,
                    )
                )

            elif metric_name == "faithfulness":
                if not FAITHFULNESS_AVAILABLE:
                    logger.warning("Skipping Faithfulness (not installed)")
                    continue
                metrics.append(FaithfulnessMetric(**metric_params))

            elif metric_name == "hallucination":
                if not HALLUCINATION_AVAILABLE:
                    logger.warning("Skipping Hallucination (not installed)")
                    continue
                metrics.append(HallucinationMetric(**metric_params))

            # Try Ragas metrics (with ragas_ prefix or direct name)
            elif metric_name.startswith("ragas_") or metric_name in [
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevancy",
                "answer_similarity",
                "answer_correctness",
            ]:
                try:
                    from ragicamp.metrics.ragas_adapter import create_ragas_metric

                    # Remove ragas_ prefix if present
                    ragas_name = metric_name.replace("ragas_", "")
                    logger.debug("Using Ragas metric: %s", ragas_name)
                    metrics.append(create_ragas_metric(ragas_name, **metric_params))
                except ImportError:
                    logger.warning("Ragas not installed, skipping %s", metric_name)
                    continue
                except Exception as e:
                    logger.warning("Failed to create Ragas metric %s: %s", metric_name, e)
                    continue

            else:
                logger.warning("Unknown metric: %s, skipping", metric_name)

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
