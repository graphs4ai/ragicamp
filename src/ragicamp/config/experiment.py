"""Experiment configuration using dataclasses.

Simple, type-safe configuration system using Python dataclasses and YAML.
No heavy dependencies like Hydra - just clean, simple configs.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

from ragicamp.corpus.base import CorpusConfig


@dataclass
class RetrieverConfig:
    """Retriever configuration.

    Attributes:
        type: Retriever type ("dense", "sparse", "hybrid")
        embedding_model: Model for dense retrieval
        top_k: Number of documents to retrieve
        index_type: FAISS index type ("flat", "ivf", "hnsw")
    """

    type: str = "dense"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    index_type: str = "flat"


@dataclass
class ModelConfig:
    """Language model configuration.

    Attributes:
        name: HuggingFace model identifier
        load_in_8bit: Use 8-bit quantization for memory efficiency
        load_in_4bit: Use 4-bit quantization (even more efficient)
        device: Device to run on ("cuda", "cpu")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """

    name: str = "google/gemma-2-2b-it"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration.

    Attributes:
        dataset: Dataset name ("natural_questions", "hotpotqa", "triviaqa")
        split: Dataset split ("train", "validation", "test")
        num_examples: Number of examples to evaluate
        filter_no_answer: Filter questions without explicit answers
        metrics: List of metrics to compute
    """

    dataset: str = "natural_questions"
    split: str = "validation"
    num_examples: int = 100
    filter_no_answer: bool = True
    metrics: List[str] = field(default_factory=lambda: ["exact_match", "f1"])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Single source of truth for an experiment run. Can be serialized to/from YAML.

    Attributes:
        name: Experiment identifier
        corpus: Document corpus configuration
        retriever: Retriever configuration
        model: Language model configuration
        evaluation: Evaluation configuration
        output_dir: Where to save results (relative to outputs/)

    Example:
        >>> config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")
        >>> results = run_experiment(config)
        >>> config.to_yaml("outputs/my_experiment/config.yaml")
    """

    name: str
    corpus: CorpusConfig
    retriever: RetrieverConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output_dir: str = "experiments"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            ExperimentConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert nested dicts to dataclass instances
        if "corpus" in data and isinstance(data["corpus"], dict):
            data["corpus"] = CorpusConfig(**data["corpus"])

        if "retriever" in data and isinstance(data["retriever"], dict):
            data["retriever"] = RetrieverConfig(**data["retriever"])

        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])

        if "evaluation" in data and isinstance(data["evaluation"], dict):
            data["evaluation"] = EvaluationConfig(**data["evaluation"])

        return cls(**data)

    def to_yaml(self, path: str):
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return asdict(self)

    def __str__(self) -> str:
        return f"Experiment: {self.name} ({self.evaluation.dataset})"


# Preset configurations for common experiments


def create_baseline_config(dataset: str = "natural_questions") -> ExperimentConfig:
    """Create baseline DirectLLM configuration.

    Args:
        dataset: Dataset to evaluate on

    Returns:
        ExperimentConfig for baseline experiment
    """
    return ExperimentConfig(
        name=f"baseline_direct_llm_{dataset}",
        corpus=CorpusConfig(name="none", source="none", version="none"),
        retriever=RetrieverConfig(type="none"),
        model=ModelConfig(name="google/gemma-2-2b-it", load_in_8bit=True),
        evaluation=EvaluationConfig(
            dataset=dataset, num_examples=100, metrics=["exact_match", "f1"]
        ),
    )


def create_fixed_rag_config(
    dataset: str = "natural_questions", corpus_version: str = "20231101.simple"
) -> ExperimentConfig:
    """Create FixedRAG configuration.

    Args:
        dataset: Dataset to evaluate on
        corpus_version: Wikipedia version to use

    Returns:
        ExperimentConfig for FixedRAG experiment
    """
    return ExperimentConfig(
        name=f"fixed_rag_{dataset}",
        corpus=CorpusConfig(
            name=f"wikipedia_{corpus_version.split('.')[-1]}",
            source="wikimedia/wikipedia",
            version=corpus_version,
            max_docs=10000,  # For testing, remove for full
        ),
        retriever=RetrieverConfig(type="dense", embedding_model="all-MiniLM-L6-v2", top_k=3),
        model=ModelConfig(name="google/gemma-2-2b-it", load_in_8bit=True),
        evaluation=EvaluationConfig(
            dataset=dataset, num_examples=100, metrics=["exact_match", "f1"]
        ),
    )
