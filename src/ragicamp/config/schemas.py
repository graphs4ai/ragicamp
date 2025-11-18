"""Configuration schemas for RAGiCamp experiments.

This module provides Pydantic models for validating and standardizing
experiment configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration."""

    type: str = Field(default="huggingface", description="Model type")
    model_name: str = Field(..., description="Model identifier")
    device: str = Field(default="cuda", description="Device (cuda/cpu)")
    load_in_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    load_in_4bit: bool = Field(default=False, description="Use 4-bit quantization")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")

    class Config:
        extra = "allow"  # Allow additional fields for model-specific params


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(..., description="Dataset name")
    split: str = Field(default="validation", description="Dataset split")
    num_examples: Optional[int] = Field(default=None, description="Number of examples to use")
    filter_no_answer: bool = Field(default=True, description="Filter questions without answers")
    cache_dir: Path = Field(default=Path("data/datasets"), description="Cache directory")

    class Config:
        extra = "allow"


class RetrieverConfig(BaseModel):
    """Retriever configuration."""

    type: str = Field(default="dense", description="Retriever type")
    name: str = Field(default="retriever", description="Retriever name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    index_type: str = Field(default="flat", description="Index type")
    artifact_path: Optional[str] = Field(default=None, description="Path to saved artifact")

    class Config:
        extra = "allow"


class AgentConfig(BaseModel):
    """Agent configuration."""

    type: str = Field(..., description="Agent type")
    name: str = Field(..., description="Agent name")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    top_k: Optional[int] = Field(default=5, description="Top-k documents for RAG")

    class Config:
        extra = "allow"


class MetricConfig(BaseModel):
    """Metric configuration (can be string or dict)."""

    name: str = Field(..., description="Metric name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Metric parameters")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    batch_size: Optional[int] = Field(default=None, description="Batch size for evaluation")
    num_examples: Optional[int] = Field(default=None, description="Number of examples to evaluate")

    class Config:
        extra = "allow"


class OutputConfig(BaseModel):
    """Output configuration."""

    save_predictions: bool = Field(default=False, description="Save predictions to file")
    output_path: Optional[str] = Field(default=None, description="Output file path")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")

    @validator("output_path")
    def validate_output_path(cls, v, values):
        """Ensure output_path is set if save_predictions is True."""
        if values.get("save_predictions") and not v:
            raise ValueError("output_path must be set when save_predictions is True")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""

    num_epochs: int = Field(default=1, description="Number of training epochs")
    eval_interval: int = Field(default=100, description="Evaluation interval")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    batch_size: int = Field(default=32, description="Training batch size")

    class Config:
        extra = "allow"


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    # Required fields
    agent: AgentConfig = Field(..., description="Agent configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    metrics: List[Union[str, Dict[str, Any]]] = Field(..., description="Metrics configuration")

    # Optional fields
    judge_model: Optional[ModelConfig] = Field(default=None, description="Judge model for LLM metrics")
    retriever: Optional[RetrieverConfig] = Field(default=None, description="Retriever configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation settings")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output settings")
    training: Optional[TrainingConfig] = Field(default=None, description="Training settings")

    @validator("retriever")
    def validate_retriever(cls, v, values):
        """Validate that RAG agents have a retriever."""
        agent_config = values.get("agent")
        if agent_config:
            agent_type = agent_config.type
            if agent_type in ["fixed_rag", "bandit_rag", "mdp_rag"] and not v:
                raise ValueError(f"{agent_type} agent requires a retriever configuration")
        return v

    @validator("metrics", pre=True)
    def normalize_metrics(cls, v):
        """Normalize metrics to list of MetricConfig objects."""
        normalized = []
        for metric in v:
            if isinstance(metric, str):
                normalized.append({"name": metric, "params": {}})
            elif isinstance(metric, dict):
                if "name" not in metric:
                    raise ValueError("Metric dict must have 'name' field")
                normalized.append({
                    "name": metric["name"],
                    "params": metric.get("params", {})
                })
            else:
                raise ValueError(f"Invalid metric type: {type(metric)}")
        return normalized

    class Config:
        extra = "forbid"  # Don't allow extra fields at top level


def parse_metric_config(metric: Union[str, Dict[str, Any]]) -> MetricConfig:
    """Parse a metric configuration.

    Args:
        metric: Either a string (metric name) or dict with name and params

    Returns:
        MetricConfig object
    """
    if isinstance(metric, str):
        return MetricConfig(name=metric, params={})
    elif isinstance(metric, dict):
        return MetricConfig(**metric)
    else:
        raise ValueError(f"Invalid metric type: {type(metric)}")

