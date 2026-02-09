"""Configuration schemas for RAGiCamp experiments.

This module provides Pydantic models for validating and standardizing
experiment configurations.
"""

from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, validator

from ragicamp.core.constants import Defaults


class ModelConfig(BaseModel):
    """Model configuration.

    Supports three model types:
    - huggingface: Standard HuggingFace transformers (with optional BitsAndBytes quantization)
    - vllm: vLLM backend with PagedAttention (recommended for long context)
    - openai: OpenAI API models
    """

    type: str = Field(default="huggingface", description="Model type: huggingface, vllm, openai")
    model_name: str = Field(..., description="Model identifier")
    device: str = Field(default="cuda", description="Device (cuda/cpu) - for huggingface only")

    # HuggingFace-specific quantization (BitsAndBytes)
    load_in_8bit: bool = Field(default=False, description="Use 8-bit quantization (HuggingFace)")
    load_in_4bit: bool = Field(default=False, description="Use 4-bit quantization (HuggingFace)")

    # vLLM-specific options
    dtype: str = Field(
        default="bfloat16",
        description="Model dtype for vLLM: 'bfloat16', 'float16', 'float32', 'auto'",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="vLLM quantization: 'awq', 'gptq', 'squeezellm', or None (no quantization)",
    )
    gpu_memory_utilization: float = Field(
        default=0.50,
        description="Fraction of GPU memory to use (vLLM). Default 0.50 leaves room for embedder (0.25).",
    )
    max_model_len: Optional[int] = Field(
        default=None,
        description="Maximum context length (vLLM). If None, uses model's default.",
    )
    tensor_parallel_size: int = Field(
        default=1,
        description="Number of GPUs for tensor parallelism (vLLM).",
    )
    kv_cache_dtype: Optional[str] = Field(
        default=None,
        description="KV cache dtype (vLLM): 'auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'. "
        "Use 'fp8' for ~2x KV cache memory reduction without weight quantization.",
    )
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable automatic prefix caching for efficiency (vLLM).",
    )

    # Generation parameters
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


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    strategy: str = Field(
        default="recursive",
        description="Chunking strategy: 'fixed', 'sentence', 'paragraph', 'recursive'",
    )
    chunk_size: int = Field(default=Defaults.CHUNK_SIZE, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=Defaults.CHUNK_OVERLAP, description="Overlap between chunks")
    min_chunk_size: int = Field(default=50, description="Minimum chunk size (discard smaller)")

    @validator("strategy")
    def validate_strategy(cls, v):
        """Validate chunking strategy."""
        allowed = ["fixed", "sentence", "paragraph", "recursive"]
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{v}'")
        return v


class RetrieverConfig(BaseModel):
    """Retriever configuration."""

    type: str = Field(default="dense", description="Retriever type")
    name: str = Field(default="retriever", description="Retriever name")
    embedding_model: str = Field(default=Defaults.EMBEDDING_MODEL, description="Embedding model")
    index_type: str = Field(default="flat", description="Index type")
    artifact_path: Optional[str] = Field(default=None, description="Path to saved artifact")
    chunking: Optional[ChunkingConfig] = Field(
        default=None, description="Document chunking config (for indexing)"
    )

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
    params: dict[str, Any] = Field(default_factory=dict, description="Metric parameters")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    mode: str = Field(
        default="both",
        description="Evaluation mode: 'generate' (predictions only), 'evaluate' (metrics only), or 'both'",
    )
    batch_size: Optional[int] = Field(default=None, description="Batch size for evaluation")
    num_examples: Optional[int] = Field(default=None, description="Number of examples to evaluate")
    predictions_file: Optional[str] = Field(
        default=None, description="Path to predictions file (required for 'evaluate' mode)"
    )

    # Checkpointing (existing)
    checkpoint_every: Optional[int] = Field(
        default=None, description="Save checkpoint every N examples"
    )
    resume_from_checkpoint: bool = Field(
        default=False, description="Resume from checkpoint if exists"
    )
    retry_failures: bool = Field(default=False, description="Retry failed examples on resume")

    # State management (new)
    save_state: bool = Field(
        default=True, description="Save experiment state for phase-level resumption"
    )
    state_file: Optional[str] = Field(default=None, description="Path to state file")
    force_rerun_phases: list[str] = Field(
        default_factory=list, description="List of phase names to force rerun (e.g., ['metrics'])"
    )

    @validator("mode")
    def validate_mode(cls, v):
        """Validate mode is one of the allowed values."""
        allowed = ["generate", "evaluate", "both"]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got '{v}'")
        return v

    @validator("predictions_file", always=True)
    def validate_predictions_file(cls, v, values):
        """Ensure predictions_file is set for evaluate mode."""
        mode = values.get("mode")
        if mode == "evaluate" and not v:
            raise ValueError("predictions_file must be set when mode='evaluate'")
        return v

    class Config:
        extra = "allow"


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    enabled: bool = Field(default=True, description="Enable MLflow tracking")
    experiment_name: Optional[str] = Field(default=None, description="MLflow experiment name")
    tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    run_name: Optional[str] = Field(default=None, description="MLflow run name")
    tags: dict[str, str] = Field(default_factory=dict, description="MLflow run tags")
    log_artifacts: bool = Field(default=True, description="Log artifacts to MLflow")
    log_models: bool = Field(default=False, description="Log models to MLflow (slower)")


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


class OptunaConfig(BaseModel):
    """Optuna hyperparameter optimization configuration."""

    enabled: bool = Field(default=False, description="Enable Optuna optimization")
    n_trials: int = Field(default=20, description="Number of optimization trials")
    study_name: Optional[str] = Field(default=None, description="Optuna study name")
    storage: Optional[str] = Field(default=None, description="Optuna storage URI (for persistence)")
    direction: str = Field(
        default="maximize", description="Optimization direction (maximize/minimize)"
    )
    metric_to_optimize: str = Field(default="f1", description="Metric name to optimize")

    # Parameter search spaces (examples)
    search_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter search spaces (e.g., {'top_k': [1, 20], 'temperature': [0.1, 2.0]})",
    )


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    # Fields required only for generate/both modes
    agent: Optional[AgentConfig] = Field(None, description="Agent configuration")
    model: Optional[ModelConfig] = Field(None, description="Model configuration")
    dataset: Optional[DatasetConfig] = Field(None, description="Dataset configuration")

    # Always required
    metrics: list[Union[str, dict[str, Any]]] = Field(..., description="Metrics configuration")

    # Optional fields
    judge_model: Optional[ModelConfig] = Field(
        default=None, description="Judge model for LLM metrics"
    )
    retriever: Optional[RetrieverConfig] = Field(
        default=None, description="Retriever configuration"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation settings"
    )
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output settings")
    training: Optional[TrainingConfig] = Field(default=None, description="Training settings")
    mlflow: MLflowConfig = Field(
        default_factory=MLflowConfig, description="MLflow tracking settings"
    )
    optuna: Optional[OptunaConfig] = Field(default=None, description="Optuna optimization settings")

    @validator("agent")
    def validate_agent_for_mode(cls, v, values):
        """Validate that agent is provided when needed."""
        evaluation = values.get("evaluation")
        if evaluation and evaluation.mode in ["generate", "both"] and not v:
            raise ValueError("agent is required for 'generate' and 'both' modes")
        return v

    @validator("model")
    def validate_model_for_mode(cls, v, values):
        """Validate that model is provided when needed."""
        evaluation = values.get("evaluation")
        if evaluation and evaluation.mode in ["generate", "both"] and not v:
            raise ValueError("model is required for 'generate' and 'both' modes")
        return v

    @validator("dataset")
    def validate_dataset_for_mode(cls, v, values):
        """Validate that dataset is provided when needed."""
        evaluation = values.get("evaluation")
        if evaluation and evaluation.mode in ["generate", "both"] and not v:
            raise ValueError("dataset is required for 'generate' and 'both' modes")
        return v

    @validator("retriever")
    def validate_retriever(cls, v, values):
        """Validate that RAG agents have a retriever."""
        agent_config = values.get("agent")
        if agent_config:
            agent_type = agent_config.type
            if agent_type in ["fixed_rag", "iterative_rag", "self_rag"] and not v:
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
                normalized.append({"name": metric["name"], "params": metric.get("params", {})})
            else:
                raise ValueError(f"Invalid metric type: {type(metric)}")
        return normalized

    class Config:
        extra = "forbid"  # Don't allow extra fields at top level


def parse_metric_config(metric: Union[str, dict[str, Any]]) -> MetricConfig:
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
