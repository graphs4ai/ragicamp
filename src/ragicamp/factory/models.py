"""Model factory for creating language models from configuration.

Provides ModelFactory for creating HuggingFace, OpenAI, and vLLM models.
"""

from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.models import _VLLM_AVAILABLE, HuggingFaceModel, LanguageModel, OpenAIModel, VLLMModel

logger = get_logger(__name__)


def validate_model_config(config: dict[str, Any]) -> None:
    """Validate model configuration.

    Args:
        config: Model config dict with 'type' and model-specific params

    Raises:
        ValueError: If config is invalid
    """
    model_type = config.get("type", "huggingface")
    valid_types = ("huggingface", "openai", "vllm")
    if model_type not in valid_types:
        raise ValueError(f"Invalid model type: {model_type}. Valid: {', '.join(valid_types)}")
    if model_type == "huggingface" and not config.get("model_name"):
        raise ValueError("HuggingFace model requires 'model_name'")
    if model_type == "vllm" and not config.get("model_name"):
        raise ValueError("vLLM model requires 'model_name'")
    if model_type == "openai" and not config.get("name"):
        raise ValueError("OpenAI model requires 'name'")


class ModelFactory:
    """Factory for creating language models from configuration.

    Supports extension via registration:
        @ModelFactory.register("anthropic")
        class AnthropicModel(LanguageModel):
            ...
    """

    # Custom model registry
    _custom_models: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Register a custom model type.

        Usage:
            @ModelFactory.register("anthropic")
            class AnthropicModel(LanguageModel):
                ...
        """

        def decorator(model_class: type) -> type:
            cls._custom_models[name] = model_class
            return model_class

        return decorator

    @staticmethod
    def parse_spec(
        spec: str,
        quantization: str = "none",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Parse a model spec string into a config dict.

        Args:
            spec: Model spec like 'hf:google/gemma-2b-it', 'vllm:meta-llama/Llama-2-7b',
                  or 'openai:gpt-4o-mini'
            quantization: Quantization setting:
                         - For HuggingFace: '4bit', '8bit', 'none' (default: 'none')
                         - For vLLM: 'awq', 'gptq', 'squeezellm', 'none' (default: 'none')
            **kwargs: Additional model parameters

        Returns:
            Config dict suitable for create()

        Example:
            >>> config = ModelFactory.parse_spec("vllm:meta-llama/Llama-2-7b")
            >>> model = ModelFactory.create(config)
        """
        if ":" in spec:
            provider, model_name = spec.split(":", 1)
        else:
            provider, model_name = "openai", spec

        if provider in ("hf", "huggingface"):
            config = {
                "type": "huggingface",
                "model_name": model_name,
                "load_in_4bit": quantization == "4bit",
                "load_in_8bit": quantization == "8bit",
            }
        elif provider == "vllm":
            config = {
                "type": "vllm",
                "model_name": model_name,
                "dtype": "bfloat16",
            }
            if quantization and quantization != "none":
                config["quantization"] = quantization
        elif provider == "openai":
            config = {
                "type": "openai",
                "name": model_name,
                "temperature": 0.0,
            }
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'hf:', 'vllm:', or 'openai:'")

        config.update(kwargs)
        return config

    @classmethod
    def create(cls, config: dict[str, Any]) -> LanguageModel:
        """Create a language model from configuration.

        Args:
            config: Model configuration dict with 'type' and model-specific params

        Returns:
            Instantiated LanguageModel

        Example:
            >>> config = {"type": "huggingface", "model_name": "google/gemma-2-2b-it"}
            >>> model = ModelFactory.create(config)
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
        elif model_type == "vllm":
            if not _VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM is not installed. Install it with: pip install vllm\n"
                    "Note: vLLM requires CUDA and a compatible GPU."
                )
            return VLLMModel(**config_copy)
        elif model_type == "openai":
            return OpenAIModel(**config_copy)
        else:
            available = ["huggingface", "vllm", "openai"] + list(cls._custom_models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
