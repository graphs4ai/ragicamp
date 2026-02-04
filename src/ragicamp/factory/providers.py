"""Provider factory for creating model providers from configuration.

Creates EmbedderProvider and GeneratorProvider for the clean architecture
where models are loaded/unloaded via context managers.
"""

from typing import Any

from ragicamp.core.logging import get_logger
from ragicamp.models.providers import (
    EmbedderConfig,
    EmbedderProvider,
    GeneratorConfig,
    GeneratorProvider,
)

logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating model providers from configuration.
    
    Usage:
        # From model spec string
        gen_provider = ProviderFactory.create_generator("vllm:meta-llama/Llama-3.2-3B")
        emb_provider = ProviderFactory.create_embedder("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        
        # Use with context managers
        with gen_provider.load() as generator:
            answers = generator.batch_generate(prompts)
    """
    
    @staticmethod
    def parse_generator_spec(
        spec: str,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> GeneratorConfig:
        """Parse a model spec string into GeneratorConfig.
        
        Args:
            spec: Model spec like 'vllm:meta-llama/Llama-3.2-3B' or 'hf:google/gemma-2b-it'
            quantization: Quantization setting (awq, gptq, 4bit, 8bit)
            **kwargs: Additional config params
        
        Returns:
            GeneratorConfig
        """
        if ":" in spec:
            provider, model_name = spec.split(":", 1)
        else:
            provider, model_name = "vllm", spec
        
        backend = "vllm" if provider == "vllm" else "hf"
        
        return GeneratorConfig(
            model_name=model_name,
            backend=backend,
            quantization=quantization,
            dtype=kwargs.get("dtype", "auto"),
            trust_remote_code=kwargs.get("trust_remote_code", True),
            max_model_len=kwargs.get("max_model_len"),
        )
    
    @staticmethod
    def create_generator(
        spec: str | dict[str, Any],
        quantization: str | None = None,
        **kwargs: Any,
    ) -> GeneratorProvider:
        """Create a GeneratorProvider from spec.
        
        Args:
            spec: Model spec string or config dict
            quantization: Quantization setting
            **kwargs: Additional params
        
        Returns:
            GeneratorProvider ready for .load()
        """
        if isinstance(spec, dict):
            config = GeneratorConfig(
                model_name=spec.get("model_name", spec.get("name", "")),
                backend=spec.get("backend", spec.get("type", "vllm")),
                quantization=spec.get("quantization"),
                dtype=spec.get("dtype", "auto"),
                trust_remote_code=spec.get("trust_remote_code", True),
                max_model_len=spec.get("max_model_len"),
            )
        else:
            config = ProviderFactory.parse_generator_spec(spec, quantization, **kwargs)
        
        logger.info("Creating generator provider: %s (%s)", config.model_name, config.backend)
        return GeneratorProvider(config)
    
    @staticmethod
    def parse_embedder_spec(
        spec: str,
        backend: str = "vllm",
        **kwargs: Any,
    ) -> EmbedderConfig:
        """Parse an embedder spec into EmbedderConfig.
        
        Args:
            spec: Model name like 'BAAI/bge-large-en-v1.5'
            backend: 'vllm' or 'sentence_transformers'
            **kwargs: Additional config params
        
        Returns:
            EmbedderConfig
        """
        return EmbedderConfig(
            model_name=spec,
            backend=backend,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
    
    @staticmethod
    def create_embedder(
        spec: str | dict[str, Any],
        backend: str = "vllm",
        **kwargs: Any,
    ) -> EmbedderProvider:
        """Create an EmbedderProvider from spec.
        
        Args:
            spec: Model name or config dict
            backend: 'vllm' or 'sentence_transformers'
            **kwargs: Additional params
        
        Returns:
            EmbedderProvider ready for .load()
        """
        if isinstance(spec, dict):
            config = EmbedderConfig(
                model_name=spec.get("model_name", spec.get("embedding_model", "")),
                backend=spec.get("backend", spec.get("embedding_backend", backend)),
                trust_remote_code=spec.get("trust_remote_code", True),
            )
        else:
            config = ProviderFactory.parse_embedder_spec(spec, backend, **kwargs)
        
        logger.info("Creating embedder provider: %s (%s)", config.model_name, config.backend)
        return EmbedderProvider(config)
