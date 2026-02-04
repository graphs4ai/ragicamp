"""Model providers with lazy loading and lifecycle management.

Design principles:
- Models are not loaded until explicitly requested
- Context managers ensure proper cleanup
- Each model can use full GPU when it has exclusive access
- True batch operations for throughput
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ragicamp.core.constants import Defaults
from ragicamp.core.logging import get_logger
from ragicamp.utils.resource_manager import ResourceManager

logger = get_logger(__name__)


# =============================================================================
# Base Provider Protocol
# =============================================================================

class ModelProvider(ABC):
    """Base class for lazy model loading with lifecycle management."""
    
    @abstractmethod
    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator[Any]:
        """Load model and yield it, then unload on exit.
        
        Usage:
            with provider.load() as model:
                result = model.encode(texts)
            # Model automatically unloaded
        """
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier."""
        ...


# =============================================================================
# Embedder Provider
# =============================================================================

@dataclass
class EmbedderConfig:
    """Configuration for embedder."""
    model_name: str
    backend: str = "vllm"  # "vllm" or "sentence_transformers"
    trust_remote_code: bool = True


class EmbedderProvider(ModelProvider):
    """Provides embedder with lazy loading and proper cleanup.
    
    Usage:
        provider = EmbedderProvider(EmbedderConfig("BAAI/bge-large-en-v1.5"))
        
        with provider.load() as embedder:
            embeddings = embedder.batch_encode(texts)
        # Embedder unloaded, GPU memory freed
    """
    
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._embedder = None
    
    @property
    def model_name(self) -> str:
        return self.config.model_name
    
    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator["Embedder"]:
        """Load embedder, yield it, then unload."""
        if gpu_fraction is None:
            gpu_fraction = Defaults.VLLM_GPU_MEMORY_FRACTION_FULL
        
        logger.info("Loading embedder: %s (gpu=%.0f%%)", self.model_name, gpu_fraction * 100)
        
        try:
            if self.config.backend == "vllm":
                embedder = self._load_vllm(gpu_fraction)
            else:
                embedder = self._load_sentence_transformers()
            
            self._embedder = embedder
            yield embedder
            
        finally:
            self._unload()
    
    def _load_vllm(self, gpu_fraction: float) -> "Embedder":
        """Load vLLM embedder."""
        from ragicamp.models.vllm_embedder import VLLMEmbedder
        
        vllm_embedder = VLLMEmbedder(
            model_name=self.config.model_name,
            gpu_memory_fraction=gpu_fraction,
            trust_remote_code=self.config.trust_remote_code,
        )
        return VLLMEmbedderWrapper(vllm_embedder)
    
    def _load_sentence_transformers(self) -> "Embedder":
        """Load sentence-transformers embedder."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(self.config.model_name)
        return SentenceTransformerWrapper(model)
    
    def _unload(self):
        """Unload embedder and free GPU memory."""
        if self._embedder is not None:
            if hasattr(self._embedder, 'unload'):
                self._embedder.unload()
            self._embedder = None
        
        ResourceManager.clear_gpu_memory()
        logger.info("Embedder unloaded: %s", self.model_name)


class Embedder(ABC):
    """Protocol for embedders with batch operations."""
    
    @abstractmethod
    def batch_encode(self, texts: list[str]) -> Any:
        """Encode multiple texts to embeddings."""
        ...
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        ...
    
    def unload(self) -> None:
        """Unload model (optional override)."""
        pass


class VLLMEmbedderWrapper(Embedder):
    """Wrapper around VLLMEmbedder implementing Embedder protocol."""
    
    def __init__(self, embedder):
        self._embedder = embedder
    
    def batch_encode(self, texts: list[str]) -> Any:
        return self._embedder.encode(texts)
    
    def get_dimension(self) -> int:
        return self._embedder.get_sentence_embedding_dimension()
    
    def unload(self):
        self._embedder.unload()


class SentenceTransformerWrapper(Embedder):
    """Wrapper around SentenceTransformer implementing Embedder protocol."""
    
    def __init__(self, model):
        self._model = model
    
    def batch_encode(self, texts: list[str]) -> Any:
        return self._model.encode(texts, show_progress_bar=True)
    
    def get_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


# =============================================================================
# Generator Provider
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for generator."""
    model_name: str
    backend: str = "vllm"  # "vllm" or "hf"
    quantization: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    max_model_len: int | None = None


class GeneratorProvider(ModelProvider):
    """Provides generator with lazy loading and proper cleanup.
    
    Usage:
        provider = GeneratorProvider(GeneratorConfig("meta-llama/Llama-3.2-3B-Instruct"))
        
        with provider.load() as generator:
            answers = generator.batch_generate(prompts)
        # Generator unloaded, GPU memory freed
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._generator = None
    
    @property
    def model_name(self) -> str:
        return self.config.model_name
    
    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator["Generator"]:
        """Load generator, yield it, then unload."""
        if gpu_fraction is None:
            gpu_fraction = Defaults.VLLM_GPU_MEMORY_FRACTION_FULL
        
        logger.info("Loading generator: %s (gpu=%.0f%%)", self.model_name, gpu_fraction * 100)
        
        try:
            if self.config.backend == "vllm":
                generator = self._load_vllm(gpu_fraction)
            else:
                generator = self._load_hf()
            
            self._generator = generator
            yield generator
            
        finally:
            self._unload()
    
    def _load_vllm(self, gpu_fraction: float) -> "Generator":
        """Load vLLM generator."""
        from vllm import LLM
        
        llm = LLM(
            model=self.config.model_name,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            gpu_memory_utilization=gpu_fraction,
            trust_remote_code=self.config.trust_remote_code,
            max_model_len=self.config.max_model_len,
        )
        return VLLMGeneratorWrapper(llm, self.config.model_name)
    
    def _load_hf(self) -> "Generator":
        """Load HuggingFace generator."""
        raise NotImplementedError("HF backend not yet implemented")
    
    def _unload(self):
        """Unload generator and free GPU memory."""
        if self._generator is not None:
            if hasattr(self._generator, 'unload'):
                self._generator.unload()
            self._generator = None
        
        ResourceManager.clear_gpu_memory()
        logger.info("Generator unloaded: %s", self.model_name)


class Generator(ABC):
    """Protocol for generators with batch operations."""
    
    @abstractmethod
    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for multiple prompts."""
        ...
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response for single prompt."""
        return self.batch_generate([prompt], **kwargs)[0]
    
    def unload(self) -> None:
        """Unload model (optional override)."""
        pass


class VLLMGeneratorWrapper(Generator):
    """Wrapper around vLLM LLM implementing Generator protocol."""
    
    def __init__(self, llm, model_name: str):
        self._llm = llm
        self._model_name = model_name
    
    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.0),
        )
        
        outputs = self._llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def unload(self):
        # vLLM doesn't have explicit unload, but we can delete the reference
        del self._llm
