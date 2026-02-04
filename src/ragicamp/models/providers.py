"""Model providers with lazy loading and lifecycle management.

Design principles:
- Models are not loaded until explicitly requested
- Context managers ensure proper cleanup
- Each model can use full GPU when it has exclusive access
- True batch operations for throughput
"""

import os
import shutil

# Set vLLM attention backend fallback if nvcc is not available
# This must happen before vLLM is imported
if not shutil.which("nvcc") and "VLLM_ATTENTION_BACKEND" not in os.environ:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

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
            if hasattr(self._embedder, "unload"):
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
        """Load vLLM generator with B200-optimized settings."""
        import torch
        from vllm import LLM

        # Auto-detect optimal settings based on GPU
        gpu_mem_gb = 0
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        llm_kwargs = {
            "model": self.config.model_name,
            "dtype": self.config.dtype,
            "gpu_memory_utilization": gpu_fraction,
            "trust_remote_code": self.config.trust_remote_code,
            "enable_prefix_caching": True,  # Cache common prompt prefixes
        }

        if self.config.max_model_len:
            llm_kwargs["max_model_len"] = self.config.max_model_len

        # B200 (192GB) optimizations
        if gpu_mem_gb >= 160:
            llm_kwargs["max_num_seqs"] = 512  # More concurrent sequences
            llm_kwargs["max_num_batched_tokens"] = 32768  # 32k tokens per batch
            logger.info("B200 detected (%.0fGB): using high-throughput settings", gpu_mem_gb)
        elif gpu_mem_gb >= 80:
            llm_kwargs["max_num_seqs"] = 256
            llm_kwargs["max_num_batched_tokens"] = 16384

        llm = LLM(**llm_kwargs)
        return VLLMGeneratorWrapper(llm, self.config.model_name)

    def _load_hf(self) -> "Generator":
        """Load HuggingFace generator."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.dtype == "float16" else "auto",
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        model.eval()
        
        return HFGeneratorWrapper(model, tokenizer, self.config.model_name)

    def _unload(self):
        """Unload generator and free GPU memory."""
        if self._generator is not None:
            if hasattr(self._generator, "unload"):
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


class HFGeneratorWrapper(Generator):
    """Wrapper around HuggingFace model implementing Generator protocol."""

    def __init__(self, model, tokenizer, model_name: str):
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        import torch

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.0)

        # Tokenize with padding
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        # Move to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only new tokens
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[:, input_length:]

        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def unload(self):
        import gc
        import torch

        del self._model
        del self._tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Reranker Provider
# =============================================================================


@dataclass
class RerankerConfig:
    """Configuration for reranker."""

    model_name: str = "bge"  # bge, bge-base, ms-marco, or HF path
    batch_size: int = 32


class RerankerProvider(ModelProvider):
    """Provides reranker with lazy loading and proper cleanup.

    Usage:
        provider = RerankerProvider(RerankerConfig("bge"))

        with provider.load() as reranker:
            reranked = reranker.rerank(query, documents, top_k=5)
        # Reranker unloaded, GPU memory freed
    """

    MODELS = {
        "bge": "BAAI/bge-reranker-large",
        "bge-base": "BAAI/bge-reranker-base",
        "bge-v2": "BAAI/bge-reranker-v2-m3",
        "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-large": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    }

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._reranker = None

    @property
    def model_name(self) -> str:
        return self.MODELS.get(self.config.model_name, self.config.model_name)

    @contextmanager
    def load(self, gpu_fraction: float | None = None) -> Iterator["RerankerWrapper"]:
        """Load reranker, yield it, then unload."""
        import torch
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker: %s", self.model_name)

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = CrossEncoder(
                self.model_name,
                device=device,
                trust_remote_code=True,
            )

            self._reranker = RerankerWrapper(model, self.config.batch_size)
            yield self._reranker

        finally:
            self._unload()

    def _unload(self):
        """Unload reranker and free GPU memory."""
        if self._reranker is not None:
            self._reranker.unload()
            self._reranker = None

        ResourceManager.clear_gpu_memory()
        logger.info("Reranker unloaded: %s", self.model_name)


class RerankerWrapper:
    """Wrapper around CrossEncoder implementing reranker interface."""

    def __init__(self, model, batch_size: int = 32):
        self._model = model
        self._batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int,
    ) -> list:
        """Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of Document objects
            top_k: Number to return

        Returns:
            Top-k documents sorted by reranker score
        """
        if not documents:
            return []

        # Create pairs
        pairs = [(query, doc.text) for doc in documents]

        # Score
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        return sorted_docs[:top_k]

    def batch_rerank(
        self,
        queries: list[str],
        documents_list: list[list],
        top_k: int,
    ) -> list[list]:
        """Batch rerank for multiple queries.

        Args:
            queries: List of search queries
            documents_list: List of document lists
            top_k: Number to return per query

        Returns:
            List of top-k document lists
        """
        if not queries:
            return []

        # Build all pairs
        all_pairs = []
        pair_indices = []

        for q_idx, (query, docs) in enumerate(zip(queries, documents_list)):
            for d_idx, doc in enumerate(docs):
                all_pairs.append((query, doc.text))
                pair_indices.append((q_idx, d_idx))

        if not all_pairs:
            return [[] for _ in queries]

        # Score all at once
        scores = self._model.predict(
            all_pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Assign scores
        for (q_idx, d_idx), score in zip(pair_indices, scores):
            documents_list[q_idx][d_idx].score = float(score)

        # Sort and return
        results = []
        for docs in documents_list:
            sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
            results.append(sorted_docs[:top_k])

        return results

    def unload(self):
        """Unload model."""
        import gc
        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
