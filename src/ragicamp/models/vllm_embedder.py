"""vLLM-based embedding model for high-throughput embedding generation.

Uses vLLM's continuous batching for faster embedding than sentence-transformers.
See: https://docs.vllm.ai/en/latest/getting_started/examples/embedding.html

Supported models (examples):
- intfloat/e5-mistral-7b-instruct
- Alibaba-NLP/gte-Qwen2-7B-instruct
- BAAI/bge-en-icl (instruction-following)
- Salesforce/SFR-Embedding-Mistral

Note: Not all sentence-transformer models are supported by vLLM.
Check vLLM model compatibility before using.
"""

import os
from typing import TYPE_CHECKING, Optional

# Disable tokenizers parallelism to avoid fork warnings with multiprocessing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from ragicamp.core.logging import get_logger

if TYPE_CHECKING:
    import vllm

logger = get_logger(__name__)


class VLLMEmbedder:
    """vLLM-based embedding model with continuous batching.

    Provides a similar interface to SentenceTransformer.encode() but uses
    vLLM for faster GPU inference with continuous batching.
    """

    def __init__(
        self,
        model_name: str,
        gpu_memory_fraction: float = 0.9,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
    ):
        """Initialize vLLM embedder.

        Args:
            model_name: HuggingFace model name (must be vLLM-compatible)
            gpu_memory_fraction: Fraction of GPU memory to use (0.9 for index building)
            enforce_eager: Use eager mode (False = use CUDA graphs for speed)
            trust_remote_code: Trust remote code in model
        """
        self.model_name = model_name
        self.gpu_memory_fraction = gpu_memory_fraction
        self.enforce_eager = enforce_eager
        self.trust_remote_code = trust_remote_code

        self._llm: Optional[vllm.LLM] = None
        self._embedding_dim: Optional[int] = None

    @property
    def llm(self):
        """Lazy load the vLLM model."""
        if self._llm is None:
            from vllm import LLM

            logger.info(
                "Loading vLLM embedding model: %s (gpu_mem=%.1f%%)",
                self.model_name,
                self.gpu_memory_fraction * 100,
            )

            self._llm = LLM(
                model=self.model_name,
                task="embed",
                trust_remote_code=self.trust_remote_code,
                gpu_memory_utilization=self.gpu_memory_fraction,
                enforce_eager=self.enforce_eager,
            )

            logger.info("vLLM embedding model loaded: %s", self.model_name)

        return self._llm

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension by encoding a test sentence."""
        if self._embedding_dim is None:
            test_output = self.llm.embed(["test"])
            self._embedding_dim = len(test_output[0].outputs.embedding)
            logger.info("Embedding dimension: %d", self._embedding_dim)
        return self._embedding_dim

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 256,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences to embeddings using vLLM.

        Args:
            sentences: Single string or list of strings to encode
            batch_size: Batch size (vLLM handles batching internally)
            show_progress_bar: Show progress (vLLM handles this)
            normalize_embeddings: L2 normalize embeddings
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            numpy array of embeddings, shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # vLLM handles batching internally with continuous batching
        outputs = self.llm.embed(sentences)

        # Pre-allocate array for faster extraction (avoids Python list overhead)
        n_outputs = len(outputs)
        if n_outputs == 0:
            return np.array([], dtype=np.float32).reshape(0, self.get_sentence_embedding_dimension())

        embedding_dim = len(outputs[0].outputs.embedding)
        embeddings = np.empty((n_outputs, embedding_dim), dtype=np.float32)

        # Extract embeddings directly into pre-allocated array
        for i, output in enumerate(outputs):
            embeddings[i] = output.outputs.embedding

        # Optionally normalize (in-place for speed)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.divide(embeddings, norms, out=embeddings)

        return embeddings

    def unload(self):
        """Unload the model from GPU memory."""
        if self._llm is not None:
            # vLLM doesn't have explicit unload, but we can delete the reference
            del self._llm
            self._llm = None
            self._embedding_dim = None

            # Clear CUDA cache
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("vLLM embedding model unloaded: %s", self.model_name)


def create_embedder(
    model_name: str,
    backend: str = "sentence_transformers",
    vllm_gpu_memory_fraction: float = 0.5,
    vllm_enforce_eager: bool = True,
):
    """Create an embedder using the specified backend.

    Args:
        model_name: HuggingFace model name
        backend: 'sentence_transformers' or 'vllm'
        vllm_gpu_memory_fraction: GPU memory fraction for vLLM
        vllm_enforce_eager: Use eager mode for vLLM

    Returns:
        Embedder instance (SentenceTransformer or VLLMEmbedder)
    """
    if backend == "vllm":
        logger.info("Using vLLM embedding backend for: %s", model_name)
        return VLLMEmbedder(
            model_name=model_name,
            gpu_memory_fraction=vllm_gpu_memory_fraction,
            enforce_eager=vllm_enforce_eager,
        )
    else:
        from sentence_transformers import SentenceTransformer

        logger.info("Using sentence-transformers backend for: %s", model_name)
        return SentenceTransformer(model_name, trust_remote_code=True)
