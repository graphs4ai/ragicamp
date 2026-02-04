"""Embedder protocol and factory.

Provides a unified interface for embedding backends (vLLM, SentenceTransformers, etc.)
and a factory function to create embedders without backend-specific code in callers.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ragicamp.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol defining the common interface for all embedding backends.

    All embedders must implement:
    - encode(): Convert text to embeddings
    - get_sentence_embedding_dimension(): Return embedding dimensionality
    - unload(): Release resources (GPU memory, etc.)
    """

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 256,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences to embeddings.

        Args:
            sentences: Single string or list of strings to encode
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress
            normalize_embeddings: L2 normalize embeddings
            **kwargs: Backend-specific arguments

        Returns:
            numpy array of embeddings, shape (n_sentences, embedding_dim)
        """
        ...

    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    def unload(self) -> None:
        """Release resources (GPU memory, model weights, etc.)."""
        ...


def create_embedder(
    model_name: str,
    backend: str = "vllm",
    gpu_memory_fraction: float = 0.7,
    enforce_eager: bool = False,
    use_flash_attn: bool = True,
    use_compile: bool = True,
    max_model_len: int | None = None,
    **kwargs,
) -> Embedder:
    """Create an embedder using the specified backend.

    This is the single entry point for creating embedders. Callers don't need
    to know about backend-specific details.

    Args:
        model_name: HuggingFace model name
        backend: 'vllm' or 'sentence_transformers'
        gpu_memory_fraction: GPU memory fraction (vLLM only)
        enforce_eager: Use eager mode instead of CUDA graphs (vLLM only)
        use_flash_attn: Use Flash Attention 2 if available (sentence_transformers only)
        use_compile: Apply torch.compile (sentence_transformers only)
        max_model_len: Maximum sequence length for embeddings (vLLM only).
                      Set higher if chunks exceed model's default (e.g., 512 for Stella).
        **kwargs: Additional backend-specific arguments

    Returns:
        Embedder instance that implements the Embedder protocol
    """
    if backend == "vllm":
        from .vllm_embedder import VLLMEmbedder

        logger.info("Creating vLLM embedder for: %s", model_name)
        return VLLMEmbedder(
            model_name=model_name,
            gpu_memory_fraction=gpu_memory_fraction,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
        )
    else:
        from .st_embedder import SentenceTransformerEmbedder

        logger.info("Creating SentenceTransformer embedder for: %s", model_name)
        return SentenceTransformerEmbedder(
            model_name=model_name,
            use_flash_attn=use_flash_attn,
            use_compile=use_compile,
        )
