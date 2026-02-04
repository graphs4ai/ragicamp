"""vLLM-based embedding model for high-throughput embedding generation.

Uses vLLM's continuous batching for faster embedding than sentence-transformers.
See: https://docs.vllm.ai/en/latest/getting_started/examples/embedding.html

Supported models (vLLM 0.15+):

Top MTEB performers:
- Qwen/Qwen3-Embedding-8B         # MTEB #1 (70.58), 8B params, 32k context, 100+ languages
- Alibaba-NLP/gte-Qwen2-7B-instruct  # MTEB 70.24, 7B params
- Alibaba-NLP/gte-Qwen2-1.5B-instruct  # Fast alternative, 1.5B params

BGE family (BERT-based, fast):
- BAAI/bge-large-en-v1.5          # Proven baseline, 335M params
- BAAI/bge-m3                     # Multilingual, supports sparse/ColBERT (vLLM 0.15+)
- BAAI/bge-en-icl                 # Instruction-following

Decoder-based (large but powerful):
- intfloat/e5-mistral-7b-instruct # 7B params, top MTEB retrieval
- Salesforce/SFR-Embedding-Mistral  # MTEB 68.17

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
        gpu_memory_fraction: float = 0.95,  # Use almost all VRAM
        enforce_eager: bool = False,  # CUDA graphs improve throughput
        trust_remote_code: bool = True,
        max_num_seqs: int = 8192,  # Massive batch concurrency for large GPUs
        max_num_batched_tokens: int = 131072,  # 128k tokens per batch (B200 can handle it)
        max_model_len: int = 2048,  # Shorter context = more sequences in parallel
    ):
        """Initialize vLLM embedder optimized for maximum throughput.

        Args:
            model_name: HuggingFace model name (must be vLLM-compatible)
            gpu_memory_fraction: Fraction of GPU memory to use (0.95 = max throughput)
            enforce_eager: Use CUDA graphs for throughput (False = enable graphs)
            trust_remote_code: Trust remote code in model
            max_num_seqs: Max concurrent sequences (8192 for large GPUs like B200)
            max_num_batched_tokens: Max tokens per batch (higher = better GPU utilization)
            max_model_len: Max sequence length (2048 is enough for 512-token chunks)
        """
        self.model_name = model_name
        self.gpu_memory_fraction = gpu_memory_fraction
        self.enforce_eager = enforce_eager
        self.trust_remote_code = trust_remote_code
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len

        self._llm: Optional[vllm.LLM] = None
        self._embedding_dim: Optional[int] = None

    @property
    def llm(self):
        """Lazy load the vLLM model."""
        if self._llm is None:
            from vllm import LLM

            # Check GPU availability before loading
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
                logger.info("GPU detected: %s (%.1f GB total, %.2f GB allocated)", 
                           gpu_name, gpu_mem, gpu_mem_allocated)
            else:
                logger.warning("No GPU detected! vLLM will be slow on CPU.")

            logger.info(
                "Loading vLLM embedding model: %s (gpu_mem_util=%.1f%%)",
                self.model_name,
                self.gpu_memory_fraction * 100,
            )

            # vLLM 0.15+ API: use runner="pooling" instead of task="embed"
            # See: https://docs.vllm.ai/en/stable/models/pooling_models/
            #
            # THROUGHPUT OPTIMIZATIONS for large GPUs (B200, H100, etc.):
            # - max_num_seqs=8192: Massive batch concurrency
            # - max_num_batched_tokens=131072: Process 128k tokens per forward pass
            # - max_model_len=2048: Shorter context = more sequences fit in memory
            # - enforce_eager=False: CUDA graphs reduce kernel launch overhead
            # - gpu_memory_utilization=0.95: Use all available VRAM
            
            logger.info(
                "Throughput config: max_num_seqs=%d, max_batched_tokens=%d, max_model_len=%d",
                self.max_num_seqs,
                self.max_num_batched_tokens,
                self.max_model_len,
            )
            
            self._llm = LLM(
                model=self.model_name,
                runner="pooling",  # vLLM 0.15+ pooling mode
                trust_remote_code=self.trust_remote_code,
                gpu_memory_utilization=self.gpu_memory_fraction,
                enforce_eager=self.enforce_eager,
                dtype="bfloat16",  # bfloat16 for fastest inference on modern GPUs
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_model_len,
            )

            # Log actual GPU memory after loading
            if torch.cuda.is_available():
                gpu_mem_after = torch.cuda.memory_allocated(0) / 1e9
                logger.info("vLLM embedding model loaded: %s (GPU memory: %.2f GB)", 
                           self.model_name, gpu_mem_after)

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
            batch_size: Ignored - vLLM handles batching internally
            show_progress_bar: Show progress (vLLM handles this)
            normalize_embeddings: L2 normalize embeddings
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            numpy array of embeddings, shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # Log first encode to verify GPU is being used
        if not hasattr(self, "_first_encode_logged"):
            import torch
            if torch.cuda.is_available():
                gpu_mem_before = torch.cuda.memory_allocated(0) / 1e9
                logger.debug("GPU memory before encode: %.2f GB", gpu_mem_before)
            self._first_encode_logged = True

        # vLLM handles batching internally with continuous batching
        outputs = self.llm.embed(sentences)

        # Pre-allocate array for faster extraction (avoids Python list overhead)
        n_outputs = len(outputs)
        if n_outputs == 0:
            return np.array([], dtype=np.float32).reshape(
                0, self.get_sentence_embedding_dimension()
            )

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
