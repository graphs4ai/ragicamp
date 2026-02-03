"""vLLM-based embedding model for high-throughput embedding generation.

Uses vLLM's continuous batching for faster embedding than sentence-transformers.
See: https://docs.vllm.ai/en/latest/getting_started/examples/embedding.html

Two modes available:
- VLLMEmbedder: In-process vLLM (simpler, but blocks during encode)
- VLLMServerEmbedder: Subprocess server with async HTTP (enables true overlap)

Supported models (examples):
- intfloat/e5-mistral-7b-instruct
- Alibaba-NLP/gte-Qwen2-7B-instruct
- BAAI/bge-en-icl (instruction-following)
- Salesforce/SFR-Embedding-Mistral

Note: Not all sentence-transformer models are supported by vLLM.
Check vLLM model compatibility before using.
"""

import asyncio
import os
import signal
import subprocess
import time
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


class VLLMServerEmbedder:
    """vLLM server-based embedder with async HTTP for true GPU/CPU overlap.

    Starts vLLM as a subprocess server and uses async HTTP requests.
    This allows overlapping CPU post-processing with GPU embedding.
    """

    def __init__(
        self,
        model_name: str,
        gpu_memory_fraction: float = 0.9,
        host: str = "127.0.0.1",
        port: int = 8000,
        max_concurrent: int = 2,
    ):
        """Initialize vLLM server embedder.

        Args:
            model_name: HuggingFace model name
            gpu_memory_fraction: Fraction of GPU memory to use
            host: Server host
            port: Server port
            max_concurrent: Max concurrent requests to server
        """
        self.model_name = model_name
        self.gpu_memory_fraction = gpu_memory_fraction
        self.host = host
        self.port = port
        self.max_concurrent = max_concurrent
        self.base_url = f"http://{host}:{port}"

        self._process: Optional[subprocess.Popen] = None
        self._embedding_dim: Optional[int] = None

    def _start_server(self) -> None:
        """Start vLLM server as subprocess."""
        if self._process is not None:
            return

        logger.info("Starting vLLM server for: %s", self.model_name)

        cmd = [
            "vllm", "serve", self.model_name,
            "--task", "embed",
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_fraction),
            "--trust-remote-code",
        ]

        # Start server process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for server to be ready
        self._wait_for_server(timeout=300)
        logger.info("vLLM server ready at %s", self.base_url)

    def _wait_for_server(self, timeout: float = 300) -> None:
        """Wait for server to be ready."""
        import urllib.request
        import urllib.error

        start = time.time()
        health_url = f"{self.base_url}/health"

        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                pass

            # Check if process died
            if self._process and self._process.poll() is not None:
                raise RuntimeError(f"vLLM server died with code {self._process.returncode}")

            time.sleep(2)
            print(".", end="", flush=True)

        raise TimeoutError(f"vLLM server not ready after {timeout}s")

    async def _encode_async(self, texts: list[str]) -> np.ndarray:
        """Encode texts using async HTTP."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model_name, "input": texts},
                timeout=aiohttp.ClientTimeout(total=3600),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Embedding request failed: {error}")

                data = await resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return np.array(embeddings, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension by encoding a test sentence."""
        if self._embedding_dim is None:
            self._start_server()
            test_emb = asyncio.run(self._encode_async(["test"]))
            self._embedding_dim = test_emb.shape[1]
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
        """Encode sentences to embeddings.

        Same interface as VLLMEmbedder.encode() for compatibility.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        self._start_server()

        # Use asyncio to run the async encode
        embeddings = asyncio.run(self._encode_async(sentences))

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.divide(embeddings, norms, out=embeddings)

        return embeddings

    def unload(self):
        """Stop the server and clean up."""
        if self._process is not None:
            logger.info("Stopping vLLM server...")
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            self._embedding_dim = None
            logger.info("vLLM server stopped")

    def __del__(self):
        self.unload()


def create_embedder(
    model_name: str,
    backend: str = "sentence_transformers",
    vllm_gpu_memory_fraction: float = 0.5,
    vllm_enforce_eager: bool = True,
):
    """Create an embedder using the specified backend.

    Args:
        model_name: HuggingFace model name
        backend: 'sentence_transformers', 'vllm', or 'vllm_server'
        vllm_gpu_memory_fraction: GPU memory fraction for vLLM
        vllm_enforce_eager: Use eager mode for vLLM (not used for vllm_server)

    Returns:
        Embedder instance (SentenceTransformer, VLLMEmbedder, or VLLMServerEmbedder)
    """
    if backend == "vllm":
        logger.info("Using vLLM embedding backend for: %s", model_name)
        return VLLMEmbedder(
            model_name=model_name,
            gpu_memory_fraction=vllm_gpu_memory_fraction,
            enforce_eager=vllm_enforce_eager,
        )
    elif backend == "vllm_server":
        logger.info("Using vLLM server embedding backend for: %s", model_name)
        return VLLMServerEmbedder(
            model_name=model_name,
            gpu_memory_fraction=vllm_gpu_memory_fraction,
        )
    else:
        from sentence_transformers import SentenceTransformer

        logger.info("Using sentence-transformers backend for: %s", model_name)
        return SentenceTransformer(model_name, trust_remote_code=True)
