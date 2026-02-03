"""Language model interfaces."""

from ragicamp.models.base import LanguageModel
from ragicamp.models.embedder import Embedder, create_embedder
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.models.openai import OpenAIModel

# vLLM is optional - only import if available
try:
    from ragicamp.models.vllm import VLLMModel

    _VLLM_AVAILABLE = True
except ImportError:
    VLLMModel = None  # type: ignore
    _VLLM_AVAILABLE = False

__all__ = [
    "LanguageModel",
    "HuggingFaceModel",
    "OpenAIModel",
    "VLLMModel",
    "Embedder",
    "create_embedder",
]
