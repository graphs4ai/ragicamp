"""Base class for language models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class LanguageModel(ABC):
    """Base class for all language models.

    This abstraction allows us to work with different LLM providers
    (HuggingFace, OpenAI, Anthropic, etc.) through a unified interface.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize the language model.

        Args:
            model_name: Identifier for the model
            **kwargs: Model-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate text from a prompt.

        Args:
            prompt: Single prompt string or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated text (single string or list matching input)
        """
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        # Default simple approximation - override in subclasses
        return len(text.split())

    def batch_generate(self, prompts: List[str], batch_size: int = 8, **kwargs: Any) -> List[str]:
        """Generate text for multiple prompts in batches.

        Args:
            prompts: List of prompts
            batch_size: Batch size for generation
            **kwargs: Generation parameters

        Returns:
            List of generated texts
        """
        # Default implementation - can be overridden for efficiency
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            results.extend(self.generate(batch, **kwargs))
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
