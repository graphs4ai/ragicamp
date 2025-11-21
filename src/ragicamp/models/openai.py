"""OpenAI model implementation."""

from typing import Any, List, Optional, Union

import openai
import tiktoken

from ragicamp.models.base import LanguageModel


class OpenAIModel(LanguageModel):
    """Language model implementation using OpenAI API."""

    def __init__(
        self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs: Any
    ):
        """Initialize OpenAI model.

        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)

        if api_key:
            openai.api_key = api_key

        # Initialize tokenizer for counting
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate text using OpenAI API."""
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        results = []
        for p in prompts:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": p}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )
            results.append(response.choices[0].message.content)

        return results if is_batch else results[0]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI embeddings API."""
        response = openai.embeddings.create(model="text-embedding-ada-002", input=texts)
        return [item.embedding for item in response.data]

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))
