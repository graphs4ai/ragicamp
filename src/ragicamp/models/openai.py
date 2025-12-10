"""OpenAI model implementation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Union

import openai
import tiktoken

from ragicamp.models.base import LanguageModel


class OpenAIModel(LanguageModel):
    """Language model implementation using OpenAI API."""

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        max_workers: int = 10,  # Parallel API calls
        **kwargs: Any,
    ):
        """Initialize OpenAI model.

        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            max_workers: Maximum parallel API calls (default: 10)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.max_workers = max_workers

        if api_key:
            openai.api_key = api_key

        # Initialize tokenizer for counting
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _single_generate(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        **kwargs: Any,
    ) -> str:
        """Generate text for a single prompt."""
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )
        return response.choices[0].message.content

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        parallel: bool = True,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate text using OpenAI API.
        
        Args:
            prompt: Single prompt or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            parallel: Use parallel API calls for batches (default: True)
            **kwargs: Additional API parameters
            
        Returns:
            Generated text or list of texts
        """
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Single prompt - no parallelization needed
        if len(prompts) == 1:
            result = self._single_generate(
                prompts[0], max_tokens, temperature, top_p, stop, **kwargs
            )
            return [result] if is_batch else result

        # Multiple prompts - use parallel execution
        if parallel and len(prompts) > 1:
            results = [None] * len(prompts)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks with their indices
                future_to_idx = {
                    executor.submit(
                        self._single_generate,
                        p, max_tokens, temperature, top_p, stop, **kwargs
                    ): i
                    for i, p in enumerate(prompts)
                }
                
                # Collect results maintaining order
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        # On error, store error message
                        results[idx] = f"[ERROR: {str(e)}]"
            
            return results
        else:
            # Sequential fallback
            results = []
            for p in prompts:
                results.append(
                    self._single_generate(p, max_tokens, temperature, top_p, stop, **kwargs)
                )
            return results

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI embeddings API."""
        response = openai.embeddings.create(model="text-embedding-ada-002", input=texts)
        return [item.embedding for item in response.data]

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))
