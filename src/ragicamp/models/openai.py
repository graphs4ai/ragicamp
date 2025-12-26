"""OpenAI model implementation with async support.

This module provides both sync and async interfaces to the OpenAI API.
Use async methods (agenerate, agenerate_single) for efficient parallel
API calls, especially in LLM-as-a-judge metrics.

Example:
    >>> model = OpenAIModel("gpt-4o-mini")
    >>> 
    >>> # Sync usage (backward compatible)
    >>> response = model.generate("Hello, world!")
    >>> 
    >>> # Async usage (for parallel calls)
    >>> responses = await model.agenerate(["prompt1", "prompt2", "prompt3"])
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Union

import openai
import tiktoken
from openai import AsyncOpenAI

from ragicamp.models.base import LanguageModel


class OpenAIModel(LanguageModel):
    """Language model implementation using OpenAI API.
    
    Supports both synchronous and asynchronous generation.
    Use async methods for efficient parallel API calls.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_workers: int = 10,  # Parallel API calls
        max_concurrent_async: int = 20,  # Max concurrent async calls
        **kwargs: Any,
    ):
        """Initialize OpenAI model.

        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            max_workers: Maximum parallel thread-based API calls (default: 10)
            max_concurrent_async: Maximum concurrent async API calls (default: 20)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.max_workers = max_workers
        self.max_concurrent_async = max_concurrent_async
        self._api_key = api_key

        if api_key:
            openai.api_key = api_key

        # Initialize tokenizer for counting
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Lazy-initialized async client
        self._async_client: Optional[AsyncOpenAI] = None
    
    @property
    def async_client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        return self._async_client

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
    
    # =========================================================================
    # Async Methods
    # =========================================================================
    
    async def agenerate_single(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text for a single prompt asynchronously.
        
        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            system_message: Optional system message
            **kwargs: Additional API parameters
            
        Returns:
            Generated text
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )
        return response.choices[0].message.content
    
    async def agenerate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text for multiple prompts asynchronously with rate limiting.
        
        Uses asyncio.Semaphore to limit concurrent API calls.
        
        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            system_message: Optional system message for all prompts
            **kwargs: Additional API parameters
            
        Returns:
            List of generated texts (in same order as prompts)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_async)
        
        async def rate_limited_generate(idx: int, prompt: str) -> tuple[int, str]:
            """Generate with rate limiting, returning index for ordering."""
            async with semaphore:
                try:
                    result = await self.agenerate_single(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        system_message=system_message,
                        **kwargs,
                    )
                    return (idx, result)
                except Exception as e:
                    return (idx, f"[ERROR: {str(e)}]")
        
        # Create tasks for all prompts
        tasks = [rate_limited_generate(i, p) for i, p in enumerate(prompts)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Sort by index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
    
    async def aget_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI embeddings API asynchronously.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = await self.async_client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts,
        )
        return [item.embedding for item in response.data]
