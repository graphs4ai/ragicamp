"""HuggingFace model implementation with proper resource management."""

import gc
from typing import Any, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ragicamp.models.base import LanguageModel


class HuggingFaceModel(LanguageModel):
    """Language model implementation using HuggingFace transformers.

    Includes proper resource management with unload() method to free GPU memory.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ):
        """Initialize HuggingFace model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (cuda/cpu)
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization (more aggressive)
            **kwargs: Additional model loading arguments
        """
        super().__init__(model_name, **kwargs)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._use_quantization = load_in_8bit or load_in_4bit
        self._is_loaded = False

        # Configure quantization using BitsAndBytesConfig (new API)
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not defined (required for batch processing)
        # Many models like Llama don't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self._use_quantization else None,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,  # Reduces peak memory during loading
            **kwargs,
        )

        if not self._use_quantization:
            self.model = self.model.to(self.device)

        # Enable gradient checkpointing to save memory (trades compute for memory)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.model.eval()
        self._is_loaded = True

    def unload(self) -> None:
        """Unload model and tokenizer to free GPU memory.

        Call this when done with the model to release GPU resources
        before loading other heavy models (like BERTScore, BLEURT).
        """
        if not self._is_loaded:
            return

        # Delete model
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        # Delete tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self._is_loaded = False
        print(f"🗑️  Model {self.model_name} unloaded from GPU")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._is_loaded

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """Generate text using HuggingFace model."""
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Tokenize - get the actual device from the model
        if hasattr(self.model, "device"):
            target_device = self.model.device
        else:
            # For quantized models with device_map="auto", get first parameter's device
            target_device = next(self.model.parameters()).device

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
            target_device
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or 256,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove input prompt from output
        results = []
        for prompt_text, generated in zip(prompts, generated_texts):
            # Remove the prompt from the generated text
            if generated.startswith(prompt_text):
                generated = generated[len(prompt_text) :].strip()
            results.append(generated)

        return results if is_batch else results[0]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using the model's hidden states."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state
            embeddings = outputs.hidden_states[-1].mean(dim=1)

        return embeddings.cpu().tolist()

    def count_tokens(self, text: str) -> int:
        """Count tokens using the tokenizer."""
        return len(self.tokenizer.encode(text))
