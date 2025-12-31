"""Prompt building utilities for RAG systems.

This module provides a centralized way to build prompts for different
experiment types (direct, RAG) with support for few-shot examples.

Usage:
    # Create prompt builder from config
    builder = PromptBuilder.from_config("fewshot", dataset="hotpotqa")
    
    # Build prompts
    direct_prompt = builder.build_direct(query="What is AI?")
    rag_prompt = builder.build_rag(query="What is AI?", context="...")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FewShotExample:
    """A single few-shot example."""

    question: str
    answer: str

    def format(self) -> str:
        """Format as Question/Answer pair."""
        return f"Question: {self.question}\nAnswer: {self.answer}"


@dataclass
class PromptConfig:
    """Configuration for prompt building.

    Attributes:
        system_prompt: System-level instruction
        style: Answer style instruction (e.g., "Give ONLY the answer")
        stop_instruction: Instruction to stop generating
        knowledge_instruction: For RAG, whether to use own knowledge
        examples: Few-shot examples
    """

    system_prompt: str = "You are a helpful assistant."
    style: str = ""
    stop_instruction: str = ""
    knowledge_instruction: str = ""
    examples: List[FewShotExample] = field(default_factory=list)

    @property
    def examples_text(self) -> str:
        """Format all examples as text."""
        if not self.examples:
            return ""
        return "\n\n".join(ex.format() for ex in self.examples) + "\n\n"

    @property
    def instruction_text(self) -> str:
        """Combine style and stop instructions."""
        parts = []
        if self.style:
            parts.append(self.style)
        if self.stop_instruction:
            parts.append(self.stop_instruction)
        return "\n".join(parts)


class PromptBuilder:
    """Builds prompts for direct and RAG experiments.

    This is the single source of truth for prompt construction.
    Supports default, concise, and few-shot prompt styles.

    Example:
        >>> builder = PromptBuilder.from_config("fewshot", dataset="hotpotqa")
        >>> prompt = builder.build_direct("What is the capital of France?")
        >>> print(prompt)
        You are a helpful assistant.

        Give ONLY the final answer...
        IMPORTANT: Give only ONE short answer...

        Question: What government position...
        Answer: Deputy Prime Minister of Israel

        ...

        Question: What is the capital of France?
        Answer:
    """

    # Cache for loaded fewshot examples
    _fewshot_cache: Optional[Dict[str, Any]] = None

    def __init__(self, config: PromptConfig):
        """Initialize with a prompt configuration.

        Args:
            config: PromptConfig with all prompt settings
        """
        self.config = config

    def build_direct(self, query: str) -> str:
        """Build prompt for direct (no RAG) experiments.

        Args:
            query: The question to answer

        Returns:
            Complete prompt string
        """
        parts = [self.config.system_prompt]

        # Add instructions if present
        if self.config.instruction_text:
            parts.append(self.config.instruction_text)

        # Add few-shot examples
        if self.config.examples:
            parts.append(self.config.examples_text.rstrip())

        # Add the actual question
        parts.append(f"Question: {query}\nAnswer:")

        return "\n\n".join(parts)

    def build_rag(self, query: str, context: str) -> str:
        """Build prompt for RAG experiments with retrieved context.

        Args:
            query: The question to answer
            context: Retrieved context (formatted documents)

        Returns:
            Complete prompt string with context
        """
        parts = [self.config.system_prompt]

        # RAG-specific instruction
        rag_instruction = "Use the context to answer."
        if self.config.knowledge_instruction:
            rag_instruction += f" {self.config.knowledge_instruction}"
        if self.config.style:
            rag_instruction += f" {self.config.style}"

        parts.append(rag_instruction)

        # Add stop instruction
        if self.config.stop_instruction:
            parts.append(self.config.stop_instruction)

        # Add few-shot examples (before context)
        if self.config.examples:
            parts.append(self.config.examples_text.rstrip())

        # Add context and question
        parts.append(f"Context:\n{context}")
        parts.append(f"Question: {query}\nAnswer:")

        return "\n\n".join(parts)

    @classmethod
    def _load_fewshot_file(cls) -> Dict[str, Any]:
        """Load fewshot examples from YAML file (cached)."""
        if cls._fewshot_cache is not None:
            return cls._fewshot_cache

        # Try multiple locations
        paths = [
            Path(__file__).parent.parent.parent.parent / "conf" / "prompts" / "fewshot_examples.yaml",
            Path("conf/prompts/fewshot_examples.yaml"),
        ]

        for path in paths:
            if path.exists():
                with open(path) as f:
                    cls._fewshot_cache = yaml.safe_load(f) or {}
                    return cls._fewshot_cache

        cls._fewshot_cache = {}
        return cls._fewshot_cache

    @classmethod
    def from_config(
        cls,
        prompt_type: str = "default",
        dataset: str = "nq",
        system_prompt: str = "You are a helpful assistant.",
    ) -> "PromptBuilder":
        """Create a PromptBuilder from a config type.

        Args:
            prompt_type: One of "default", "concise", "fewshot", "fewshot_3", "fewshot_1"
            dataset: Dataset name for loading appropriate fewshot examples
            system_prompt: Base system prompt

        Returns:
            Configured PromptBuilder

        Example:
            >>> builder = PromptBuilder.from_config("fewshot", dataset="hotpotqa")
        """
        if prompt_type == "default":
            return cls(PromptConfig(
                system_prompt=system_prompt,
                style="Give ONLY the answer, nothing else.",
            ))

        if prompt_type == "concise":
            return cls(PromptConfig(
                system_prompt=system_prompt,
                style="Give ONLY the answer - no explanations.",
            ))

        if prompt_type.startswith("fewshot"):
            # Determine number of examples
            n_examples = {"fewshot": 5, "fewshot_3": 3, "fewshot_1": 1}.get(prompt_type, 5)

            # Load dataset-specific config
            fewshot_data = cls._load_fewshot_file()
            dataset_config = fewshot_data.get(dataset, {})

            # Extract examples
            raw_examples = dataset_config.get("examples", [])[:n_examples]
            examples = [
                FewShotExample(question=ex["question"], answer=ex["answer"])
                for ex in raw_examples
            ]

            return cls(PromptConfig(
                system_prompt=system_prompt,
                style=dataset_config.get("style", "Give a short, direct answer."),
                stop_instruction=dataset_config.get("stop_instruction", ""),
                knowledge_instruction=dataset_config.get("knowledge_instruction", ""),
                examples=examples,
            ))

        # Default fallback
        return cls(PromptConfig(system_prompt=system_prompt))

    # =========================================================================
    # Legacy factory methods (for backwards compatibility)
    # =========================================================================

    @staticmethod
    def create_default() -> "PromptBuilder":
        """Create a prompt builder with default settings."""
        return PromptBuilder.from_config("default")

    @staticmethod
    def create_concise() -> "PromptBuilder":
        """Create a prompt builder for concise answers."""
        return PromptBuilder.from_config("concise")
