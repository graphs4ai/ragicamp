"""Prompt building utilities for RAG systems.

This module provides a centralized way to build prompts for different
experiment types (direct, RAG) with support for few-shot examples.

Usage:
    builder = PromptBuilder.from_config("fewshot", dataset="hotpotqa")
    prompt = builder.build_direct(query="What is AI?")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class FewShotExample:
    """A single few-shot example."""

    question: str
    answer: str


@dataclass
class PromptConfig:
    """Configuration for prompt building."""

    style: str = ""
    stop_instruction: str = ""
    knowledge_instruction: str = ""
    examples: list[FewShotExample] = field(default_factory=list)


class PromptBuilder:
    """Builds clean, well-structured prompts for QA experiments.

    Design principles:
    1. Clear task instruction at the top
    2. Examples in a labeled section (if fewshot)
    3. Clear transition to the actual question
    4. Consistent formatting throughout
    """

    _fewshot_cache: Optional[dict[str, Any]] = None

    def __init__(self, config: PromptConfig):
        self.config = config

    def build_direct(self, query: str) -> str:
        """Build prompt for direct (no retrieval) QA.

        Structure:
            [Task instruction - use your knowledge]

            [Examples section - if fewshot]

            Question: {query}
            Answer:
        """
        parts = []

        # Task instruction - emphasize using own knowledge (no context!)
        task = "Answer the question using your knowledge."
        if self.config.style:
            task += f" {self.config.style}"
        if self.config.stop_instruction:
            task += f" {self.config.stop_instruction}"
        task += " If you don't know, answer 'Unknown'."
        parts.append(task)

        # Examples section (if any)
        if self.config.examples:
            parts.append(self._format_examples())

        # The actual question
        parts.append(f"Question: {query}\nAnswer:")

        return "\n\n".join(parts)

    def build_rag(self, query: str, context: str) -> str:
        """Build prompt for RAG (with retrieved context).

        Structure:
            [Task instruction - use context + own knowledge]

            [Examples section - if fewshot]

            Context:
            {retrieved documents}

            Question: {query}
            Answer:
        """
        parts = []

        # Task instruction - encourage using both context AND own knowledge
        task = "Answer the question using the context below and your own knowledge."
        if self.config.knowledge_instruction:
            task += f" {self.config.knowledge_instruction}"
        if self.config.style:
            task += f" {self.config.style}"
        if self.config.stop_instruction:
            task += f" {self.config.stop_instruction}"
        task += " If you don't know, answer 'Unknown'."
        parts.append(task)

        # Examples section (if any)
        if self.config.examples:
            parts.append(self._format_examples())

        # Context and question
        parts.append(f"Context:\n{context}")
        parts.append(f"Question: {query}\nAnswer:")

        return "\n\n".join(parts)

    def _format_examples(self) -> str:
        """Format examples in a clear, labeled section."""
        lines = ["Examples:"]
        for ex in self.config.examples:
            lines.append(f"Q: {ex.question}")
            lines.append(f"A: {ex.answer}")
            lines.append("")  # blank line between examples
        return "\n".join(lines).rstrip()

    @classmethod
    def _load_fewshot_file(cls) -> dict[str, Any]:
        """Load fewshot examples from YAML file (cached)."""
        if cls._fewshot_cache is not None:
            return cls._fewshot_cache

        paths = [
            Path(__file__).parent.parent.parent.parent
            / "conf"
            / "prompts"
            / "fewshot_examples.yaml",
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
    def from_config(cls, prompt_type: str = "default", dataset: str = "nq") -> "PromptBuilder":
        """Create a PromptBuilder from a config type.

        Args:
            prompt_type: "default", "concise", "fewshot", "fewshot_3", "fewshot_1"
            dataset: Dataset name for loading appropriate fewshot examples

        Returns:
            Configured PromptBuilder
        """
        if prompt_type == "default":
            return cls(
                PromptConfig(
                    style="Give only the answer, no explanations.",
                )
            )

        if prompt_type == "concise":
            return cls(
                PromptConfig(
                    style="Reply with just the answer.",
                )
            )

        if prompt_type.startswith("fewshot"):
            n_examples = {"fewshot": 5, "fewshot_3": 3, "fewshot_1": 1}.get(prompt_type, 5)

            fewshot_data = cls._load_fewshot_file()
            dataset_config = fewshot_data.get(dataset, {})

            raw_examples = dataset_config.get("examples", [])[:n_examples]
            examples = [
                FewShotExample(question=ex["question"], answer=ex["answer"]) for ex in raw_examples
            ]

            return cls(
                PromptConfig(
                    style=dataset_config.get("style", "Give a short, direct answer."),
                    stop_instruction=dataset_config.get("stop_instruction", ""),
                    knowledge_instruction=dataset_config.get("knowledge_instruction", ""),
                    examples=examples,
                )
            )

        return cls(PromptConfig())

    # Legacy compatibility
    @staticmethod
    def create_default() -> "PromptBuilder":
        return PromptBuilder.from_config("default")

    @staticmethod
    def create_concise() -> "PromptBuilder":
        return PromptBuilder.from_config("concise")
