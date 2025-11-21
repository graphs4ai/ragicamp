"""Prompt building utilities for RAG systems."""

from typing import Any, Dict, Optional


class PromptBuilder:
    """Utility class for building prompts for RAG systems.

    Centralizes prompt construction logic that was previously duplicated
    across agent implementations.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        context_template: str = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        separator: str = "\n\n",
    ):
        """Initialize prompt builder.

        Args:
            system_prompt: System-level instruction for the model
            context_template: Template for question+context prompts
            separator: Separator between system prompt and context template
        """
        self.system_prompt = system_prompt
        self.context_template = context_template
        self.separator = separator

    def build_prompt(self, query: str, context: Optional[str] = None, **kwargs: Any) -> str:
        """Build a complete prompt for the LLM.

        Args:
            query: The user's question
            context: Retrieved context (optional)
            **kwargs: Additional variables for template formatting

        Returns:
            Complete formatted prompt

        Examples:
            >>> builder = PromptBuilder()
            >>> builder.build_prompt("What is AI?", "AI stands for Artificial Intelligence")
            'You are a helpful assistant.\\n\\nContext:\\nAI stands for Artificial Intelligence\\n\\nQuestion: What is AI?\\n\\nAnswer:'
        """
        # Prepare template variables
        template_vars = {"query": query, **kwargs}

        if context is not None:
            template_vars["context"] = context

        # Build the prompt
        parts = [self.system_prompt]

        # Format the context template
        formatted_template = self.context_template.format(**template_vars)
        parts.append(formatted_template)

        return self.separator.join(parts)

    def build_direct_prompt(self, query: str, **kwargs: Any) -> str:
        """Build a prompt without context (direct LLM).

        Args:
            query: The user's question
            **kwargs: Additional variables

        Returns:
            Prompt for direct LLM query

        Example:
            >>> builder = PromptBuilder()
            >>> builder.build_direct_prompt("What is AI?")
            'You are a helpful assistant.\\n\\nQuestion: What is AI?\\n\\nAnswer:'
        """
        template = "Question: {query}\n\nAnswer:"
        formatted = template.format(query=query, **kwargs)
        return f"{self.system_prompt}{self.separator}{formatted}"

    def build_rag_prompt(
        self, query: str, context: str, instruction: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Build a RAG prompt with context.

        Args:
            query: The user's question
            context: Retrieved context
            instruction: Optional additional instruction
            **kwargs: Additional variables

        Returns:
            Complete RAG prompt

        Example:
            >>> builder = PromptBuilder()
            >>> builder.build_rag_prompt(
            ...     "What is AI?",
            ...     "AI stands for Artificial Intelligence",
            ...     instruction="Answer concisely"
            ... )
        """
        parts = [self.system_prompt]

        if instruction:
            parts.append(instruction)

        formatted = self.context_template.format(query=query, context=context, **kwargs)
        parts.append(formatted)

        return self.separator.join(parts)

    @staticmethod
    def create_default() -> "PromptBuilder":
        """Create a prompt builder with default settings.

        Returns:
            PromptBuilder with standard RAG settings
        """
        return PromptBuilder(
            system_prompt="You are a helpful assistant. Use the provided context to answer questions accurately.",
            context_template="Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        )

    @staticmethod
    def create_concise() -> "PromptBuilder":
        """Create a prompt builder optimized for concise answers.

        Returns:
            PromptBuilder configured for brief responses
        """
        return PromptBuilder(
            system_prompt="You are a helpful assistant. Answer questions concisely and accurately based on the provided context.",
            context_template="Context:\n{context}\n\nQuestion: {query}\n\nProvide a brief, accurate answer:",
        )

    @staticmethod
    def create_detailed() -> "PromptBuilder":
        """Create a prompt builder optimized for detailed answers.

        Returns:
            PromptBuilder configured for comprehensive responses
        """
        return PromptBuilder(
            system_prompt="You are a knowledgeable assistant. Provide detailed, well-explained answers based on the given context.",
            context_template="Context:\n{context}\n\nQuestion: {query}\n\nProvide a detailed answer with explanations:",
        )

    @staticmethod
    def create_extractive() -> "PromptBuilder":
        """Create a prompt builder for extractive QA.

        Returns:
            PromptBuilder configured for extracting answers from context
        """
        return PromptBuilder(
            system_prompt="You are an assistant that extracts exact answers from the given context.",
            context_template="Context:\n{context}\n\nQuestion: {query}\n\nExtract the answer from the context above. If the answer is not in the context, say 'Not found'.\n\nAnswer:",
        )
