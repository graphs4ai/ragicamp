"""Direct LLM agent - Baseline 1: No retrieval, just ask the LLM."""

from typing import Any, Optional

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.core.schemas import AgentType, RAGResponseMeta
from ragicamp.models.base import LanguageModel
from ragicamp.utils.prompts import PromptBuilder


class DirectLLMAgent(RAGAgent):
    """Baseline agent that directly queries an LLM without retrieval.

    This is the simplest baseline: just prompt the LLM with the question
    and return its answer.
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        prompt_builder: Optional[PromptBuilder] = None,
        # Legacy parameters for backwards compatibility
        system_prompt: str = "You are a helpful assistant. Answer questions accurately and concisely.",
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the direct LLM agent.

        Args:
            name: Agent identifier
            model: The language model to use
            prompt_builder: PromptBuilder instance for building prompts.
                          If not provided, creates default from system_prompt.
            system_prompt: (Legacy) System prompt for the LLM
            prompt_template: (Legacy) Custom template - ignored if prompt_builder provided
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model

        # Use provided prompt_builder or create from legacy params
        if prompt_builder is not None:
            self.prompt_builder = prompt_builder
        else:
            # Legacy: create basic builder using style field
            from ragicamp.utils.prompts import PromptConfig

            self.prompt_builder = PromptBuilder(PromptConfig(style=system_prompt))

        # Legacy template support (for backwards compat with study.py)
        self._legacy_template = prompt_template

    def _build_prompt(self, query: str) -> str:
        """Build prompt for a query."""
        # If legacy template provided, use it directly
        if self._legacy_template:
            style = self.prompt_builder.config.style or ""
            return f"{style}\n\n{self._legacy_template.format(question=query)}"

        # Use prompt builder
        return self.prompt_builder.build_direct(query)

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer by directly querying the LLM.

        Args:
            query: The input question
            **kwargs: Additional generation parameters

        Returns:
            RAGResponse with the LLM's answer
        """
        context = RAGContext(query=query)
        prompt = self._build_prompt(query)
        answer = self.model.generate(prompt, **kwargs)

        return RAGResponse(
            answer=answer,
            context=context,
            prompt=prompt,
            metadata=RAGResponseMeta(agent_type=AgentType.DIRECT_LLM),
        )

    def batch_answer(self, queries: list[str], **kwargs: Any) -> list[RAGResponse]:
        """Generate answers for multiple queries using batch processing.

        This is much faster than calling answer() in a loop because it
        processes all queries in a single forward pass through the model.

        Args:
            queries: List of input questions
            **kwargs: Additional generation parameters

        Returns:
            List of RAGResponse objects, one per query
        """
        prompts = [self._build_prompt(q) for q in queries]
        answers = self.model.generate(prompts, **kwargs)

        responses = []
        for query, prompt, answer in zip(queries, prompts, answers):
            context = RAGContext(query=query)
            response = RAGResponse(
                answer=answer,
                context=context,
                prompt=prompt,
                metadata=RAGResponseMeta(
                    agent_type=AgentType.DIRECT_LLM,
                    batch_processing=True,
                ),
            )
            responses.append(response)

        return responses
