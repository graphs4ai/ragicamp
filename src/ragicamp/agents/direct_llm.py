"""Direct LLM Agent - No retrieval, just generation.

This is the simplest baseline: batch generate all answers.
Only the generator model is used (full GPU).
"""

from pathlib import Path
from typing import Callable

from tqdm import tqdm

from ragicamp.agents.base import Agent, AgentResult, Query, Step, StepTimer
from ragicamp.core.logging import get_logger
from ragicamp.models.base import LanguageModel
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)


class DirectLLMAgent(Agent):
    """Direct LLM agent - no retrieval, just generation.
    
    Execution strategy:
    - Single phase: batch generate all answers
    - Generator uses full GPU
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        prompt_builder: PromptBuilder | None = None,
        **config,
    ):
        super().__init__(name, **config)
        self.model = model
        self.prompt_builder = prompt_builder or PromptBuilder(PromptConfig())

    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process all queries with batch generation."""
        # Load checkpoint if resuming
        results, completed_idx = [], set()
        if checkpoint_path:
            results, completed_idx = self._load_checkpoint(checkpoint_path)
        
        pending = [q for q in queries if q.idx not in completed_idx]
        
        if not pending:
            logger.info("All queries already completed")
            return results
        
        logger.info("Generating %d answers (batch mode)", len(pending))
        
        # Build all prompts
        prompts = [self.prompt_builder.build_direct(q.text) for q in pending]
        
        # Batch generate
        with StepTimer("batch_generate", model=self._get_model_name()) as step:
            step.input = {"num_queries": len(pending)}
            answers = self.model.generate(prompts)
            step.output = {"num_answers": len(answers)}
        
        # Build results
        iterator = zip(pending, prompts, answers)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Building results")
        
        for query, prompt, answer in iterator:
            result = AgentResult(
                query=query,
                answer=answer,
                steps=[Step(
                    type="generate",
                    input=query.text,
                    output=answer,
                    model=self._get_model_name(),
                )],
                prompt=prompt,
            )
            results.append(result)
            
            if on_result:
                on_result(result)
        
        if checkpoint_path:
            self._save_checkpoint(results, checkpoint_path)
        
        return results

    def _get_model_name(self) -> str | None:
        return getattr(self.model, 'model_name', None)
