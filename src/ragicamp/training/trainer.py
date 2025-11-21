"""Trainer for adaptive RAG agents."""

from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ragicamp.agents.base import RAGAgent
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric


class Trainer:
    """Trainer for adaptive RAG agents (bandit/MDP-based).

    Handles the training loop: generate answers, compute rewards,
    update policies.
    """

    def __init__(
        self,
        agent: RAGAgent,
        dataset: QADataset,
        metrics: List[Metric],
        reward_metric: str = "f1",
        **kwargs: Any,
    ):
        """Initialize trainer.

        Args:
            agent: The RAG agent to train
            dataset: Training dataset
            metrics: List of metrics for evaluation
            reward_metric: Which metric to use as reward signal
            **kwargs: Additional configuration
        """
        self.agent = agent
        self.dataset = dataset
        self.metrics = {m.name: m for m in metrics}
        self.reward_metric = reward_metric
        self.config = kwargs

        # Training history
        self.history: List[Dict[str, Any]] = []

    def train(
        self, num_epochs: int = 1, batch_size: int = 1, eval_interval: int = 100, **kwargs: Any
    ) -> Dict[str, Any]:
        """Train the agent.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size (for future batching support)
            eval_interval: Evaluate every N examples
            **kwargs: Additional training parameters

        Returns:
            Training statistics
        """
        total_reward = 0.0
        num_examples = 0

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            # Iterate over dataset
            for i, example in enumerate(tqdm(self.dataset, desc="Training")):
                # Generate answer
                response = self.agent.answer(example.question)

                # Compute reward
                reward = self._compute_reward(
                    prediction=response.answer, references=example.answers
                )

                # Update agent (if applicable)
                self._update_agent(response, reward)

                # Track statistics
                total_reward += reward
                num_examples += 1

                # Periodic evaluation
                if (i + 1) % eval_interval == 0:
                    avg_reward = total_reward / num_examples
                    print(f"\nExamples: {num_examples}, Avg Reward: {avg_reward:.4f}")
                    total_reward = 0.0
                    num_examples = 0

        return {"total_examples": len(self.dataset) * num_epochs, "history": self.history}

    def _compute_reward(self, prediction: str, references: List[str]) -> float:
        """Compute reward for a prediction.

        Args:
            prediction: Predicted answer
            references: Reference answers

        Returns:
            Reward value (0-1)
        """
        if self.reward_metric not in self.metrics:
            raise ValueError(f"Reward metric '{self.reward_metric}' not found")

        metric = self.metrics[self.reward_metric]
        result = metric.compute_single(prediction, references)

        # All metrics now return Dict[str, float]
        # Try to find the reward metric score, fallback to first value
        if self.reward_metric in result:
            return result[self.reward_metric]

        # Try common keys
        for key in ["f1", "exact_match", "bertscore_f1", "llm_judge_score"]:
            if key in result:
                return result[key]

        # Fallback to first value
        return list(result.values())[0]

    def _update_agent(self, response: Any, reward: float) -> None:
        """Update agent based on response and reward.

        Args:
            response: Agent response
            reward: Computed reward
        """
        # Check if agent has update_policy method (bandit/MDP agents)
        if hasattr(self.agent, "update_policy"):
            # For bandit agents
            if hasattr(response, "context") and hasattr(response.context, "metadata"):
                params = response.context.metadata.get("selected_params")
                if params:
                    self.agent.update_policy(
                        query=response.context.query, params=params, reward=reward
                    )

            # For MDP agents
            if hasattr(response, "context") and hasattr(response.context, "intermediate_steps"):
                trajectory = response.context.intermediate_steps
                if trajectory:
                    self.agent.update_policy(trajectory=trajectory, reward=reward)
