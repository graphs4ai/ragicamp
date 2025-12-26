"""Trainer for adaptive RAG agents.

Supports both online training and trajectory collection for offline training.

Example (online training):
    >>> trainer = Trainer(agent, dataset, metrics)
    >>> trainer.train(num_epochs=3)

Example (trajectory collection for offline training):
    >>> trainer = Trainer(agent, dataset, metrics, trajectory_store="trajectories/my_exp")
    >>> trainer.collect_trajectories()  # Only collect, no policy updates
    >>>
    >>> # Later, train offline
    >>> trainer.train_offline(num_epochs=10)
"""

from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ragicamp.agents.base import RAGAgent
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric


class Trainer:
    """Trainer for adaptive RAG agents (bandit/MDP-based).

    Handles the training loop: generate answers, compute rewards,
    update policies. Also supports trajectory collection for offline training.
    """

    def __init__(
        self,
        agent: RAGAgent,
        dataset: QADataset,
        metrics: List[Metric],
        reward_metric: str = "f1",
        trajectory_store_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize trainer.

        Args:
            agent: The RAG agent to train
            dataset: Training dataset
            metrics: List of metrics for evaluation
            reward_metric: Which metric to use as reward signal
            trajectory_store_path: Path for trajectory storage (enables collection)
            **kwargs: Additional configuration
        """
        self.agent = agent
        self.dataset = dataset
        self.metrics = {m.name: m for m in metrics}
        self.reward_metric = reward_metric
        self.config = kwargs

        # Training history
        self.history: List[Dict[str, Any]] = []

        # Trajectory store for offline training
        self._trajectory_store = None
        if trajectory_store_path:
            from ragicamp.training.trajectory_store import TrajectoryStore

            self._trajectory_store = TrajectoryStore(trajectory_store_path)

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

    def collect_trajectories(
        self,
        num_examples: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Collect trajectories without updating the policy.

        Use this to gather experience for offline training.

        Args:
            num_examples: Number of examples to collect (None = all)
            **kwargs: Additional parameters

        Returns:
            Collection statistics
        """
        from ragicamp.training.trajectory_store import Trajectory

        if self._trajectory_store is None:
            raise ValueError(
                "No trajectory store configured. " "Pass trajectory_store_path to Trainer.__init__"
            )

        examples = list(self.dataset)
        if num_examples:
            examples = examples[:num_examples]

        print(f"\n📚 Collecting trajectories for {len(examples)} examples...")

        collected = 0
        total_reward = 0.0

        for example in tqdm(examples, desc="Collecting"):
            # Generate answer
            response = self.agent.answer(example.question)

            # Compute reward
            reward = self._compute_reward(prediction=response.answer, references=example.answers)

            # Extract trajectory steps
            steps = []
            if hasattr(response, "context"):
                if hasattr(response.context, "intermediate_steps"):
                    steps = response.context.intermediate_steps
                elif hasattr(response.context, "metadata"):
                    # For bandit agents, create a single step
                    steps = [
                        {
                            "state": {"query": response.context.query},
                            "action": response.context.metadata.get("selected_params", {}),
                            "reward": reward,
                        }
                    ]

            # Create and save trajectory
            trajectory = Trajectory(
                question_id=example.id,
                question=example.question,
                steps=steps,
                final_answer=response.answer,
                final_reward=reward,
                metadata={
                    "agent_name": self.agent.name,
                    "expected_answers": example.answers,
                },
            )
            self._trajectory_store.save(trajectory)

            collected += 1
            total_reward += reward

        stats = {
            "collected": collected,
            "mean_reward": total_reward / collected if collected > 0 else 0.0,
            "store_path": str(self._trajectory_store.path),
        }

        print(f"\n✓ Collected {collected} trajectories")
        print(f"  Mean reward: {stats['mean_reward']:.4f}")
        print(f"  Saved to: {stats['store_path']}")

        return stats

    def train_offline(
        self,
        num_epochs: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train from stored trajectories (offline RL).

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            shuffle: Shuffle trajectories each epoch
            **kwargs: Additional training parameters

        Returns:
            Training statistics
        """
        if self._trajectory_store is None:
            raise ValueError(
                "No trajectory store configured. " "Pass trajectory_store_path to Trainer.__init__"
            )

        if len(self._trajectory_store) == 0:
            raise ValueError("No trajectories in store. " "Run collect_trajectories() first.")

        print(f"\n🎓 Training offline from {len(self._trajectory_store)} trajectories...")
        print(f"   Epochs: {num_epochs}, Batch size: {batch_size}")

        total_updates = 0
        epoch_rewards = []

        for epoch in range(num_epochs):
            epoch_reward = 0.0
            epoch_count = 0

            for batch in self._trajectory_store.iter_batches(batch_size, shuffle=shuffle):
                for trajectory in batch:
                    # Update policy based on trajectory
                    if hasattr(self.agent, "update_policy"):
                        if trajectory.steps:
                            self.agent.update_policy(
                                trajectory=trajectory.steps,
                                reward=trajectory.final_reward,
                            )

                    epoch_reward += trajectory.final_reward
                    epoch_count += 1
                    total_updates += 1

            avg_reward = epoch_reward / epoch_count if epoch_count > 0 else 0.0
            epoch_rewards.append(avg_reward)
            print(f"  Epoch {epoch + 1}/{num_epochs}: avg_reward={avg_reward:.4f}")

        stats = {
            "num_epochs": num_epochs,
            "total_updates": total_updates,
            "epoch_rewards": epoch_rewards,
            "final_avg_reward": epoch_rewards[-1] if epoch_rewards else 0.0,
        }

        print(f"\n✓ Offline training complete")
        print(f"  Total updates: {total_updates}")

        return stats

    def save_policy(self, path: str) -> None:
        """Save the agent's policy to disk.

        Args:
            path: Path to save policy
        """
        if hasattr(self.agent, "policy") and hasattr(self.agent.policy, "save"):
            self.agent.policy.save(path)
            print(f"✓ Policy saved to: {path}")
        else:
            raise ValueError("Agent does not have a saveable policy")

    def load_policy(self, path: str) -> None:
        """Load the agent's policy from disk.

        Args:
            path: Path to load policy from
        """
        if hasattr(self.agent, "policy") and hasattr(self.agent.policy, "load"):
            self.agent.policy.load(path)
            print(f"✓ Policy loaded from: {path}")
        else:
            raise ValueError("Agent does not have a loadable policy")
