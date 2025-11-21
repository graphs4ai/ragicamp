"""MDP-based policies for sequential decision making."""

import json
import random
from typing import Any, Dict, List

import numpy as np

from ragicamp.policies.base import Policy


class RandomMDPPolicy(Policy):
    """Random policy for MDP (baseline).

    Randomly selects actions for debugging and baseline comparison.
    """

    def __init__(self, name: str, action_types: List[str] = None, **kwargs: Any):
        """Initialize random MDP policy.

        Args:
            name: Policy identifier
            action_types: List of possible action types
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.action_types = action_types or ["retrieve", "reformulate", "generate"]
        self.action_params = {
            "retrieve": [{"top_k": 3}, {"top_k": 5}, {"top_k": 10}],
            "reformulate": [{}],
            "generate": [{}],
        }

    def select_action(self, state: Dict[str, Any] = None, **context: Any) -> Dict[str, Any]:
        """Randomly select an action."""
        # Simple strategy: retrieve a few times, then generate
        if state and state.get("step", 0) >= 2:
            action_type = "generate"
        else:
            action_type = random.choice(["retrieve", "reformulate", "generate"])

        # Select random parameters
        params = random.choice(self.action_params.get(action_type, [{}]))

        return {"type": action_type, "params": params}

    def update(
        self, trajectory: List[Dict[str, Any]] = None, reward: float = 0.0, **kwargs: Any
    ) -> None:
        """No learning for random policy."""
        pass


class QLearningMDPPolicy(Policy):
    """Q-learning policy for MDP-based RAG.

    Learns state-action values for sequential RAG decisions.
    """

    def __init__(
        self,
        name: str,
        action_types: List[str] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize Q-learning policy.

        Args:
            name: Policy identifier
            action_types: List of possible action types
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.action_types = action_types or ["retrieve", "reformulate", "generate"]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-table: state -> action -> value
        # For simplicity, we'll use a dict with state features as key
        self.q_table: Dict[str, Dict[str, float]] = {}

    def select_action(self, state: Dict[str, Any] = None, **context: Any) -> Dict[str, Any]:
        """Select action using epsilon-greedy Q-learning."""
        state_key = self._get_state_key(state)

        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.action_types}

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            action_type = random.choice(self.action_types)
        else:
            action_type = max(self.q_table[state_key].items(), key=lambda x: x[1])[0]

        # Generate parameters based on action type
        params = self._get_action_params(action_type, state)

        return {"type": action_type, "params": params}

    def update(
        self, trajectory: List[Dict[str, Any]] = None, reward: float = 0.0, **kwargs: Any
    ) -> None:
        """Update Q-values based on trajectory and final reward.

        Args:
            trajectory: List of (state, action) tuples
            reward: Final reward
            **kwargs: Additional feedback
        """
        if not trajectory:
            return

        # Backward update from final reward
        current_reward = reward

        for step in reversed(trajectory):
            state = step.get("state")
            action = step.get("action")

            if not state or not action:
                continue

            state_key = self._get_state_key(state)
            action_type = action.get("type", "generate")

            # Initialize if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in self.action_types}

            # Q-learning update
            old_q = self.q_table[state_key].get(action_type, 0.0)

            # TD update: Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            # For episodic tasks, we propagate the final reward backward
            self.q_table[state_key][action_type] = old_q + self.learning_rate * (
                current_reward - old_q
            )

            # Discount for previous steps
            current_reward *= self.discount_factor

    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Convert state to a hashable key.

        For simplicity, we use a few key features.
        """
        if not state:
            return "initial"

        # Extract key features
        step = state.get("step", 0)
        num_docs = len(state.get("retrieved_docs", []))
        num_reformulations = len(state.get("reformulations", []))

        return f"step_{step}_docs_{num_docs}_reformulations_{num_reformulations}"

    def _get_action_params(self, action_type: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for an action based on state."""
        if action_type == "retrieve":
            # Adaptive top_k based on state
            num_docs = len(state.get("retrieved_docs", [])) if state else 0
            if num_docs == 0:
                return {"top_k": 5}
            else:
                return {"top_k": 3}

        return {}

    def save(self, path: str) -> None:
        """Save Q-table to disk."""
        state = {
            "action_types": self.action_types,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "q_table": self.q_table,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load Q-table from disk."""
        with open(path, "r") as f:
            state = json.load(f)

        self.action_types = state["action_types"]
        self.learning_rate = state["learning_rate"]
        self.discount_factor = state["discount_factor"]
        self.epsilon = state["epsilon"]
        self.q_table = state["q_table"]
