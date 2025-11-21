"""Bandit algorithms for parameter selection."""

import json
import random
from typing import Any, Dict, List

import numpy as np

from ragicamp.policies.base import Policy


class EpsilonGreedyBandit(Policy):
    """Epsilon-greedy bandit for discrete action selection.

    Maintains Q-values for each action and selects:
    - Best action with probability (1 - epsilon)
    - Random action with probability epsilon
    """

    def __init__(
        self, name: str, actions: List[Dict[str, Any]], epsilon: float = 0.1, **kwargs: Any
    ):
        """Initialize epsilon-greedy bandit.

        Args:
            name: Policy identifier
            actions: List of possible actions (parameter configurations)
            epsilon: Exploration rate
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.actions = actions
        self.epsilon = epsilon

        # Initialize Q-values and counts
        self.q_values = np.zeros(len(actions))
        self.action_counts = np.zeros(len(actions))

    def select_action(self, **context: Any) -> Dict[str, Any]:
        """Select action using epsilon-greedy strategy."""
        # Exploration
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.actions) - 1)
        # Exploitation
        else:
            action_idx = int(np.argmax(self.q_values))

        return self.actions[action_idx].copy()

    def update(self, action: Dict[str, Any], reward: float, **kwargs: Any) -> None:
        """Update Q-values based on observed reward.

        Args:
            action: The action that was taken
            reward: The observed reward
            **kwargs: Additional feedback
        """
        # Find action index
        action_idx = self._find_action_index(action)
        if action_idx is None:
            return

        # Update counts
        self.action_counts[action_idx] += 1

        # Update Q-value using incremental mean
        n = self.action_counts[action_idx]
        old_q = self.q_values[action_idx]
        self.q_values[action_idx] = old_q + (reward - old_q) / n

    def _find_action_index(self, action: Dict[str, Any]) -> int:
        """Find index of action in action list."""
        for i, a in enumerate(self.actions):
            if a == action:
                return i
        return None

    def save(self, path: str) -> None:
        """Save policy state to disk."""
        state = {
            "actions": self.actions,
            "epsilon": self.epsilon,
            "q_values": self.q_values.tolist(),
            "action_counts": self.action_counts.tolist(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load policy state from disk."""
        with open(path, "r") as f:
            state = json.load(f)

        self.actions = state["actions"]
        self.epsilon = state["epsilon"]
        self.q_values = np.array(state["q_values"])
        self.action_counts = np.array(state["action_counts"])


class UCBBandit(Policy):
    """Upper Confidence Bound (UCB) bandit algorithm.

    Balances exploration and exploitation using confidence bounds.
    """

    def __init__(self, name: str, actions: List[Dict[str, Any]], c: float = 2.0, **kwargs: Any):
        """Initialize UCB bandit.

        Args:
            name: Policy identifier
            actions: List of possible actions
            c: Exploration parameter
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.actions = actions
        self.c = c

        # Initialize Q-values and counts
        self.q_values = np.zeros(len(actions))
        self.action_counts = np.zeros(len(actions))
        self.total_count = 0

    def select_action(self, **context: Any) -> Dict[str, Any]:
        """Select action using UCB strategy."""
        # Try each action at least once
        if np.any(self.action_counts == 0):
            action_idx = int(np.argmin(self.action_counts))
        else:
            # Compute UCB values
            ucb_values = self.q_values + self.c * np.sqrt(
                np.log(self.total_count) / self.action_counts
            )
            action_idx = int(np.argmax(ucb_values))

        return self.actions[action_idx].copy()

    def update(self, action: Dict[str, Any], reward: float, **kwargs: Any) -> None:
        """Update based on observed reward."""
        action_idx = self._find_action_index(action)
        if action_idx is None:
            return

        # Update counts
        self.action_counts[action_idx] += 1
        self.total_count += 1

        # Update Q-value
        n = self.action_counts[action_idx]
        old_q = self.q_values[action_idx]
        self.q_values[action_idx] = old_q + (reward - old_q) / n

    def _find_action_index(self, action: Dict[str, Any]) -> int:
        """Find index of action in action list."""
        for i, a in enumerate(self.actions):
            if a == action:
                return i
        return None

    def save(self, path: str) -> None:
        """Save policy state to disk."""
        state = {
            "actions": self.actions,
            "c": self.c,
            "q_values": self.q_values.tolist(),
            "action_counts": self.action_counts.tolist(),
            "total_count": self.total_count,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load policy state from disk."""
        with open(path, "r") as f:
            state = json.load(f)

        self.actions = state["actions"]
        self.c = state["c"]
        self.q_values = np.array(state["q_values"])
        self.action_counts = np.array(state["action_counts"])
        self.total_count = state["total_count"]
