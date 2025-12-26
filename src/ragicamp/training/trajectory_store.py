"""Trajectory storage for RL training.

This module provides persistent storage for RL trajectories, enabling:
- Collection of (state, action, reward) tuples during evaluation
- Offline training from stored trajectories
- Experience replay with random sampling

Example:
    >>> from ragicamp.training import TrajectoryStore, Trajectory
    >>>
    >>> # Collect trajectories during evaluation
    >>> store = TrajectoryStore("trajectories/nq_gemma2b")
    >>> for question, answer, reward in evaluation_results:
    ...     trajectory = Trajectory(
    ...         question_id=question.id,
    ...         question=question.text,
    ...         steps=[{"state": ..., "action": ..., "reward": ...}],
    ...         final_answer=answer,
    ...         final_reward=reward,
    ...     )
    ...     store.save(trajectory)
    >>>
    >>> # Later, train from trajectories
    >>> for batch in store.iter_batches(batch_size=32):
    ...     policy.update(batch)
"""

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Trajectory:
    """Single episode trajectory for RL training.

    Represents one question-answering episode with:
    - The question and context
    - Intermediate steps taken (for multi-step agents)
    - Final answer and reward

    Attributes:
        question_id: Unique identifier for the question
        question: The question text
        steps: List of (state, action, intermediate_reward) dicts
        final_answer: The final generated answer
        final_reward: The reward/score for this episode
        metadata: Additional information (model, retriever params, etc.)
        timestamp: When this trajectory was collected
    """

    question_id: str
    question: str
    steps: List[Dict[str, Any]]
    final_answer: str
    final_reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create from dictionary."""
        return cls(**data)

    def get_total_steps(self) -> int:
        """Get number of steps in this trajectory."""
        return len(self.steps)

    def get_discounted_return(self, gamma: float = 0.99) -> float:
        """Compute discounted return from intermediate rewards.

        Args:
            gamma: Discount factor

        Returns:
            Discounted sum of rewards
        """
        total = 0.0
        discount = 1.0

        for step in self.steps:
            reward = step.get("reward", 0.0)
            total += discount * reward
            discount *= gamma

        # Add final reward
        total += discount * self.final_reward
        return total


class TrajectoryStore:
    """Persistent storage for RL trajectories.

    Stores trajectories as individual JSON files for:
    - Incremental collection (append without loading all)
    - Efficient random sampling
    - Easy inspection and debugging

    Directory structure:
        store_path/
        ├── metadata.json
        ├── trajectories/
        │   ├── 00000001.json
        │   ├── 00000002.json
        │   └── ...
        └── index.json  (optional, for fast lookup)
    """

    def __init__(self, path: str, create: bool = True):
        """Initialize trajectory store.

        Args:
            path: Directory path for the store
            create: Create directory if it doesn't exist
        """
        self.path = Path(path)
        self.trajectories_dir = self.path / "trajectories"
        self.metadata_path = self.path / "metadata.json"

        if create:
            self.trajectories_dir.mkdir(parents=True, exist_ok=True)

        # Load or create metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {
                "created_at": datetime.now().isoformat(),
                "count": 0,
                "total_reward": 0.0,
            }
            self._save_metadata()

    def _save_metadata(self):
        """Save metadata atomically."""
        temp_path = self.metadata_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
        shutil.move(str(temp_path), str(self.metadata_path))

    def save(self, trajectory: Trajectory) -> str:
        """Save a trajectory to the store.

        Args:
            trajectory: Trajectory to save

        Returns:
            Path to saved trajectory file
        """
        # Generate filename based on count
        self._metadata["count"] += 1
        count = self._metadata["count"]
        filename = f"{count:08d}.json"
        filepath = self.trajectories_dir / filename

        # Save trajectory
        with open(filepath, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2)

        # Update metadata
        self._metadata["total_reward"] += trajectory.final_reward
        self._metadata["updated_at"] = datetime.now().isoformat()
        self._save_metadata()

        return str(filepath)

    def save_batch(self, trajectories: List[Trajectory]) -> List[str]:
        """Save multiple trajectories.

        Args:
            trajectories: List of trajectories to save

        Returns:
            List of saved file paths
        """
        paths = []
        for traj in trajectories:
            paths.append(self.save(traj))
        return paths

    def load(self, trajectory_id: int) -> Optional[Trajectory]:
        """Load a specific trajectory by ID.

        Args:
            trajectory_id: The trajectory number (1-indexed)

        Returns:
            Trajectory or None if not found
        """
        filename = f"{trajectory_id:08d}.json"
        filepath = self.trajectories_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        return Trajectory.from_dict(data)

    def load_all(self) -> List[Trajectory]:
        """Load all trajectories from the store.

        Returns:
            List of all trajectories
        """
        trajectories = []
        for filepath in sorted(self.trajectories_dir.glob("*.json")):
            with open(filepath, "r") as f:
                data = json.load(f)
            trajectories.append(Trajectory.from_dict(data))
        return trajectories

    def sample(self, n: int, replace: bool = False) -> List[Trajectory]:
        """Randomly sample trajectories from the store.

        Args:
            n: Number of trajectories to sample
            replace: Sample with replacement

        Returns:
            List of sampled trajectories
        """
        all_files = list(self.trajectories_dir.glob("*.json"))

        if not all_files:
            return []

        if replace:
            selected = random.choices(all_files, k=n)
        else:
            n = min(n, len(all_files))
            selected = random.sample(all_files, n)

        trajectories = []
        for filepath in selected:
            with open(filepath, "r") as f:
                data = json.load(f)
            trajectories.append(Trajectory.from_dict(data))

        return trajectories

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[List[Trajectory]]:
        """Iterate over trajectories in batches.

        Args:
            batch_size: Number of trajectories per batch
            shuffle: Shuffle before iterating

        Yields:
            Batches of trajectories
        """
        all_files = list(self.trajectories_dir.glob("*.json"))

        if shuffle:
            random.shuffle(all_files)

        batch = []
        for filepath in all_files:
            with open(filepath, "r") as f:
                data = json.load(f)
            batch.append(Trajectory.from_dict(data))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining
        if batch:
            yield batch

    def filter(
        self,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> List[Trajectory]:
        """Filter trajectories by criteria.

        Args:
            min_reward: Minimum final reward
            max_reward: Maximum final reward
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps

        Returns:
            Filtered list of trajectories
        """
        filtered = []

        for filepath in self.trajectories_dir.glob("*.json"):
            with open(filepath, "r") as f:
                data = json.load(f)

            traj = Trajectory.from_dict(data)

            # Apply filters
            if min_reward is not None and traj.final_reward < min_reward:
                continue
            if max_reward is not None and traj.final_reward > max_reward:
                continue
            if min_steps is not None and len(traj.steps) < min_steps:
                continue
            if max_steps is not None and len(traj.steps) > max_steps:
                continue

            filtered.append(traj)

        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored trajectories.

        Returns:
            Dictionary with statistics
        """
        count = self._metadata.get("count", 0)

        if count == 0:
            return {
                "count": 0,
                "mean_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            }

        # Compute statistics
        rewards = []
        step_counts = []

        for filepath in self.trajectories_dir.glob("*.json"):
            with open(filepath, "r") as f:
                data = json.load(f)
            rewards.append(data["final_reward"])
            step_counts.append(len(data.get("steps", [])))

        return {
            "count": count,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "mean_steps": sum(step_counts) / len(step_counts) if step_counts else 0.0,
            "created_at": self._metadata.get("created_at"),
            "updated_at": self._metadata.get("updated_at"),
        }

    def clear(self):
        """Remove all trajectories from the store."""
        for filepath in self.trajectories_dir.glob("*.json"):
            filepath.unlink()

        self._metadata = {
            "created_at": datetime.now().isoformat(),
            "count": 0,
            "total_reward": 0.0,
        }
        self._save_metadata()

    def __len__(self) -> int:
        """Get number of stored trajectories."""
        return self._metadata.get("count", 0)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"TrajectoryStore(path='{self.path}', "
            f"count={stats['count']}, "
            f"mean_reward={stats['mean_reward']:.3f})"
        )
