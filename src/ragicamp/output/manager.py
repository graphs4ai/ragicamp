"""Output management for organizing experiment results."""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import subprocess

from ragicamp.config.experiment import ExperimentConfig


class OutputManager:
    """Manage experiment outputs in organized directory structure.

    Organizes outputs as:
    outputs/
    ├── experiments/
    │   ├── experiment_name_v1/
    │   │   ├── config.yaml
    │   │   ├── metadata.json
    │   │   ├── contexts.json  (if RAG)
    │   │   ├── predictions.json
    │   │   └── results.json
    │   └── experiment_name_v2/
    │       └── ...
    └── comparisons/
        └── dataset_comparison.json

    Example:
        >>> mgr = OutputManager()
        >>> exp_dir = mgr.create_experiment_dir("my_experiment")
        >>> mgr.save_experiment(config, results, exp_dir)
        >>> exps = mgr.list_experiments(dataset="natural_questions")
    """

    def __init__(self, base_dir: str = "outputs"):
        """Initialize output manager.

        Args:
            base_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.comparisons_dir = self.base_dir / "comparisons"

        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_dir(self, experiment_name: str) -> Path:
        """Create directory for new experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to experiment directory
        """
        exp_dir = self.experiments_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save_experiment(
        self, config: ExperimentConfig, results: Dict[str, Any], exp_dir: Path
    ) -> None:
        """Save complete experiment artifacts.

        Args:
            config: Experiment configuration
            results: Results dictionary with metrics
            exp_dir: Directory to save to
        """
        # Save config
        config.to_yaml(exp_dir / "config.yaml")

        # Save metadata
        metadata = {
            "experiment_name": config.name,
            "timestamp": datetime.now().isoformat(),
            "dataset": config.evaluation.dataset,
            "num_examples": config.evaluation.num_examples,
            "corpus": config.corpus.name,
            "retriever_type": config.retriever.type,
            "model": config.model.name,
            "git_commit": self._get_git_commit(),
        }
        self._save_json(metadata, exp_dir / "metadata.json")

        # Save results
        self._save_json(results, exp_dir / "results.json")

        print(f"✓ Saved experiment to: {exp_dir}")

    def list_experiments(
        self, dataset: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered.

        Args:
            dataset: Filter by dataset name (optional)
            limit: Maximum number to return (optional)

        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            metadata_file = exp_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                metadata = json.load(f)

            # Filter by dataset if specified
            if dataset is None or metadata.get("dataset") == dataset:
                metadata["path"] = str(exp_dir)
                experiments.append(metadata)

        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Limit if specified
        if limit:
            experiments = experiments[:limit]

        return experiments

    def get_experiment(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific experiment.

        Args:
            experiment_name: Name of experiment

        Returns:
            Experiment metadata or None if not found
        """
        exp_dir = self.experiments_dir / experiment_name
        metadata_file = exp_dir / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            metadata = json.load(f)

        metadata["path"] = str(exp_dir)
        return metadata

    def compare_experiments(
        self, exp_names: List[str], output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comparison of multiple experiments.

        Args:
            exp_names: List of experiment names to compare
            output_file: Optional file to save comparison (in comparisons/)

        Returns:
            Dictionary mapping experiment names to their results
        """
        comparison = {}

        for exp_name in exp_names:
            exp_dir = self.experiments_dir / exp_name
            results_file = exp_dir / "results.json"

            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)

                # Load metadata too
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    results["metadata"] = metadata

                comparison[exp_name] = results
            else:
                print(f"⚠️  Results not found for: {exp_name}")

        # Save if output file specified
        if output_file:
            output_path = self.comparisons_dir / output_file
            self._save_json(comparison, output_path)
            print(f"✓ Saved comparison to: {output_path}")

        return comparison

    def print_comparison(self, exp_names: List[str], metrics: Optional[List[str]] = None) -> None:
        """Print comparison table to console.

        Args:
            exp_names: List of experiment names
            metrics: Metrics to compare (default: all)
        """
        comparison = self.compare_experiments(exp_names)

        if not comparison:
            print("No experiments to compare")
            return

        # Determine metrics to show
        if metrics is None:
            # Get all metrics from first experiment
            first_exp = next(iter(comparison.values()))
            metrics = [
                k
                for k in first_exp.keys()
                if k not in ["metadata", "num_examples", "agent_name", "dataset_name"]
            ]

        # Print header
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPARISON")
        print("=" * 80)

        # Print table
        print(f"\n{'Experiment':<40} | " + " | ".join(f"{m:>12}" for m in metrics))
        print("-" * 80)

        for exp_name, results in comparison.items():
            values = []
            for metric in metrics:
                value = results.get(metric, 0.0)
                if isinstance(value, float):
                    values.append(f"{value:>12.4f}")
                else:
                    values.append(f"{value:>12}")

            print(f"{exp_name:<40} | " + " | ".join(values))

        print("=" * 80 + "\n")

    def clean_old_experiments(self, keep: int = 10) -> None:
        """Remove old experiment directories, keeping most recent.

        Args:
            keep: Number of most recent experiments to keep
        """
        experiments = self.list_experiments()

        if len(experiments) <= keep:
            print(f"Only {len(experiments)} experiments, nothing to clean")
            return

        to_remove = experiments[keep:]

        print(f"Removing {len(to_remove)} old experiments (keeping {keep} most recent):")
        for exp in to_remove:
            exp_path = Path(exp["path"])
            print(f"  Removing: {exp_path.name}")
            self._rm_tree(exp_path)

        print(f"✓ Cleaned {len(to_remove)} experiments")

    def _save_json(self, data: Any, path: Path) -> None:
        """Save data as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _rm_tree(self, path: Path) -> None:
        """Recursively remove directory."""
        import shutil

        if path.exists():
            shutil.rmtree(path)
