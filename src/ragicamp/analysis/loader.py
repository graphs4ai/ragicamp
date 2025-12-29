"""Load experiment results from disk.

Supports loading from:
- Individual experiment directories (metadata.json, *_summary.json)
- Study comparison files (comparison.json)
- Hydra multirun directories
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    """Single experiment result with parsed metadata.

    All fields are optional to handle different result formats gracefully.
    """

    name: str
    type: str = "unknown"  # direct or rag
    model: str = "unknown"
    dataset: str = "unknown"
    prompt: str = "unknown"
    quantization: str = "unknown"
    retriever: Optional[str] = None
    top_k: Optional[int] = None
    batch_size: int = 1
    num_questions: int = 0

    # Metrics
    f1: float = 0.0
    exact_match: float = 0.0
    bertscore_f1: float = 0.0
    bleurt: float = 0.0

    # Metadata
    duration: float = 0.0
    throughput_qps: float = 0.0
    timestamp: str = ""

    # Raw data for access to all metrics
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary (comparison.json format)."""
        results = data.get("results", {})
        return cls(
            name=data.get("name", "unknown"),
            type=data.get("type", "unknown"),
            model=data.get("model", "unknown"),
            dataset=data.get("dataset", "unknown"),
            prompt=data.get("prompt", "unknown"),
            quantization=data.get("quantization", "unknown"),
            retriever=data.get("retriever"),
            top_k=data.get("top_k"),
            batch_size=data.get("batch_size", 1),
            num_questions=data.get("num_questions", results.get("num_examples", 0)),
            f1=results.get("f1", 0.0),
            exact_match=results.get("exact_match", 0.0),
            bertscore_f1=results.get("bertscore_f1", 0.0),
            bleurt=results.get("bleurt", 0.0),
            duration=data.get("duration", 0.0),
            throughput_qps=data.get("throughput_qps", 0.0),
            timestamp=data.get("timestamp", ""),
            raw=data,
        )

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any], summary: Dict[str, Any]) -> "ExperimentResult":
        """Create from metadata.json + summary.json format."""
        metrics = summary.get("overall_metrics", summary)
        return cls(
            name=metadata.get("name", summary.get("agent_name", "unknown")),
            type=metadata.get("type", "unknown"),
            model=metadata.get("model", "unknown"),
            dataset=metadata.get("dataset", summary.get("dataset_name", "unknown")),
            prompt=metadata.get("prompt", "unknown"),
            quantization=metadata.get("quantization", "unknown"),
            retriever=metadata.get("retriever"),
            top_k=metadata.get("top_k"),
            batch_size=metadata.get("batch_size", 1),
            num_questions=metadata.get("num_questions", summary.get("num_examples", 0)),
            f1=metrics.get("f1", 0.0),
            exact_match=metrics.get("exact_match", 0.0),
            bertscore_f1=metrics.get("bertscore_f1", 0.0),
            bleurt=metrics.get("bleurt", 0.0),
            duration=metadata.get("duration", 0.0),
            throughput_qps=metadata.get("throughput_qps", 0.0),
            timestamp=metadata.get("timestamp", summary.get("timestamp", "")),
            raw={**metadata, "summary": summary},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "model": self.model,
            "dataset": self.dataset,
            "prompt": self.prompt,
            "quantization": self.quantization,
            "retriever": self.retriever,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "num_questions": self.num_questions,
            "f1": self.f1,
            "exact_match": self.exact_match,
            "bertscore_f1": self.bertscore_f1,
            "bleurt": self.bleurt,
            "duration": self.duration,
            "throughput_qps": self.throughput_qps,
            "timestamp": self.timestamp,
        }

    @property
    def model_short(self) -> str:
        """Short model name without prefix."""
        name = self.model
        if name.startswith("hf:"):
            name = name[3:]
        return name.split("/")[-1]

    @property
    def retriever_short(self) -> str:
        """Short retriever name."""
        if not self.retriever:
            return "none"
        return self.retriever.replace("simple_", "").replace("recursive_", "")


class ResultsLoader:
    """Load experiment results from a study directory.

    Supports multiple formats:
    - comparison.json (aggregated results from run_study.py)
    - Individual experiment directories with metadata.json
    - Hydra multirun directories

    Example:
        loader = ResultsLoader("outputs/comprehensive_baseline")
        results = loader.load_all()
        print(f"Loaded {len(results)} experiments")
    """

    def __init__(self, base_dir: Union[str, Path]):
        """Initialize loader.

        Args:
            base_dir: Base directory containing experiment results
        """
        self.base_dir = Path(base_dir)

    def load_all(self) -> List[ExperimentResult]:
        """Load all experiment results from the directory.

        Tries comparison.json first (fastest), falls back to individual dirs.

        Returns:
            List of ExperimentResult objects
        """
        # Try comparison.json first (most complete)
        comparison_file = self.base_dir / "comparison.json"
        if comparison_file.exists():
            return self._load_from_comparison(comparison_file)

        # Fall back to individual directories
        return self._load_from_directories()

    def _load_from_comparison(self, path: Path) -> List[ExperimentResult]:
        """Load from comparison.json file."""
        logger.debug("Loading from comparison.json: %s", path)
        with open(path) as f:
            data = json.load(f)

        experiments = data.get("experiments", [])
        results = [ExperimentResult.from_dict(exp) for exp in experiments]
        logger.info("Loaded %d experiments from comparison.json", len(results))
        return results

    def _load_from_directories(self) -> List[ExperimentResult]:
        """Load from individual experiment directories."""
        results = []

        # Find all metadata.json files
        for metadata_path in self.base_dir.rglob("metadata.json"):
            exp_dir = metadata_path.parent

            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Find corresponding summary file
                summary_files = list(exp_dir.glob("*_summary.json"))
                if not summary_files:
                    logger.warning("No summary file for %s", exp_dir.name)
                    continue

                with open(summary_files[0]) as f:
                    summary = json.load(f)

                result = ExperimentResult.from_metadata(metadata, summary)
                results.append(result)

            except Exception as e:
                logger.warning("Failed to load %s: %s", exp_dir.name, e)

        logger.info("Loaded %d experiments from directories", len(results))
        return results

    def load_predictions(self, experiment_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load predictions for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            List of prediction dicts or None if not found
        """
        exp_dir = self.base_dir / experiment_name

        # Find predictions file
        pred_files = list(exp_dir.glob("*_predictions.json"))
        if not pred_files:
            return None

        with open(pred_files[0]) as f:
            data = json.load(f)

        return data.get("predictions", [])

