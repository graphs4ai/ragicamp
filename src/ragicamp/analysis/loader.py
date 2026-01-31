"""Load experiment results from disk.

Supports loading from:
- Individual experiment directories (metadata.json, results.json, predictions.json)
- Legacy format: metadata.json + *_summary.json + *_predictions.json
- Study comparison files (comparison.json)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

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
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bleurt: float = 0.0
    llm_judge: Optional[float] = None

    # Metadata
    duration: float = 0.0
    throughput_qps: float = 0.0
    timestamp: str = ""

    # Raw data for access to all metrics
    raw: dict[str, Any] = field(default_factory=dict)

    # Parsed RAG details (extracted from retriever name)
    corpus: str = "unknown"
    embedding_model: str = "unknown"
    chunk_size: int = 0
    chunk_strategy: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary (comparison.json format)."""
        # Support both "results" (old format) and "metrics" (new format)
        results = data.get("results", {}) or data.get("metrics", {})
        retriever = data.get("retriever")

        # Parse RAG details from retriever name
        # Format: {corpus}_{embedding}_{strategy}_{chunk_size}
        # e.g., simple_minilm_recursive_1024
        corpus = "unknown"
        embedding_model = "unknown"
        chunk_strategy = "unknown"
        chunk_size = 0

        if retriever:
            parts = retriever.split("_")
            if len(parts) >= 4:
                corpus = parts[0]  # simple, en
                embedding_model = parts[1]  # minilm, e5, mpnet
                chunk_strategy = parts[2]  # recursive, fixed
                try:
                    chunk_size = int(parts[3])  # 512, 1024
                except (ValueError, IndexError):
                    pass

        return cls(
            name=data.get("name", "unknown"),
            type=data.get("type", "unknown"),
            model=data.get("model", "unknown"),
            dataset=data.get("dataset", "unknown"),
            prompt=data.get("prompt", "unknown"),
            quantization=data.get("quantization", "unknown"),
            retriever=retriever,
            top_k=data.get("top_k"),
            batch_size=data.get("batch_size", 1),
            num_questions=data.get("num_questions", results.get("num_examples", 0)),
            f1=results.get("f1", 0.0),
            exact_match=results.get("exact_match", 0.0),
            bertscore_f1=results.get("bertscore_f1", 0.0),
            bertscore_precision=results.get("bertscore_precision", 0.0),
            bertscore_recall=results.get("bertscore_recall", 0.0),
            bleurt=results.get("bleurt", 0.0),
            llm_judge=results.get("llm_judge_qa") or results.get("llm_judge"),
            duration=data.get("duration", 0.0),
            throughput_qps=data.get("throughput_qps", 0.0),
            timestamp=data.get("timestamp", ""),
            raw=data,
            corpus=corpus,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_strategy=chunk_strategy,
        )

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], summary: dict[str, Any]) -> "ExperimentResult":
        """Create from metadata.json + summary.json format."""
        metrics = summary.get("overall_metrics", summary)
        retriever = metadata.get("retriever")

        # Parse RAG details from retriever name
        corpus = "unknown"
        embedding_model = "unknown"
        chunk_strategy = "unknown"
        chunk_size = 0

        if retriever:
            parts = retriever.split("_")
            if len(parts) >= 4:
                corpus = parts[0]
                embedding_model = parts[1]
                chunk_strategy = parts[2]
                try:
                    chunk_size = int(parts[3])
                except (ValueError, IndexError):
                    pass

        return cls(
            name=metadata.get("name", summary.get("agent_name", "unknown")),
            type=metadata.get("type", "unknown"),
            model=metadata.get("model", "unknown"),
            dataset=metadata.get("dataset", summary.get("dataset_name", "unknown")),
            prompt=metadata.get("prompt", "unknown"),
            quantization=metadata.get("quantization", "unknown"),
            retriever=retriever,
            top_k=metadata.get("top_k"),
            batch_size=metadata.get("batch_size", 1),
            num_questions=metadata.get("num_questions", summary.get("num_examples", 0)),
            f1=metrics.get("f1", 0.0),
            exact_match=metrics.get("exact_match", 0.0),
            bertscore_f1=metrics.get("bertscore_f1", 0.0),
            bertscore_precision=metrics.get("bertscore_precision", 0.0),
            bertscore_recall=metrics.get("bertscore_recall", 0.0),
            bleurt=metrics.get("bleurt", 0.0),
            llm_judge=metrics.get("llm_judge_qa") or metrics.get("llm_judge"),
            duration=metadata.get("duration", 0.0),
            throughput_qps=metadata.get("throughput_qps", 0.0),
            timestamp=metadata.get("timestamp", summary.get("timestamp", "")),
            raw={**metadata, "summary": summary},
            corpus=corpus,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_strategy=chunk_strategy,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "type": self.type,
            "model": self.model,
            "model_short": self.model_short,
            "dataset": self.dataset,
            "prompt": self.prompt,
            "quantization": self.quantization,
            "retriever": self.retriever,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "num_questions": self.num_questions,
            # Metrics
            "f1": self.f1,
            "exact_match": self.exact_match,
            "bertscore_f1": self.bertscore_f1,
            "bertscore_precision": self.bertscore_precision,
            "bertscore_recall": self.bertscore_recall,
            "bleurt": self.bleurt,
            # Timing
            "duration": self.duration,
            "throughput_qps": self.throughput_qps,
            "timestamp": self.timestamp,
            # Parsed RAG details
            "corpus": self.corpus,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_strategy": self.chunk_strategy,
        }
        if self.llm_judge is not None:
            result["llm_judge"] = self.llm_judge
        return result

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

    def load_all(self) -> list[ExperimentResult]:
        """Load all experiment results from the directory.

        Tries comparison.json first (fastest), falls back to individual dirs.

        Returns:
            List of ExperimentResult objects
        """
        # Try study_summary.json first (aggregated results)
        summary_file = self.base_dir / "study_summary.json"
        if summary_file.exists():
            return self._load_from_comparison(summary_file)

        # Legacy: comparison.json
        comparison_file = self.base_dir / "comparison.json"
        if comparison_file.exists():
            return self._load_from_comparison(comparison_file)

        # Fall back to individual directories
        return self._load_from_directories()

    def _load_from_comparison(self, path: Path) -> list[ExperimentResult]:
        """Load from comparison.json file, enriching with llm_judge from summary files."""
        logger.debug("Loading from comparison.json: %s", path)
        with open(path) as f:
            data = json.load(f)

        experiments = data.get("experiments", [])
        results = []

        for exp in experiments:
            # Check summary file for llm_judge before creating result
            exp_name = exp.get("name", "")
            exp_dir = self.base_dir / exp_name
            summary_files = list(exp_dir.glob("*_summary.json")) if exp_dir.exists() else []

            if summary_files:
                try:
                    with open(summary_files[0]) as f:
                        summary = json.load(f)
                    metrics = summary.get("overall_metrics", summary)
                    llm_judge = metrics.get("llm_judge_qa") or metrics.get("llm_judge")
                    if llm_judge is not None:
                        # Add llm_judge to results before parsing
                        if "results" not in exp:
                            exp["results"] = {}
                        exp["results"]["llm_judge_qa"] = llm_judge
                except Exception:
                    pass

            result = ExperimentResult.from_dict(exp)
            results.append(result)

        logger.info("Loaded %d experiments from comparison.json", len(results))
        return results

    def _load_from_directories(self) -> list[ExperimentResult]:
        """Load from individual experiment directories."""
        results = []

        # Find all metadata.json files
        for metadata_path in self.base_dir.rglob("metadata.json"):
            exp_dir = metadata_path.parent

            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Try new format: results.json
                results_file = exp_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        summary = json.load(f)
                    # Adapt to from_metadata expected format
                    if "overall_metrics" not in summary:
                        summary["overall_metrics"] = summary.get("metrics", {})
                else:
                    # Fall back to legacy: *_summary.json
                    summary_files = list(exp_dir.glob("*_summary.json"))
                    if not summary_files:
                        logger.warning("No results/summary file for %s", exp_dir.name)
                        continue
                    with open(summary_files[0]) as f:
                        summary = json.load(f)

                result = ExperimentResult.from_metadata(metadata, summary)
                results.append(result)

            except Exception as e:
                logger.warning("Failed to load %s: %s", exp_dir.name, e)

        logger.info("Loaded %d experiments from directories", len(results))
        return results

    def load_predictions(self, experiment_name: str) -> Optional[list[dict[str, Any]]]:
        """Load predictions for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            List of prediction dicts or None if not found
        """
        exp_dir = self.base_dir / experiment_name

        # Try new format: predictions.json
        pred_file = exp_dir / "predictions.json"
        if pred_file.exists():
            with open(pred_file) as f:
                data = json.load(f)
            return data.get("predictions", [])

        # Fall back to legacy: *_predictions.json
        pred_files = list(exp_dir.glob("*_predictions.json"))
        if not pred_files:
            return None

        with open(pred_files[0]) as f:
            data = json.load(f)

        return data.get("predictions", [])
