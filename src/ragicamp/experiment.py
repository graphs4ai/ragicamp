"""Unified Experiment abstraction for RAGiCamp.

This module provides a single, clean interface for running experiments:

    from ragicamp import Experiment

    exp = Experiment.from_config("conf/study/my_study.yaml")
    results = exp.run()

Or programmatically:

    exp = Experiment(
        name="my_experiment",
        model=model,
        agent=agent,
        dataset=dataset,
        metrics=metrics,
    )
    results = exp.run(batch_size=8, checkpoint_every=50)
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ragicamp.agents.base import RAGAgent
from ragicamp.core.logging import get_logger

logger = get_logger(__name__)
from ragicamp.datasets.base import QADataset
from ragicamp.metrics.base import Metric
from ragicamp.models.base import LanguageModel
from ragicamp.utils.paths import ensure_dir
from ragicamp.utils.resource_manager import ResourceManager


@dataclass
class ExperimentCallbacks:
    """Callbacks for monitoring experiment progress.

    All callbacks are optional. Set them to receive notifications at key points:

    Example:
        callbacks = ExperimentCallbacks(
            on_batch_start=lambda i, n: print(f"Starting batch {i}/{n}"),
            on_complete=lambda r: print(f"Done! F1={r.f1:.3f}"),
        )
        exp.run(callbacks=callbacks)
    """

    on_batch_start: Optional[Callable[[int, int], None]] = None
    """Called before each batch. Args: (batch_index, total_batches)"""

    on_batch_end: Optional[Callable[[int, int, List[str]], None]] = None
    """Called after each batch. Args: (batch_index, total_batches, predictions)"""

    on_checkpoint: Optional[Callable[[int, Path], None]] = None
    """Called when checkpoint is saved. Args: (num_predictions, checkpoint_path)"""

    on_complete: Optional[Callable[["ExperimentResult"], None]] = None
    """Called when experiment completes. Args: (result)"""


@dataclass
class ExperimentResult:
    """Result of running an experiment."""

    name: str
    metrics: Dict[str, float]
    num_examples: int
    duration_seconds: float
    output_path: Optional[Path] = None
    predictions_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def f1(self) -> float:
        return self.metrics.get("f1", 0.0)

    @property
    def exact_match(self) -> float:
        return self.metrics.get("exact_match", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "num_examples": self.num_examples,
            "duration_seconds": self.duration_seconds,
            "output_path": str(self.output_path) if self.output_path else None,
            "predictions_path": str(self.predictions_path) if self.predictions_path else None,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        ensure_dir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        return f"ExperimentResult(name={self.name!r}, f1={self.f1:.3f}, em={self.exact_match:.3f})"


@dataclass
class Experiment:
    """Unified experiment abstraction.

    Combines model, agent, dataset, and metrics into a single runnable unit.
    Handles checkpointing, resource management, and result saving automatically.
    """

    name: str
    agent: RAGAgent
    dataset: QADataset
    metrics: List[Metric] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Optional references for cleanup
    _model: Optional[LanguageModel] = field(default=None, repr=False)

    def run(
        self,
        batch_size: int = 1,
        checkpoint_every: int = 50,
        resume: bool = True,
        save_predictions: bool = True,
        callbacks: Optional[ExperimentCallbacks] = None,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Run the experiment.

        Args:
            batch_size: Number of examples to process in parallel
            checkpoint_every: Save checkpoint every N examples
            resume: Resume from checkpoint if available
            save_predictions: Save predictions to disk
            callbacks: Optional callbacks for progress monitoring
            **kwargs: Additional arguments passed to agent.answer()

        Returns:
            ExperimentResult with metrics and metadata
        """
        callbacks = callbacks or ExperimentCallbacks()
        import torch
        from tqdm import tqdm

        start_time = time.time()
        output_path = self.output_dir / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        predictions_path = output_path / "predictions.json"
        checkpoint_path = output_path / "checkpoint.json"
        results_path = output_path / "results.json"

        # Check if already completed
        if resume and results_path.exists():
            logger.info("Experiment %s already completed. Loading results...", self.name)
            with open(results_path) as f:
                data = json.load(f)
            return ExperimentResult(
                name=self.name,
                metrics=data.get("metrics", {}),
                num_examples=data.get("num_examples", 0),
                duration_seconds=data.get("duration_seconds", 0),
                output_path=output_path,
                predictions_path=predictions_path if predictions_path.exists() else None,
            )

        logger.info("Running: %s", self.name)

        ResourceManager.clear_gpu_memory()

        examples = list(self.dataset)
        predictions = []
        references = []
        questions = []
        start_idx = 0

        # Resume from checkpoint
        if resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    checkpoint = json.load(f)
                predictions = checkpoint.get("predictions", [])
                references = checkpoint.get("references", [])
                questions = checkpoint.get("questions", [])
                start_idx = len(predictions)
                logger.info("Resumed from checkpoint: %d/%d", start_idx, len(examples))
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s", e)
                start_idx = 0

        # Generate predictions
        logger.info("Generating answers for %d examples...", len(examples))

        if batch_size > 1 and hasattr(self.agent, "batch_answer"):
            # Batch processing
            for i in tqdm(range(start_idx, len(examples), batch_size), desc="Batches"):
                batch = examples[i : i + batch_size]
                batch_queries = [ex.question for ex in batch]

                try:
                    responses = self.agent.batch_answer(batch_queries, **kwargs)
                    for ex, resp in zip(batch, responses):
                        predictions.append(resp.answer)
                        references.append(ex.answers)
                        questions.append(ex.question)
                except Exception as e:
                    logger.warning("Batch failed: %s", e)
                    for ex in batch:
                        predictions.append(f"[ERROR: {str(e)[:50]}]")
                        references.append(ex.answers)
                        questions.append(ex.question)

                # Checkpoint
                if checkpoint_every and len(predictions) % checkpoint_every == 0:
                    self._save_checkpoint(checkpoint_path, predictions, references, questions)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Sequential processing
            for i, ex in enumerate(
                tqdm(examples[start_idx:], desc="Questions", initial=start_idx, total=len(examples))
            ):
                try:
                    response = self.agent.answer(ex.question, **kwargs)
                    predictions.append(response.answer)
                except Exception as e:
                    predictions.append(f"[ERROR: {str(e)[:50]}]")

                references.append(ex.answers)
                questions.append(ex.question)

                # Checkpoint
                if checkpoint_every and len(predictions) % checkpoint_every == 0:
                    self._save_checkpoint(checkpoint_path, predictions, references, questions)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Unload model before computing metrics
        self._unload_model()

        # Compute metrics
        logger.info("Computing metrics...")
        metric_results = {}

        for metric in self.metrics:
            try:
                logger.debug("Computing %s...", metric.name)
                if metric.name == "llm_judge":
                    scores = metric.compute(
                        predictions=predictions, references=references, questions=questions
                    )
                else:
                    scores = metric.compute(predictions=predictions, references=references)
                metric_results.update(scores)
            except Exception as e:
                logger.warning("%s failed: %s", metric.name, e)

        duration = time.time() - start_time

        # Build result
        result = ExperimentResult(
            name=self.name,
            metrics=metric_results,
            num_examples=len(examples),
            duration_seconds=duration,
            output_path=output_path,
            metadata={
                "agent": self.agent.name,
                "dataset": self.dataset.name,
                "batch_size": batch_size,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Save predictions
        if save_predictions:
            predictions_data = {
                "experiment": self.name,
                "predictions": [
                    {"question": q, "prediction": p, "expected": r}
                    for q, p, r in zip(questions, predictions, references)
                ],
            }
            with open(predictions_path, "w") as f:
                json.dump(predictions_data, f, indent=2)
            result.predictions_path = predictions_path

        # Save results
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info(
            "Done! F1=%.1f%% EM=%.1f%% (%.1fs)",
            result.f1 * 100,
            result.exact_match * 100,
            duration,
        )

        return result

    def _save_checkpoint(
        self, path: Path, predictions: List[str], references: List, questions: List[str]
    ) -> None:
        """Save checkpoint atomically."""
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "references": references,
                    "questions": questions,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )
        temp_path.replace(path)

    def _unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if self._model and hasattr(self._model, "unload"):
            self._model.unload()
        elif hasattr(self.agent, "model") and hasattr(self.agent.model, "unload"):
            self.agent.model.unload()
        ResourceManager.clear_gpu_memory()

    @classmethod
    def from_spec(
        cls,
        name: str,
        model: LanguageModel,
        agent: RAGAgent,
        dataset: QADataset,
        metrics: List[Metric],
        output_dir: Union[str, Path] = "outputs",
    ) -> "Experiment":
        """Create experiment from components.

        Args:
            name: Experiment name
            model: Language model
            agent: RAG agent
            dataset: Evaluation dataset
            metrics: List of metrics
            output_dir: Output directory

        Returns:
            Configured Experiment
        """
        return cls(
            name=name,
            agent=agent,
            dataset=dataset,
            metrics=metrics,
            output_dir=Path(output_dir),
            _model=model,
        )


def run_experiments(
    experiments: List[Experiment],
    skip_existing: bool = True,
    **kwargs: Any,
) -> List[ExperimentResult]:
    """Run multiple experiments with proper resource management.

    Args:
        experiments: List of experiments to run
        skip_existing: Skip experiments with existing results
        **kwargs: Arguments passed to each experiment's run()

    Returns:
        List of ExperimentResults
    """
    results = []

    for i, exp in enumerate(experiments, 1):
        logger.info("[%d/%d] %s", i, len(experiments), exp.name)

        if skip_existing and (exp.output_dir / exp.name / "results.json").exists():
            logger.info("Skipping %s (exists)", exp.name)
            # Load existing result
            with open(exp.output_dir / exp.name / "results.json") as f:
                data = json.load(f)
            results.append(
                ExperimentResult(
                    name=exp.name,
                    metrics=data.get("metrics", {}),
                    num_examples=data.get("num_examples", 0),
                    duration_seconds=data.get("duration_seconds", 0),
                    output_path=exp.output_dir / exp.name,
                )
            )
            continue

        try:
            result = exp.run(**kwargs)
            results.append(result)
        except Exception as e:
            logger.error("Experiment %s failed: %s", exp.name, e)
            results.append(
                ExperimentResult(
                    name=exp.name,
                    metrics={},
                    num_examples=0,
                    duration_seconds=0,
                    metadata={"error": str(e)},
                )
            )

        ResourceManager.clear_gpu_memory()

    return results
