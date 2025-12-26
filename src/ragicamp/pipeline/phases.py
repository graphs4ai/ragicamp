"""Experiment phases - modular building blocks for pipelines.

Each phase is a self-contained unit that:
1. Takes inputs
2. Produces outputs
3. Manages its own resources
4. Supports checkpointing for resume capability

Phases can be composed to build complete experiments with automatic
checkpoint/resume support.

Example:
    >>> from ragicamp.pipeline import GenerationPhase, MetricsPhase
    >>> from ragicamp.checkpointing import ExperimentState
    >>>
    >>> # Create state for checkpointing
    >>> state = ExperimentState.load_or_create(...)
    >>>
    >>> # Generation with question-level checkpoints
    >>> gen_phase = GenerationPhase(
    ...     model_factory=lambda: HuggingFaceModel("gemma-2b"),
    ...     agent_factory=lambda m, r: DirectLLMAgent("test", m),
    ...     checkpoint_every=10,  # Save every 10 questions
    ... )
    >>> result = gen_phase.run({"dataset": ds, "experiment_state": state})
    >>>
    >>> # Metrics with per-metric checkpoints
    >>> metrics_phase = MetricsPhase(metrics=[...])
    >>> result = metrics_phase.run({"predictions_file": ..., "experiment_state": state})
"""

import asyncio
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from ragicamp.utils.resource_manager import ResourceManager, managed_model


@dataclass
class PhaseResult:
    """Result from a pipeline phase."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None
    error: Optional[str] = None


class Phase(ABC):
    """Base class for pipeline phases."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Phase name for logging."""
        pass

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> PhaseResult:
        """Execute the phase.

        Args:
            inputs: Input data from previous phases

        Returns:
            PhaseResult with outputs
        """
        pass


class GenerationPhase(Phase):
    """Phase 1: Generate predictions using an LLM agent.

    This phase:
    1. Loads the model and agent
    2. Generates predictions for all questions (with checkpointing)
    3. Saves predictions to disk
    4. Unloads model to free memory

    Supports resume from checkpoint if generation was interrupted.
    """

    name = "generation"

    def __init__(
        self,
        model_factory: Callable,
        agent_factory: Callable,
        retriever=None,
        batch_size: int = 1,
        checkpoint_every: int = 10,
    ):
        """Initialize generation phase.

        Args:
            model_factory: Callable that creates the model
            agent_factory: Callable(model, retriever) that creates the agent
            retriever: Optional retriever for RAG agents
            batch_size: Batch size for generation
            checkpoint_every: Save checkpoint every N questions (default: 10)
        """
        self.model_factory = model_factory
        self.agent_factory = agent_factory
        self.retriever = retriever
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every

    def run(self, inputs: Dict[str, Any]) -> PhaseResult:
        """Generate predictions for the dataset with checkpointing.

        Args:
            inputs: Must contain 'dataset' and 'output_path'
                   Optional: 'experiment_state' for checkpoint/resume

        Returns:
            PhaseResult with predictions
        """
        dataset = inputs["dataset"]
        output_path = inputs.get("output_path", "outputs/predictions.json")
        state = inputs.get("experiment_state")

        # Checkpoint file path
        checkpoint_path = Path(output_path).with_suffix(".checkpoint.json")

        print(f"\n{'='*60}")
        print(f"PHASE: {self.name.upper()}")
        print(f"{'='*60}")

        examples = list(dataset)
        total_examples = len(examples)

        # Check for resume
        predictions = []
        start_idx = 0

        if state and state.can_resume_phase(self.name):
            start_idx = state.get_checkpoint_idx(self.name)
            checkpoint_file = state.get_checkpoint_file(self.name)

            if checkpoint_file and Path(checkpoint_file).exists():
                print(f"📂 Resuming from checkpoint: {start_idx}/{total_examples}")
                with open(checkpoint_file, "r") as f:
                    checkpoint_data = json.load(f)
                    predictions = checkpoint_data.get("predictions", [])

        # Update state
        if state:
            state.start_phase(self.name, total_items=total_examples)

        agent_name = "unknown"

        # Use context manager for automatic cleanup
        with managed_model(self.model_factory, "LLM") as model:
            # Create agent with the model
            agent = self.agent_factory(model, self.retriever)
            agent_name = agent.name

            remaining = total_examples - start_idx
            print(
                f"\n📝 Generating predictions: {start_idx}/{total_examples} done, {remaining} remaining..."
            )

            for i, example in enumerate(
                tqdm(
                    examples[start_idx:], desc="Generating", initial=start_idx, total=total_examples
                )
            ):
                actual_idx = start_idx + i

                try:
                    response = agent.answer(example.question)
                    predictions.append(
                        {
                            "question_id": example.id,
                            "question": example.question,
                            "prediction": response.answer,
                            "expected_answers": example.answers,
                            "metadata": response.metadata if hasattr(response, "metadata") else {},
                        }
                    )
                except Exception as e:
                    print(f"\n⚠️  Error on question {actual_idx}: {e}")
                    predictions.append(
                        {
                            "question_id": example.id,
                            "question": example.question,
                            "prediction": f"[ERROR: {str(e)[:100]}]",
                            "expected_answers": example.answers,
                            "metadata": {"error": str(e)},
                        }
                    )

                # Save checkpoint
                if (actual_idx + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(
                        predictions,
                        checkpoint_path,
                        agent_name,
                        dataset.name,
                        actual_idx + 1,
                        total_examples,
                    )

                    if state:
                        state.update_phase_checkpoint(
                            self.name, actual_idx + 1, str(checkpoint_path)
                        )
                        state.save()

                    print(f"\n💾 Checkpoint saved: {actual_idx + 1}/{total_examples}")

        # Model is automatically unloaded here

        # Save final predictions
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_data = {
            "agent_name": agent_name,
            "dataset_name": dataset.name,
            "num_examples": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
        }

        # Atomic write
        temp_path = Path(output_path).with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(predictions_data, f, indent=2)
        shutil.move(str(temp_path), output_path)

        print(f"✓ Predictions saved to: {output_path}")

        # Update state
        if state:
            state.complete_phase(self.name, output_path=output_path)
            state.save()

        # Clean up checkpoint file
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return PhaseResult(
            success=True,
            data={"predictions": predictions, "predictions_data": predictions_data},
            output_path=output_path,
        )

    def _save_checkpoint(
        self,
        predictions: List[Dict],
        path: Path,
        agent_name: str,
        dataset_name: str,
        completed: int,
        total: int,
    ):
        """Save checkpoint atomically."""
        checkpoint_data = {
            "agent_name": agent_name,
            "dataset_name": dataset_name,
            "completed": completed,
            "total": total,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
        }

        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        shutil.move(str(temp_path), str(path))


class MetricsPhase(Phase):
    """Phase 2: Compute metrics on predictions.

    This phase:
    1. Loads predictions (from file or previous phase)
    2. Computes each metric one at a time
    3. Clears GPU memory between metrics
    4. Checkpoints after each metric
    5. Saves results

    Supports resume: if interrupted, will skip already-computed metrics.
    """

    name = "metrics"

    def __init__(
        self,
        metrics: List[Any],
        judge_model=None,
        output_path: Optional[str] = None,
    ):
        """Initialize metrics phase.

        Args:
            metrics: List of metric instances
            judge_model: Optional judge model for LLM metrics
            output_path: Optional path for results (defaults to input-derived)
        """
        self.metrics = metrics
        self.judge_model = judge_model
        self.output_path = output_path

    def run(self, inputs: Dict[str, Any]) -> PhaseResult:
        """Compute metrics on predictions with per-metric checkpointing.

        Args:
            inputs: Must contain 'predictions_data' or 'predictions_file'
                   Optional: 'experiment_state' for checkpoint/resume

        Returns:
            PhaseResult with metric scores
        """
        state = inputs.get("experiment_state")

        print(f"\n{'='*60}")
        print(f"PHASE: {self.name.upper()}")
        print(f"{'='*60}")

        # Load predictions
        if "predictions_data" in inputs:
            predictions_data = inputs["predictions_data"]
        elif "predictions_file" in inputs:
            with open(inputs["predictions_file"], "r") as f:
                predictions_data = json.load(f)
        elif "last_output_path" in inputs:
            with open(inputs["last_output_path"], "r") as f:
                predictions_data = json.load(f)
        else:
            return PhaseResult(success=False, error="No predictions provided")

        # Extract data
        predictions = [p["prediction"] for p in predictions_data["predictions"]]
        references = [p["expected_answers"] for p in predictions_data["predictions"]]
        questions = [p["question"] for p in predictions_data["predictions"]]

        # Determine which metrics to run
        all_metric_names = [m.name for m in self.metrics]

        if state:
            pending_metrics = state.get_pending_metrics(self.name, all_metric_names)
            completed_metrics = [m for m in all_metric_names if m not in pending_metrics]

            if completed_metrics:
                print(f"⏭️  Skipping completed metrics: {', '.join(completed_metrics)}")

            state.start_phase(self.name, total_items=len(all_metric_names))
        else:
            pending_metrics = all_metric_names

        # Compute metrics one at a time (memory efficient)
        results = {}

        for metric in self.metrics:
            if metric.name not in pending_metrics:
                continue

            print(f"\n📊 Computing: {metric.name}")
            ResourceManager.print_memory_status(f"before {metric.name}")

            try:
                # Check if metric supports async (AsyncAPIMetric)
                if hasattr(metric, "acompute"):
                    # Use async for API-based metrics
                    scores = asyncio.run(
                        metric.acompute(
                            predictions=predictions,
                            references=references,
                            questions=questions,
                        )
                    )
                elif metric.name in ["llm_judge", "llm_judge_qa"]:
                    scores = metric.compute(
                        predictions=predictions,
                        references=references,
                        questions=questions,
                    )
                else:
                    scores = metric.compute(predictions=predictions, references=references)

                results.update(scores)
                print(f"  ✓ {metric.name}: {scores}")

                # Mark metric as complete
                if state:
                    state.mark_metric_complete(self.name, metric.name)
                    state.save()

            except Exception as e:
                print(f"  ✗ {metric.name} failed: {e}")
                results[f"{metric.name}_error"] = str(e)

            # Clear memory after each metric
            ResourceManager.clear_gpu_memory()
            ResourceManager.print_memory_status(f"after {metric.name}")

        # Add metadata
        results["num_examples"] = len(predictions)
        results["agent_name"] = predictions_data.get("agent_name", "unknown")
        results["dataset_name"] = predictions_data.get("dataset_name", "unknown")

        # Save results
        output_path = self.output_path
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"✓ Results saved to: {output_path}")

        # Update state
        if state:
            state.complete_phase(self.name, output_path=output_path, results=results)
            state.save()

        return PhaseResult(success=True, data={"results": results}, output_path=output_path)
