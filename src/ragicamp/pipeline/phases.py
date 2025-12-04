"""Experiment phases - modular building blocks for pipelines.

Each phase is a self-contained unit that:
1. Takes inputs
2. Produces outputs
3. Manages its own resources

Phases can be composed to build complete experiments.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ragicamp.pipeline.resource_manager import ResourceManager, managed_model


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
    2. Generates predictions for all questions
    3. Saves predictions to disk
    4. Unloads model to free memory
    """
    
    name = "generation"
    
    def __init__(
        self,
        model_factory,
        agent_factory,
        retriever=None,
        batch_size: int = 1,
    ):
        """Initialize generation phase.
        
        Args:
            model_factory: Callable that creates the model
            agent_factory: Callable(model, retriever) that creates the agent
            retriever: Optional retriever for RAG agents
            batch_size: Batch size for generation
        """
        self.model_factory = model_factory
        self.agent_factory = agent_factory
        self.retriever = retriever
        self.batch_size = batch_size
    
    def run(self, inputs: Dict[str, Any]) -> PhaseResult:
        """Generate predictions for the dataset.
        
        Args:
            inputs: Must contain 'dataset' and 'output_path'
            
        Returns:
            PhaseResult with predictions
        """
        dataset = inputs["dataset"]
        output_path = inputs.get("output_path", "outputs/predictions.json")
        
        print(f"\n{'='*60}")
        print(f"PHASE: {self.name.upper()}")
        print(f"{'='*60}")
        
        predictions = []
        
        # Use context manager for automatic cleanup
        with managed_model(self.model_factory, "LLM") as model:
            # Create agent with the model
            agent = self.agent_factory(model, self.retriever)
            
            print(f"\n📝 Generating predictions for {len(dataset)} examples...")
            
            examples = list(dataset)
            
            for example in tqdm(examples, desc="Generating"):
                response = agent.answer(example.question)
                predictions.append({
                    "question_id": example.id,
                    "question": example.question,
                    "prediction": response.answer,
                    "expected_answers": example.answers,
                    "metadata": response.metadata if hasattr(response, "metadata") else {},
                })
        
        # Model is automatically unloaded here
        
        # Save predictions
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_data = {
            "agent_name": agent.name,
            "dataset_name": dataset.name,
            "num_examples": len(predictions),
            "predictions": predictions,
        }
        
        with open(output_path, "w") as f:
            json.dump(predictions_data, f, indent=2)
        
        print(f"✓ Predictions saved to: {output_path}")
        
        return PhaseResult(
            success=True,
            data={"predictions": predictions, "predictions_data": predictions_data},
            output_path=output_path,
        )


class MetricsPhase(Phase):
    """Phase 2: Compute metrics on predictions.
    
    This phase:
    1. Loads predictions (from file or previous phase)
    2. Computes each metric one at a time
    3. Clears GPU memory between metrics
    4. Saves results
    """
    
    name = "metrics"
    
    def __init__(self, metrics: List[Any], judge_model=None):
        """Initialize metrics phase.
        
        Args:
            metrics: List of metric instances
            judge_model: Optional judge model for LLM metrics
        """
        self.metrics = metrics
        self.judge_model = judge_model
    
    def run(self, inputs: Dict[str, Any]) -> PhaseResult:
        """Compute metrics on predictions.
        
        Args:
            inputs: Must contain 'predictions' or 'predictions_file'
            
        Returns:
            PhaseResult with metric scores
        """
        print(f"\n{'='*60}")
        print(f"PHASE: {self.name.upper()}")
        print(f"{'='*60}")
        
        # Load predictions
        if "predictions_data" in inputs:
            predictions_data = inputs["predictions_data"]
        elif "predictions_file" in inputs:
            with open(inputs["predictions_file"], "r") as f:
                predictions_data = json.load(f)
        else:
            return PhaseResult(success=False, error="No predictions provided")
        
        # Extract data
        predictions = [p["prediction"] for p in predictions_data["predictions"]]
        references = [p["expected_answers"] for p in predictions_data["predictions"]]
        questions = [p["question"] for p in predictions_data["predictions"]]
        
        # Compute metrics one at a time (memory efficient)
        results = {}
        
        for metric in self.metrics:
            print(f"\n📊 Computing: {metric.name}")
            ResourceManager.print_memory_status(f"before {metric.name}")
            
            try:
                if metric.name in ["llm_judge", "llm_judge_qa"]:
                    scores = metric.compute(
                        predictions=predictions,
                        references=references,
                        questions=questions,
                    )
                else:
                    scores = metric.compute(predictions=predictions, references=references)
                
                results.update(scores)
                print(f"  ✓ {metric.name}: {scores}")
                
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
        
        return PhaseResult(success=True, data={"results": results})

