#!/usr/bin/env python3
"""Script to compare baseline agents."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.retrievers.dense import DenseRetriever


def main():
    """Compare baseline agents on NQ dataset."""
    print("=" * 60)
    print("RAGiCamp Baseline Comparison")
    print("=" * 60)

    # Create model
    print("\nLoading model...")
    model = HuggingFaceModel(
        model_name="google/flan-t5-small", device="cpu"  # Use small model for faster testing
    )

    # Create retriever
    print("Creating retriever...")
    retriever = DenseRetriever(name="demo_retriever", embedding_model="all-MiniLM-L6-v2")

    # Load dataset
    print("Loading dataset...")
    dataset = NaturalQuestionsDataset(split="validation")
    # Use small subset for demo
    dataset.examples = dataset.examples[:10]
    print(f"Using {len(dataset)} examples")

    # Create agents
    print("\nCreating agents...")
    agents = [
        DirectLLMAgent(
            name="Baseline_1_DirectLLM",
            model=model,
            system_prompt="Answer the following question concisely.",
        ),
        # Note: For FixedRAG to work, you need to index documents first
        # This is just a skeleton - will need actual document corpus
        # FixedRAGAgent(
        #     name="Baseline_2_FixedRAG",
        #     model=model,
        #     retriever=retriever,
        #     top_k=5
        # )
    ]

    # Create metrics
    metrics = [ExactMatchMetric(), F1Metric()]

    # Create evaluator
    evaluator = Evaluator(
        agent=agents[0], dataset=dataset, metrics=metrics  # Will be replaced in compare_agents
    )

    # Compare agents
    print("\n" + "=" * 60)
    print("Running Comparison")
    print("=" * 60)

    results = evaluator.compare_agents(agents)

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
