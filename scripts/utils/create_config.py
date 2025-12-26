#!/usr/bin/env python3
"""CLI tool for creating RAGiCamp configuration templates.

Usage:
    python create_config.py my_experiment.yaml
    python create_config.py my_experiment.yaml --type rag
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_baseline_template():
    """Create a baseline (direct LLM) template."""
    return {
        "agent": {
            "type": "direct_llm",
            "name": "my_baseline",
            "system_prompt": "You are a helpful AI assistant.",
        },
        "model": {
            "type": "huggingface",
            "model_name": "google/flan-t5-small",
            "device": "cuda",
            "load_in_8bit": False,
        },
        "dataset": {
            "name": "natural_questions",
            "split": "validation",
            "num_examples": 10,
            "filter_no_answer": True,
        },
        "evaluation": {
            "batch_size": 4,
        },
        "metrics": [
            "exact_match",
            "f1",
        ],
        "output": {
            "save_predictions": True,
            "output_path": "outputs/my_experiment.json",
        },
    }


def create_rag_template():
    """Create a RAG template."""
    return {
        "agent": {
            "type": "fixed_rag",
            "name": "my_rag",
            "system_prompt": "Use the provided context to answer questions.",
            "top_k": 5,
        },
        "model": {
            "type": "huggingface",
            "model_name": "google/flan-t5-small",
            "device": "cuda",
            "load_in_8bit": False,
        },
        "retriever": {
            "type": "dense",
            "name": "my_retriever",
            "embedding_model": "all-MiniLM-L6-v2",
            "artifact_path": "artifacts/retrievers/my_corpus",
        },
        "dataset": {
            "name": "natural_questions",
            "split": "validation",
            "num_examples": 10,
            "filter_no_answer": True,
        },
        "evaluation": {
            "batch_size": 4,
        },
        "metrics": [
            "exact_match",
            "f1",
            "bertscore",
        ],
        "output": {
            "save_predictions": True,
            "output_path": "outputs/my_rag_experiment.json",
        },
    }


def create_llm_judge_template():
    """Create a template with LLM judge."""
    return {
        "agent": {
            "type": "direct_llm",
            "name": "my_baseline_with_judge",
            "system_prompt": "You are a helpful AI assistant.",
        },
        "model": {
            "type": "huggingface",
            "model_name": "google/flan-t5-small",
            "device": "cuda",
        },
        "judge_model": {
            "type": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.0,
        },
        "dataset": {
            "name": "natural_questions",
            "split": "validation",
            "num_examples": 20,
        },
        "evaluation": {
            "batch_size": 8,
        },
        "metrics": [
            "exact_match",
            "f1",
            {
                "name": "llm_judge_qa",
                "params": {
                    "judgment_type": "binary",
                    "batch_size": 16,
                },
            },
        ],
        "output": {
            "save_predictions": True,
            "output_path": "outputs/my_experiment_with_judge.json",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create RAGiCamp configuration templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("output", help="Output path for configuration file")

    parser.add_argument(
        "--type",
        choices=["baseline", "rag", "llm_judge"],
        default="baseline",
        help="Template type (default: baseline)",
    )

    args = parser.parse_args()

    # Create appropriate template
    if args.type == "baseline":
        template = create_baseline_template()
    elif args.type == "rag":
        template = create_rag_template()
    elif args.type == "llm_judge":
        template = create_llm_judge_template()

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created {args.type} template: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Edit the config: {output_path}")
    print(f"  2. Validate: python scripts/validate_config.py {output_path}")
    print(f"  3. Run: make eval (with CONFIG={output_path})")


if __name__ == "__main__":
    main()
