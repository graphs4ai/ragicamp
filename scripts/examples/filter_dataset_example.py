#!/usr/bin/env python3
"""
Example: How to filter datasets to only include questions with explicit answers.

This demonstrates the new filtering capabilities added to handle datasets
where some questions don't have explicit ground-truth answers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.hotpotqa import HotpotQADataset


def example_1_filter_with_answers():
    """Example 1: Filter dataset in-place."""
    print("=" * 70)
    print("Example 1: Filter dataset in-place")
    print("=" * 70)
    
    # Load dataset
    dataset = NaturalQuestionsDataset(split="validation")
    print(f"\nOriginal dataset size: {len(dataset)}")
    
    # Filter to only questions with explicit answers
    dataset.filter_with_answers()
    print(f"After filtering: {len(dataset)}")
    
    # Show first few examples
    print("\nFirst 3 examples with answers:")
    for i, ex in enumerate(dataset.examples[:3]):
        print(f"\n{i+1}. Question: {ex.question}")
        print(f"   Expected answer: {ex.answers[0]}")
        print(f"   All acceptable: {ex.answers}")


def example_2_get_examples_with_answers():
    """Example 2: Get filtered list without modifying original."""
    print("\n" + "=" * 70)
    print("Example 2: Get filtered list (non-destructive)")
    print("=" * 70)
    
    # Load dataset
    dataset = HotpotQADataset(split="validation")
    print(f"\nOriginal dataset size: {len(dataset)}")
    
    # Get only examples with answers (doesn't modify original)
    filtered = dataset.get_examples_with_answers(n=10)
    print(f"Filtered examples (first 10): {len(filtered)}")
    print(f"Original dataset still: {len(dataset)}")
    
    # Show examples
    print("\nFirst 2 filtered examples:")
    for i, ex in enumerate(filtered[:2]):
        print(f"\n{i+1}. Question: {ex.question}")
        print(f"   Expected answer: {ex.answers[0] if ex.answers else 'N/A'}")


def example_3_check_answers():
    """Example 3: Check which questions have/don't have answers."""
    print("\n" + "=" * 70)
    print("Example 3: Analyze answer availability")
    print("=" * 70)
    
    # Load small sample
    dataset = NaturalQuestionsDataset(split="validation")
    dataset.examples = dataset.examples[:100]
    
    # Analyze
    with_answers = 0
    without_answers = 0
    
    for ex in dataset:
        if ex.answers and any(answer.strip() for answer in ex.answers):
            with_answers += 1
        else:
            without_answers += 1
    
    print(f"\nOut of {len(dataset)} examples:")
    print(f"  With explicit answers: {with_answers}")
    print(f"  Without explicit answers: {without_answers}")
    print(f"  Percentage with answers: {with_answers/len(dataset)*100:.1f}%")


if __name__ == "__main__":
    example_1_filter_with_answers()
    example_2_get_examples_with_answers()
    example_3_check_answers()
    
    print("\n" + "=" * 70)
    print("✓ Examples completed!")
    print("=" * 70)

