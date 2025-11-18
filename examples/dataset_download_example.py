#!/usr/bin/env python3
"""Example of using the dataset download functionality."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragicamp.datasets import NaturalQuestionsDataset


def main():
    """Demonstrate dataset download and caching."""
    
    print("=" * 70)
    print("Example 1: Download and cache Natural Questions")
    print("=" * 70)
    
    # Download and cache the dataset
    dataset = NaturalQuestionsDataset.download_and_cache(
        split="validation",
        max_examples=100,  # Limit to 100 for demo
        filter_no_answer=True,
        cache_dir=Path("data/datasets")
    )
    
    print(f"\n✓ Downloaded {len(dataset)} examples")
    print(f"  Cache location: {dataset.get_cache_path()}")
    
    print("\n" + "=" * 70)
    print("Example 2: Load from cache (instant!)")
    print("=" * 70)
    
    # Second load will use cache (much faster)
    dataset2 = NaturalQuestionsDataset(
        split="validation",
        cache_dir=Path("data/datasets"),
        use_cache=True  # This is default
    )
    
    print(f"\n✓ Loaded {len(dataset2)} examples from cache")
    
    print("\n" + "=" * 70)
    print("Example 3: Inspect the data")
    print("=" * 70)
    
    # Look at first few examples
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Question: {example.question}")
        print(f"  Answers: {example.answers}")
    
    print("\n" + "=" * 70)
    print("Example 4: Use in experiments (automatic)")
    print("=" * 70)
    
    # In experiments, just use the dataset normally
    # It will automatically check cache first!
    dataset3 = NaturalQuestionsDataset(split="validation")
    print(f"\n✓ Loaded {len(dataset3)} examples (from cache if available)")
    
    # You can also force reload from HuggingFace
    print("\nTo force re-download:")
    print("  dataset = NaturalQuestionsDataset(split='validation', use_cache=False)")
    
    print("\n✅ Done! Your datasets are now cached and ready to use.")


if __name__ == "__main__":
    main()

