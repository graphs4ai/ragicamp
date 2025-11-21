#!/usr/bin/env python3
"""Download and cache QA datasets for RAGiCamp.

This script uses the dataset classes' built-in download_and_cache method
to download datasets from HuggingFace and save them locally.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.datasets.hotpotqa import HotpotQADataset
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.triviaqa import TriviaQADataset


def download_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Path,
    max_examples: int = None,
    filter_no_answer: bool = True,
    force_download: bool = False,
    show_stats: bool = True,
):
    """Download and cache a dataset.

    Args:
        dataset_name: Name of the dataset (natural_questions, triviaqa, hotpotqa)
        split: Dataset split (train/validation/test)
        cache_dir: Directory to cache the dataset
        max_examples: Optional limit on number of examples
        filter_no_answer: Whether to filter examples without answers
        force_download: Force re-download even if cache exists
        show_stats: Whether to show statistics
    """
    print(f"\n{'='*70}")
    print(f"📥 Downloading {dataset_name} ({split} split)")
    print(f"{'='*70}\n")

    # Select the appropriate dataset class
    if dataset_name == "natural_questions":
        dataset_class = NaturalQuestionsDataset
    elif dataset_name == "triviaqa":
        dataset_class = TriviaQADataset
    elif dataset_name == "hotpotqa":
        dataset_class = HotpotQADataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Download and cache
    dataset = dataset_class.download_and_cache(
        split=split,
        cache_dir=cache_dir,
        max_examples=max_examples,
        filter_no_answer=filter_no_answer,
        force_download=force_download,
    )

    # Show statistics
    if show_stats:
        print(f"\n{'='*70}")
        print("📊 Dataset Statistics")
        print(f"{'='*70}")
        print(f"Dataset:       {dataset.name}")
        print(f"Split:         {split}")
        print(f"Size:          {len(dataset):,} examples")
        print(f"Cache path:    {dataset.get_cache_path()}")

        if len(dataset) > 0:
            print(f"\nSample question:")
            example = dataset[0]
            print(f"  Q: {example.question}")
            print(f"  A: {example.answers}")

    return dataset


def main():
    """Main function to download datasets."""
    parser = argparse.ArgumentParser(
        description="Download and cache QA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Natural Questions validation set
  python download_datasets.py --dataset natural_questions --split validation
  
  # Download subset for testing
  python download_datasets.py --dataset natural_questions --max-examples 1000
  
  # Download all splits of Natural Questions
  python download_datasets.py --dataset natural_questions --all-splits
  
  # Force re-download
  python download_datasets.py --dataset natural_questions --force
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["natural_questions", "triviaqa", "hotpotqa", "all"],
        default="natural_questions",
        help="Dataset to download",
    )

    parser.add_argument(
        "--split", type=str, default="validation", help="Dataset split (train/validation/test)"
    )

    parser.add_argument("--all-splits", action="store_true", help="Download all available splits")

    parser.add_argument(
        "--max-examples", type=int, default=None, help="Maximum number of examples to download"
    )

    parser.add_argument(
        "--no-filter", action="store_true", help="Don't filter out questions without answers"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Output directory for cached datasets",
    )

    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if cache exists"
    )

    args = parser.parse_args()

    # Determine which datasets to download
    datasets_to_download = []
    if args.dataset == "all":
        datasets_to_download = ["natural_questions", "triviaqa", "hotpotqa"]
    else:
        datasets_to_download = [args.dataset]

    # Determine which splits to download
    splits_to_download = []
    if args.all_splits:
        for dataset_name in datasets_to_download:
            if dataset_name == "natural_questions":
                splits_to_download.append((dataset_name, "train"))
                splits_to_download.append((dataset_name, "validation"))
            elif dataset_name == "triviaqa":
                splits_to_download.append((dataset_name, "train"))
                splits_to_download.append((dataset_name, "validation"))
                splits_to_download.append((dataset_name, "test"))
            elif dataset_name == "hotpotqa":
                splits_to_download.append((dataset_name, "train"))
                splits_to_download.append((dataset_name, "validation"))
    else:
        for dataset_name in datasets_to_download:
            splits_to_download.append((dataset_name, args.split))

    # Download each dataset
    for dataset_name, split in splits_to_download:
        try:
            download_dataset(
                dataset_name=dataset_name,
                split=split,
                cache_dir=args.output_dir,
                max_examples=args.max_examples,
                filter_no_answer=not args.no_filter,
                force_download=args.force,
                show_stats=True,
            )
        except Exception as e:
            print(f"\n❌ Error downloading {dataset_name} ({split}): {e}")
            continue

    print(f"\n{'='*70}")
    print("✅ Download complete!")
    print(f"{'='*70}\n")
    print(f"Datasets saved to: {args.output_dir}")
    print(f"\nTo list downloaded datasets: make list-datasets")


if __name__ == "__main__":
    main()
