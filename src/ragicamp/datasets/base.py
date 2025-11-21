"""Base classes for datasets."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QAExample:
    """Represents a question-answering example.

    Attributes:
        id: Unique example identifier
        question: The question text
        answers: List of acceptable answers
        context: Optional context/passage (for reading comprehension)
        metadata: Additional example metadata
    """

    id: str
    question: str
    answers: List[str]
    context: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QADataset(ABC):
    """Base class for QA datasets.

    Provides a unified interface for loading and accessing different
    QA datasets (NQ, HotpotQA, TriviaQA, etc.).
    """

    def __init__(
        self, name: str, split: str = "train", cache_dir: Optional[Path] = None, **kwargs: Any
    ):
        """Initialize the dataset.

        Args:
            name: Dataset identifier
            split: Dataset split (train/validation/test)
            cache_dir: Optional directory to cache processed datasets
            **kwargs: Dataset-specific configuration
        """
        self.name = name
        self.split = split
        self.config = kwargs
        self.cache_dir = cache_dir or Path("data/datasets")
        self.examples: List[QAExample] = []

    @abstractmethod
    def load(self) -> None:
        """Load the dataset from source."""
        pass

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> QAExample:
        """Get example by index."""
        return self.examples[idx]

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    def get_subset(self, n: int, seed: Optional[int] = None) -> List[QAExample]:
        """Get a random subset of examples.

        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled examples
        """
        import random

        if seed is not None:
            random.seed(seed)

        n = min(n, len(self.examples))
        return random.sample(self.examples, n)

    def filter_with_answers(self) -> None:
        """Filter dataset to only include examples with explicit answers.

        Removes examples where answers list is empty or contains only empty strings.
        Updates self.examples in-place.
        """
        original_count = len(self.examples)
        self.examples = [
            ex
            for ex in self.examples
            if ex.answers and any(answer.strip() for answer in ex.answers)
        ]
        filtered_count = original_count - len(self.examples)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} examples without explicit answers")
            print(f"Remaining: {len(self.examples)} examples")

    def get_examples_with_answers(self, n: Optional[int] = None) -> List[QAExample]:
        """Get examples that have explicit answers.

        Args:
            n: Optional maximum number of examples to return

        Returns:
            List of examples with non-empty answers
        """
        filtered = [
            ex
            for ex in self.examples
            if ex.answers and any(answer.strip() for answer in ex.answers)
        ]

        if n is not None:
            filtered = filtered[:n]

        return filtered

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', split='{self.split}', size={len(self)})"

    def get_cache_path(self) -> Path:
        """Get the path where this dataset should be cached.

        Child classes can override this to include dataset-specific parameters
        in the cache path (e.g., subset, distractor settings).

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{self.name}_{self.split}.json"

    def save_to_cache(self, info: Optional[Dict[str, Any]] = None) -> Path:
        """Save dataset to cache file.

        Args:
            info: Optional metadata to save with the dataset

        Returns:
            Path where dataset was saved
        """
        cache_path = self.get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert examples to dicts
        examples_data = [
            {
                "id": ex.id,
                "question": ex.question,
                "answers": ex.answers,
                "context": ex.context,
                "metadata": ex.metadata,
            }
            for ex in self.examples
        ]

        # Prepare data to save
        data = {
            "info": info
            or {"dataset": self.name, "split": self.split, "size": len(self.examples)},
            "examples": examples_data,
        }

        # Save to JSON
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

        return cache_path

    def load_from_cache(self) -> bool:
        """Load dataset from cache if available.

        Returns:
            True if loaded from cache, False otherwise
        """
        cache_path = self.get_cache_path()

        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            # Load examples
            self.examples = [
                QAExample(
                    id=ex["id"],
                    question=ex["question"],
                    answers=ex["answers"],
                    context=ex.get("context"),
                    metadata=ex.get("metadata", {}),
                )
                for ex in data["examples"]
            ]

            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            return False

    @classmethod
    def download_and_cache(
        cls,
        split: str = "validation",
        cache_dir: Optional[Path] = None,
        max_examples: Optional[int] = None,
        filter_no_answer: bool = True,
        force_download: bool = False,
        **kwargs: Any,
    ) -> "QADataset":
        """Download dataset and save to cache.

        This is a convenience method that:
        1. Creates a dataset instance
        2. Loads from HuggingFace (or cache if available)
        3. Applies filtering
        4. Saves to cache for future use

        Args:
            split: Dataset split to download
            cache_dir: Directory to cache the dataset
            max_examples: Optional limit on number of examples
            filter_no_answer: Whether to filter examples without answers
            force_download: Force re-download even if cache exists
            **kwargs: Additional dataset-specific arguments

        Returns:
            Loaded dataset instance
        """
        # Create instance (will auto-load)
        dataset = cls(split=split, cache_dir=cache_dir, **kwargs)

        # Check cache first unless force_download
        if not force_download and dataset.load_from_cache():
            print(f"✓ Loaded from cache: {dataset.get_cache_path()}")
            print(f"  {len(dataset)} examples")

            # Apply max_examples if specified
            if max_examples and len(dataset) > max_examples:
                dataset.examples = dataset.examples[:max_examples]

            return dataset

        # Load from source (already done in __init__)
        print(f"Loaded {len(dataset)} examples from HuggingFace")

        # Filter if requested
        if filter_no_answer:
            original_size = len(dataset)
            dataset.filter_with_answers()
            print(f"Filtered: {original_size} → {len(dataset)} examples")

        # Limit if requested
        if max_examples and len(dataset) > max_examples:
            dataset.examples = dataset.examples[:max_examples]
            print(f"Limited to {len(dataset)} examples")

        # Save to cache
        info = {
            "dataset": dataset.name,
            "split": split,
            "original_size": len(dataset),
            "filtered_size": len(dataset),
            "filter_no_answer": filter_no_answer,
        }
        cache_path = dataset.save_to_cache(info)
        print(f"✓ Saved to cache: {cache_path}")

        return dataset
