"""Wikipedia corpus implementation."""

from typing import Iterator, Optional

from datasets import load_dataset
from tqdm import tqdm

from ragicamp.corpus.base import CorpusConfig, DocumentCorpus
from ragicamp.retrievers.base import Document


class WikipediaCorpus(DocumentCorpus):
    """Wikipedia corpus using Wikimedia datasets.

    Loads Wikipedia articles from the wikimedia/wikipedia dataset on
    HuggingFace. Supports different language configurations.

    Common configurations:
    - "20231101.simple" - Simple English Wikipedia (~200k articles)
    - "20231101.en" - Full English Wikipedia (~6M articles)

    Example:
        >>> config = CorpusConfig(
        ...     name="wikipedia_simple",
        ...     source="wikimedia/wikipedia",
        ...     version="20231101.simple"
        ... )
        >>> corpus = WikipediaCorpus(config)
        >>> docs = list(corpus.load(max_docs=100))
        >>> print(f"Loaded {len(docs)} articles")
    """

    def __init__(self, config: CorpusConfig):
        """Initialize Wikipedia corpus.

        Args:
            config: Corpus configuration with:
                - source: Should be "wikimedia/wikipedia"
                - version: Wikipedia config (e.g., "20231101.simple")
        """
        super().__init__(config)
        self._dataset = None

    def load(self, max_docs: Optional[int] = None) -> Iterator[Document]:
        """Load Wikipedia articles.

        Args:
            max_docs: Maximum number of documents to load.
                     Uses config.max_docs if not specified.

        Yields:
            Document objects containing Wikipedia article text.
            Each document has:
            - id: wiki_{index}
            - text: Full article text
            - metadata: {title, url, source}

        Note:
            First call will download the dataset if not cached.
            Uses streaming to avoid loading everything into memory.
        """
        # Determine document limit
        limit = max_docs if max_docs is not None else self.config.max_docs

        # Load dataset (streaming mode for memory efficiency)
        if self._dataset is None:
            try:
                self._dataset = load_dataset(
                    self.config.source,
                    self.config.version,
                    split="train",
                    streaming=True,
                    trust_remote_code=False,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Wikipedia corpus '{self.config.version}': {e}\n"
                    f"Available configs: '20231101.simple', '20231101.en', etc."
                ) from e

        # Yield documents
        seen_titles = set()
        min_chars = self.config.metadata.get("min_chars", 100)

        desc = f"Loading {self.config.name}"
        iterator = tqdm(enumerate(self._dataset), desc=desc, total=limit)

        for i, article in iterator:
            # Check limit
            if limit and i >= limit:
                break

            title = article.get("title", "")
            text = article.get("text", "")

            # Skip if too short or duplicate
            if len(text) < min_chars or title in seen_titles:
                continue

            seen_titles.add(title)

            # Create document WITHOUT answer information
            yield Document(
                id=f"wiki_{i}",
                text=text,
                metadata={
                    "title": title,
                    "url": article.get("url", ""),
                    "source": "wikipedia",
                    "corpus": self.config.name,
                },
            )

    def get_info(self) -> dict:
        """Get Wikipedia corpus metadata."""
        info = super().get_info()
        info.update(
            {
                "corpus_type": "wikipedia",
                "language": self._get_language_from_version(),
                "streaming": True,
            }
        )
        return info

    def _get_language_from_version(self) -> str:
        """Extract language code from version string."""
        # Format: "YYYYMMDD.lang" -> extract lang
        parts = self.config.version.split(".")
        return parts[1] if len(parts) > 1 else "unknown"
