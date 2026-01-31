"""Wikipedia corpus implementation."""

import heapq
from typing import Iterator, Optional, Set

from datasets import load_dataset
from tqdm import tqdm

from ragicamp.corpus.base import CorpusConfig, DocumentCorpus
from ragicamp.retrievers.base import Document


def _get_wikirank_cache_path(language: str) -> "Path":
    """Get path for cached WikiRank scores."""
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "ragicamp" / "wikirank"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{language}_scores.pkl"


def _load_wikirank_scores(language: str) -> dict:
    """Load WikiRank scores for a language, with caching.

    Returns:
        Dict mapping normalized titles to scores
    """
    import pickle

    import pandas as pd

    cache_path = _get_wikirank_cache_path(language)

    # Try cache first
    if cache_path.exists():
        print(f"  Loading cached WikiRank scores from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Download from HuggingFace
    url = f"https://huggingface.co/datasets/lewoniewski/wikipedia_quality_wikirank/resolve/main/languages/{language}.csv"

    try:
        print(f"  Downloading from: {url}")
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load WikiRank for language '{language}': {e}\n"
            f"Available languages: en, simple, de, fr, es, etc."
        ) from e

    print(f"  Loaded {len(df):,} article scores")

    # Identify columns - the dataset has: page_id, title, wikirank_score
    title_col = "title" if "title" in df.columns else df.columns[1]
    score_col = next(
        (c for c in ["wikirank_score", "quality", "score"] if c in df.columns),
        df.columns[-1],
    )

    print(f"  Using columns: title='{title_col}', score='{score_col}'")

    # Build title -> score map (normalize titles)
    scores = {}
    for _, row in df.iterrows():
        title = str(row[title_col]).replace("_", " ").strip().lower()
        scores[title] = float(row[score_col])

    # Cache for next time
    with open(cache_path, "wb") as f:
        pickle.dump(scores, f)
    print(f"  Cached scores to {cache_path}")

    return scores


def _load_wikirank_top_titles(top_k: int, language: str = "en") -> Set[str]:
    """Load top-K article titles from WikiRank dataset.

    WikiRank combines pageviews, edit frequency, references, and link structure
    into a quality/popularity score. Higher scores = more important articles.

    Dataset: lewoniewski/wikipedia_quality_wikirank
    See: https://huggingface.co/datasets/lewoniewski/wikipedia_quality_wikirank

    Args:
        top_k: Number of top articles to keep
        language: Wikipedia language code (e.g., "en", "simple", "de")

    Returns:
        Set of normalized titles (lowercase, spaces not underscores)
    """
    print(f"Loading WikiRank scores for top {top_k:,} {language} articles...")

    scores = _load_wikirank_scores(language)

    # Find threshold for top-K
    if top_k >= len(scores):
        print(f"  Requested {top_k:,} but only {len(scores):,} available, using all")
        return set(scores.keys())

    top_scores = heapq.nlargest(top_k, scores.values())
    threshold = top_scores[-1] if top_scores else 0.0

    # Build allowed set
    allowed = {t for t, s in scores.items() if s >= threshold}
    print(f"  Filter threshold: {threshold:.6f}, passing: {len(allowed):,}")

    return allowed


def _normalize_title(title: str) -> str:
    """Normalize title for WikiRank matching."""
    return title.replace("_", " ").strip().lower()


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
                - metadata.wikirank_top_k: Optional, filter to top K important articles
        """
        super().__init__(config)
        self._dataset = None
        self._allowed_titles: Optional[Set[str]] = None

        # Load WikiRank filter if configured
        top_k = self.config.metadata.get("wikirank_top_k")
        if top_k:
            # Extract language from version (e.g., "20231101.en" -> "en")
            language = self._get_language_from_version()
            self._allowed_titles = _load_wikirank_top_titles(int(top_k), language)

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

            # WikiRank filter: skip articles not in top-K
            if self._allowed_titles is not None:
                if _normalize_title(title) not in self._allowed_titles:
                    continue

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
                "wikirank_filter": self._allowed_titles is not None,
                "wikirank_top_k": self.config.metadata.get("wikirank_top_k"),
            }
        )
        return info

    def _get_language_from_version(self) -> str:
        """Extract language code from version string."""
        # Format: "YYYYMMDD.lang" -> extract lang
        parts = self.config.version.split(".")
        return parts[1] if len(parts) > 1 else "unknown"
