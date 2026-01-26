"""Semantic chunking using embedding similarity.

Semantic chunking splits text at points where the meaning changes,
rather than at fixed character/token boundaries. This preserves
semantic coherence within chunks, improving retrieval quality.

The algorithm:
1. Split text into sentences
2. Compute embeddings for each sentence
3. Find breakpoints where cosine similarity between adjacent sentences
   drops below a threshold
4. Group sentences between breakpoints into chunks
"""

import re
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ragicamp.core.logging import get_logger
from ragicamp.corpus.chunking import ChunkConfig, ChunkingStrategy

logger = get_logger(__name__)


class SemanticChunker(ChunkingStrategy):
    """Split text at semantic boundaries using embedding similarity.

    This chunker uses an embedding model to detect where the topic
    or meaning changes in the text. It's more sophisticated than
    fixed-size chunking and produces more coherent chunks.
    """

    # Pattern for splitting into sentences
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        min_sentences_per_chunk: int = 2,
    ):
        """Initialize semantic chunker.

        Args:
            embedding_model: Sentence transformer model for embeddings
            similarity_threshold: Cosine similarity below which to split (0-1)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            min_sentences_per_chunk: Minimum sentences before allowing a split
        """
        # Create a basic config for parent class
        config = ChunkConfig(
            strategy="semantic",
            chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )
        super().__init__(config)

        self.embedding_model_name = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk

    def chunk(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks split at semantic boundaries
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Compute embeddings for all sentences
        embeddings = self.encoder.encode(sentences, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Find breakpoints based on similarity
        breakpoints = self._find_breakpoints(embeddings, sentences)

        # Build chunks from breakpoints
        chunks = self._build_chunks(sentences, breakpoints)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Use regex to split on sentence boundaries
        sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Handle case where regex doesn't split well
        if len(sentences) <= 1 and len(text) > self.max_chunk_size:
            # Fall back to splitting on newlines
            sentences = [s.strip() for s in text.split("\n") if s.strip()]

        return sentences

    def _find_breakpoints(
        self,
        embeddings: np.ndarray,
        sentences: List[str],
    ) -> List[int]:
        """Find indices where semantic breaks occur.

        Args:
            embeddings: Sentence embeddings (normalized)
            sentences: List of sentences

        Returns:
            List of indices where chunks should start
        """
        breakpoints = [0]  # First chunk always starts at 0

        current_chunk_start = 0
        current_chunk_size = 0

        for i in range(1, len(embeddings)):
            # Compute similarity with previous sentence
            similarity = float(np.dot(embeddings[i - 1], embeddings[i]))

            # Update chunk size
            current_chunk_size += len(sentences[i - 1])

            # Check if we should break
            should_break = False

            # Break if similarity is below threshold and we have enough sentences
            sentences_in_chunk = i - current_chunk_start
            if similarity < self.similarity_threshold:
                if sentences_in_chunk >= self.min_sentences_per_chunk:
                    should_break = True

            # Also break if chunk is getting too large
            if current_chunk_size >= self.max_chunk_size:
                should_break = True

            if should_break:
                breakpoints.append(i)
                current_chunk_start = i
                current_chunk_size = 0

        return breakpoints

    def _build_chunks(
        self,
        sentences: List[str],
        breakpoints: List[int],
    ) -> List[str]:
        """Build chunks from sentences and breakpoints.

        Args:
            sentences: List of sentences
            breakpoints: List of indices where chunks start

        Returns:
            List of text chunks
        """
        chunks = []

        for i in range(len(breakpoints)):
            start = breakpoints[i]
            end = breakpoints[i + 1] if i + 1 < len(breakpoints) else len(sentences)

            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)

            # Skip chunks that are too small
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def __repr__(self) -> str:
        return (
            f"SemanticChunker(model='{self.embedding_model_name}', "
            f"threshold={self.similarity_threshold})"
        )
