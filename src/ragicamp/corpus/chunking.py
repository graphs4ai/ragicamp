"""Document chunking strategies for RAG retrieval.

Chunking long documents improves retrieval quality by:
1. Better semantic matching (smaller chunks = more precise vectors)
2. Fitting context windows (avoid exceeding LLM limits)
3. Reducing noise (irrelevant sections don't dilute retrieval)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from ragicamp.retrievers.base import Document


@dataclass
class ChunkConfig:
    """Configuration for document chunking.

    Attributes:
        strategy: Chunking strategy ('fixed', 'sentence', 'paragraph', 'recursive')
        chunk_size: Target size in characters (or sentences for 'sentence' strategy)
        chunk_overlap: Overlap between consecutive chunks (prevents context loss)
        min_chunk_size: Minimum chunk size (discard smaller chunks)
        separators: Custom separators for recursive strategy
    """

    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    separators: Optional[List[str]] = None

    def __post_init__(self):
        if self.separators is None:
            # Default separators for recursive splitting (most to least specific)
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


class ChunkingStrategy(ABC):
    """Base class for document chunking strategies."""

    def __init__(self, config: ChunkConfig):
        self.config = config

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        pass

    def chunk_document(self, document: Document) -> Iterator[Document]:
        """Chunk a document, preserving metadata.

        Args:
            document: Input document

        Yields:
            Document objects for each chunk, with updated metadata
        """
        chunks = self.chunk(document.text)

        for i, chunk_text in enumerate(chunks):
            # Skip chunks that are too small
            if len(chunk_text.strip()) < self.config.min_chunk_size:
                continue

            yield Document(
                id=f"{document.id}_chunk_{i}",
                text=chunk_text.strip(),
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "parent_doc_id": document.id,
                    "chunk_strategy": self.config.strategy,
                },
            )


class FixedSizeChunker(ChunkingStrategy):
    """Split text into fixed-size character chunks with overlap.

    Simple and predictable. Best for uniform document processing.
    """

    def chunk(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)

            # Try to end at a word boundary
            if end < text_len:
                # Look back for a space
                space_pos = text.rfind(" ", start, end)
                if space_pos > start + self.config.chunk_size // 2:
                    end = space_pos

            chunks.append(text[start:end])

            # Move start, accounting for overlap
            start = end - self.config.chunk_overlap
            if start >= text_len or start == end - self.config.chunk_overlap:
                start = end  # Avoid infinite loop

        return chunks


class SentenceChunker(ChunkingStrategy):
    """Split text by sentences, grouping into chunks of N sentences.

    Better semantic boundaries than fixed-size. chunk_size = number of sentences.
    """

    # Sentence boundary pattern
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def chunk(self, text: str) -> List[str]:
        """Split text into sentence-based chunks."""
        # Split into sentences
        sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        sentences_per_chunk = max(1, self.config.chunk_size)  # chunk_size = sentences
        overlap_sentences = max(0, self.config.chunk_overlap)

        i = 0
        while i < len(sentences):
            end = min(i + sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end]
            chunks.append(" ".join(chunk_sentences))

            # Move forward with overlap
            i = end - overlap_sentences
            if i <= end - sentences_per_chunk:  # Prevent infinite loop
                i = end

        return chunks


class ParagraphChunker(ChunkingStrategy):
    """Split text by paragraphs (double newlines).

    Respects natural document structure. May produce variable chunk sizes.
    """

    def chunk(self, text: str) -> List[str]:
        """Split text into paragraph-based chunks."""
        # Split by double newlines
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        # Group paragraphs until we hit chunk_size
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # If this paragraph alone exceeds chunk_size, split it
            if para_size > self.config.chunk_size and not current_chunk:
                # Use recursive splitter for this large paragraph
                sub_chunks = self._split_large_paragraph(para)
                chunks.extend(sub_chunks)
                continue

            # If adding this paragraph exceeds chunk_size, start new chunk
            if current_size + para_size > self.config.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Overlap: keep last paragraph(s) up to overlap size
                overlap_text = current_chunk[-1] if current_chunk else ""
                if len(overlap_text) <= self.config.chunk_overlap:
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_large_paragraph(self, text: str) -> List[str]:
        """Split a large paragraph using sentence boundaries."""
        sentence_chunker = SentenceChunker(
            ChunkConfig(
                strategy="sentence",
                chunk_size=5,  # sentences per chunk
                chunk_overlap=1,
                min_chunk_size=self.config.min_chunk_size,
            )
        )
        return sentence_chunker.chunk(text)


class RecursiveChunker(ChunkingStrategy):
    """Recursively split text using a hierarchy of separators.

    The most sophisticated strategy. Tries to preserve semantic structure
    by splitting on the most specific separator first, then falling back
    to less specific ones.

    Default separator hierarchy:
    1. Paragraph breaks (\\n\\n)
    2. Line breaks (\\n)
    3. Sentence endings (. ! ?)
    4. Clause separators (; ,)
    5. Spaces

    This is similar to LangChain's RecursiveCharacterTextSplitter.
    """

    def chunk(self, text: str) -> List[str]:
        """Recursively split text into chunks."""
        return self._recursive_split(text, self.config.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separator hierarchy."""
        if not text.strip():
            return []

        # Base case: text fits in chunk
        if len(text) <= self.config.chunk_size:
            return [text]

        # Try each separator in order
        for i, sep in enumerate(separators):
            if sep in text:
                splits = text.split(sep)

                # Build chunks from splits
                chunks = []
                current_chunk = ""

                for split in splits:
                    test_chunk = (current_chunk + sep + split).strip() if current_chunk else split

                    if len(test_chunk) <= self.config.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if it exists
                        if current_chunk:
                            chunks.append(current_chunk)

                        # If this split alone is too big, recurse with next separator
                        if len(split) > self.config.chunk_size:
                            remaining_seps = (
                                separators[i + 1 :] if i + 1 < len(separators) else [" "]
                            )
                            sub_chunks = self._recursive_split(split, remaining_seps)
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split

                # Don't forget last chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # Apply overlap between chunks
                return self._apply_overlap(chunks)

        # Fallback: hard split at chunk_size
        return self._hard_split(text)

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if self.config.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap = (
                prev_chunk[-self.config.chunk_overlap :]
                if len(prev_chunk) > self.config.chunk_overlap
                else prev_chunk
            )

            # Prepend overlap to current chunk (if it doesn't already contain it)
            if not curr_chunk.startswith(overlap):
                # Find a word boundary in overlap
                space_idx = overlap.find(" ")
                if space_idx > 0:
                    overlap = overlap[space_idx + 1 :]
                result.append(overlap + " " + curr_chunk)
            else:
                result.append(curr_chunk)

        return result

    def _hard_split(self, text: str) -> List[str]:
        """Last resort: split at exact chunk_size boundaries."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.config.chunk_overlap if self.config.chunk_overlap > 0 else end
        return chunks


# Factory function
def get_chunker(config: ChunkConfig) -> ChunkingStrategy:
    """Get a chunking strategy based on config.

    Args:
        config: Chunking configuration

    Returns:
        Appropriate ChunkingStrategy instance

    Raises:
        ValueError: If strategy is unknown
    """
    strategies = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "recursive": RecursiveChunker,
    }

    if config.strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: {config.strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[config.strategy](config)


def _chunk_single_document(args: tuple) -> List[dict]:
    """Worker function for parallel chunking.
    
    Returns dicts instead of Document objects to avoid pickling issues.
    """
    doc_dict, config_dict = args
    
    # Reconstruct config from dict
    config = ChunkConfig(
        strategy=config_dict["strategy"],
        chunk_size=config_dict["chunk_size"],
        chunk_overlap=config_dict["chunk_overlap"],
        min_chunk_size=config_dict["min_chunk_size"],
        separators=config_dict.get("separators"),
    )
    
    # Reconstruct document from dict
    from ragicamp.retrievers.base import Document
    doc = Document(
        id=doc_dict["id"],
        text=doc_dict["text"],
        metadata=doc_dict["metadata"],
    )
    
    strategy = get_chunker(config)
    chunks = list(strategy.chunk_document(doc))
    
    # Return as dicts to avoid pickling issues
    return [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]


class DocumentChunker:
    """High-level interface for chunking documents.

    Example:
        >>> config = ChunkConfig(strategy="recursive", chunk_size=512, chunk_overlap=50)
        >>> chunker = DocumentChunker(config)
        >>> for chunk_doc in chunker.chunk_documents(documents):
        ...     print(f"Chunk {chunk_doc.id}: {len(chunk_doc.text)} chars")
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize document chunker.

        Args:
            config: Chunking configuration. Uses defaults if None.
        """
        self.config = config or ChunkConfig()
        self.strategy = get_chunker(self.config)

    def chunk_documents(
        self,
        documents: Iterator[Document],
        show_progress: bool = True,
    ) -> Iterator[Document]:
        """Chunk multiple documents (sequential).

        Args:
            documents: Input documents to chunk
            show_progress: Whether to show progress (via prints)

        Yields:
            Chunked Document objects
        """
        total_chunks = 0
        doc_count = 0

        for doc in documents:
            doc_count += 1
            doc_chunks = list(self.strategy.chunk_document(doc))
            total_chunks += len(doc_chunks)

            for chunk_doc in doc_chunks:
                yield chunk_doc

            if show_progress and doc_count % 100 == 0:
                print(f"  Processed {doc_count} docs → {total_chunks} chunks")

        if show_progress:
            print(f"✓ Chunking complete: {doc_count} docs → {total_chunks} chunks")
            print(f"  Avg chunks per doc: {total_chunks / max(doc_count, 1):.1f}")

    def chunk_documents_parallel(
        self,
        documents: List[Document],
        num_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Document]:
        """Chunk multiple documents in parallel using multiprocessing.

        Args:
            documents: List of documents to chunk (must be a list, not iterator)
            num_workers: Number of worker processes (default: CPU count)
            show_progress: Whether to show progress

        Returns:
            List of chunked Document objects
        """
        import multiprocessing as mp
        import time

        from tqdm import tqdm

        if num_workers is None:
            num_workers = mp.cpu_count() or 4

        doc_count = len(documents)
        if show_progress:
            print(f"    Chunking {doc_count} docs with {num_workers} workers...")

        # Convert to dicts for pickling (dataclasses can have issues)
        t0 = time.time()
        config_dict = {
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
            "separators": self.config.separators,
        }
        
        args = [
            ({"id": doc.id, "text": doc.text, "metadata": doc.metadata}, config_dict)
            for doc in documents
        ]
        if show_progress:
            print(f"      [Prep: {time.time() - t0:.1f}s]")

        # For small batches, sequential is faster (IPC overhead > processing time)
        # Threshold: ~1000 docs/worker makes IPC worthwhile
        min_docs_for_parallel = num_workers * 1000
        
        t1 = time.time()
        all_chunk_dicts = []
        
        if doc_count < min_docs_for_parallel:
            # Sequential processing - faster for small batches
            if show_progress:
                print(f"      [Sequential mode - {doc_count} docs < {min_docs_for_parallel} threshold]", flush=True)
            for doc_dict, cfg in tqdm(args, desc="      Chunking", disable=not show_progress):
                chunk_dicts = _chunk_single_document((doc_dict, cfg))
                all_chunk_dicts.extend(chunk_dicts)
        else:
            # Parallel processing - worth the IPC overhead for large batches
            docs_per_worker = max(1, doc_count // num_workers)
            if show_progress:
                print(f"      [Parallel: {num_workers} workers, {docs_per_worker} docs/worker]", flush=True)
            
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_chunk_single_document, args, chunksize=docs_per_worker)
                
                for chunk_dicts in results:
                    all_chunk_dicts.extend(chunk_dicts)
        
        if show_progress:
            print(f"      [Pool: {time.time() - t1:.1f}s]")

        # Convert dicts back to Document objects
        t2 = time.time()
        all_chunks = [
            Document(id=cd["id"], text=cd["text"], metadata=cd["metadata"])
            for cd in all_chunk_dicts
        ]
        if show_progress:
            print(f"      [Convert: {time.time() - t2:.1f}s]")

        if show_progress:
            avg = len(all_chunks) / max(doc_count, 1)
            print(f"    ✓ {doc_count} docs → {len(all_chunks)} chunks (avg: {avg:.1f})")

        return all_chunks

    def get_info(self) -> Dict[str, Any]:
        """Get chunking configuration info."""
        return {
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
        }
