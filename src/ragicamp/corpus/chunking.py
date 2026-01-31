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


# ============================================================================
# Shared memory for parallel chunking (fork inherits these, no IPC needed)
# ============================================================================
_SHARED_DOCS: Optional[List[Dict]] = None
_SHARED_CONFIG: Optional[Dict] = None


def _get_available_cpus() -> int:
    """Get the number of CPUs available to this process.
    
    In containers, os.cpu_count() returns host CPUs, not container limits.
    This function checks cgroups and CPU affinity to get the actual limit.
    
    Returns:
        Number of usable CPUs (minimum 1)
    """
    import os
    
    # Method 1: CPU affinity (works on Linux, respects taskset/cgroups)
    try:
        cpus = len(os.sched_getaffinity(0))
        if cpus > 0:
            return cpus
    except (AttributeError, OSError):
        pass
    
    # Method 2: cgroups v2 (modern containers)
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            content = f.read().strip()
            if content != "max":
                quota, period = content.split()
                if quota != "max":
                    cpus = int(int(quota) / int(period))
                    if cpus > 0:
                        return cpus
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    
    # Method 3: cgroups v1 (older containers)
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            cpus = quota // period
            if cpus > 0:
                return cpus
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    
    # Fallback: os.cpu_count()
    return os.cpu_count() or 4


def _chunk_by_index(idx: int) -> List[Dict]:
    """Chunk a document by index from shared memory.
    
    This function is called by worker processes. With fork, workers inherit
    _SHARED_DOCS and _SHARED_CONFIG from the parent process (copy-on-write),
    so no pickling/IPC is needed for the actual data.
    """
    doc_dict = _SHARED_DOCS[idx]
    config_dict = _SHARED_CONFIG
    
    # Reconstruct config from dict
    config = ChunkConfig(
        strategy=config_dict["strategy"],
        chunk_size=config_dict["chunk_size"],
        chunk_overlap=config_dict["chunk_overlap"],
        min_chunk_size=config_dict["min_chunk_size"],
        separators=config_dict.get("separators"),
    )
    
    # Reconstruct document from dict
    doc = Document(
        id=doc_dict["id"],
        text=doc_dict["text"],
        metadata=doc_dict["metadata"],
    )
    
    strategy = get_chunker(config)
    chunks = list(strategy.chunk_document(doc))
    
    # Return as dicts
    return [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]


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

    # Debug mode for verbose output (set True to diagnose chunking issues)
    DEBUG = False
    
    def chunk_document(self, document: Document) -> Iterator[Document]:
        """Chunk a document, preserving metadata.

        Args:
            document: Input document

        Yields:
            Document objects for each chunk, with updated metadata
        """
        if self.DEBUG:
            title = document.metadata.get("title", document.id)[:50]
            print(f"      [chunk_document] '{title}' ({len(document.text):,} chars)", flush=True)
        
        chunks = self.chunk(document.text)
        
        if self.DEBUG:
            print(f"      [chunk_document] → {len(chunks)} chunks", flush=True)

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
    
    # Safety limits to prevent pathological cases
    MAX_RECURSION_DEPTH = 10  # Increased to allow deeper recursion
    MAX_CHUNKS_PER_DOC = 2000  # Increased for larger docs
    
    # Debug mode - set to True to see step-by-step chunking
    DEBUG = False

    def chunk(self, text: str) -> List[str]:
        """Recursively split text into chunks.
        
        Always tries recursive splitting first. Hard split is only used as
        a last resort when no separators work or recursion depth is exceeded.
        """
        if self.DEBUG:
            print(f"        [chunk] input: {len(text):,} chars", flush=True)
        
        # Always try recursive first - hard_split is only a fallback
        if self.DEBUG:
            print(f"        [chunk] using recursive_split", flush=True)
        result = self._recursive_split(text, self.config.separators, depth=0)
        if self.DEBUG:
            print(f"        [chunk] recursive_split → {len(result)} chunks", flush=True)
        return result

    def _recursive_split(
        self, text: str, separators: List[str], depth: int = 0
    ) -> List[str]:
        """Recursively split text using separator hierarchy."""
        if not text.strip():
            return []

        # Base case: text fits in chunk
        if len(text) <= self.config.chunk_size:
            return [text]
        
        # Safety: bail out if recursion is too deep
        if depth >= self.MAX_RECURSION_DEPTH:
            return self._hard_split(text)

        # Try each separator in order
        for i, sep in enumerate(separators):
            if sep in text:
                # Avoid creating huge lists - if separator is extremely frequent,
                # skip to the next one (but still try most separators)
                sep_count = text.count(sep)
                if sep_count > 50_000:
                    continue  # Skip this separator, try next one
                
                splits = text.split(sep)

                # Build chunks from splits
                chunks = []
                current_chunk = ""

                for split in splits:
                    # Safety: bail out if we have too many chunks
                    if len(chunks) >= self.MAX_CHUNKS_PER_DOC:
                        if current_chunk:
                            chunks.append(current_chunk)
                        return chunks
                    
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
                            sub_chunks = self._recursive_split(
                                split, remaining_seps, depth=depth + 1
                            )
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
        if self.DEBUG:
            print(f"        [hard_split] input: {len(text):,} chars, chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}", flush=True)
        
        chunks = []
        start = 0
        iteration = 0
        text_len = len(text)
        
        while start < text_len:
            iteration += 1
            if self.DEBUG and iteration <= 5:
                print(f"        [hard_split] iter {iteration}: start={start}", flush=True)
            if self.DEBUG and iteration == 100:
                print(f"        [hard_split] ... (continuing, {text_len - start:,} chars remaining)", flush=True)
            
            end = min(start + self.config.chunk_size, text_len)
            chunks.append(text[start:end])
            
            # CRITICAL: Don't apply overlap on final chunk (prevents infinite loop)
            if end >= text_len:
                break
            start = end - self.config.chunk_overlap if self.config.chunk_overlap > 0 else end
        
        if self.DEBUG:
            print(f"        [hard_split] → {len(chunks)} chunks in {iteration} iterations", flush=True)
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
        max_doc_chars: int = 100_000,
        ipc_chunksize: int = 32,
    ) -> List[Document]:
        """Chunk multiple documents in parallel using multiprocessing.

        Uses shared memory (via fork) to avoid IPC overhead. Documents are stored
        in module-level globals before forking; workers inherit these via copy-on-write.
        Only indices are passed through IPC, not the actual document data.

        Args:
            documents: List of documents to chunk (must be a list, not iterator)
            num_workers: Number of worker processes (default: CPU count)
            show_progress: Whether to show progress
            max_doc_chars: Maximum document size in chars (larger docs are truncated)
            ipc_chunksize: Number of docs sent to each worker at once (smaller = better
                load balancing, larger = less IPC overhead). Default 32 is a good balance.

        Returns:
            List of chunked Document objects
        """
        global _SHARED_DOCS, _SHARED_CONFIG
        
        import multiprocessing as mp
        import time

        from tqdm import tqdm

        if num_workers is None:
            num_workers = _get_available_cpus()

        doc_count = len(documents)
        if show_progress:
            print(f"    Chunking {doc_count} docs with {num_workers} workers...")

        # Prepare shared data - store in globals BEFORE forking
        # Workers will inherit these via fork (copy-on-write, no IPC overhead)
        t0 = time.time()
        _SHARED_CONFIG = {
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
            "separators": self.config.separators,
        }
        
        # Truncate oversized documents to prevent pathological chunking times
        truncated_count = 0
        shared_docs = []
        for doc in documents:
            text = doc.text
            if len(text) > max_doc_chars:
                text = text[:max_doc_chars]
                truncated_count += 1
            shared_docs.append({"id": doc.id, "text": text, "metadata": doc.metadata})
        _SHARED_DOCS = shared_docs
        
        if show_progress and truncated_count > 0:
            tqdm.write(f"      (truncated {truncated_count} oversized docs)")

        t1 = time.time()
        all_chunk_dicts = []
        
        pool = mp.Pool(processes=num_workers)
        
        try:
            # Use imap_unordered for progress visibility and good load balancing
            results_iter = pool.imap_unordered(
                _chunk_by_index, 
                range(doc_count), 
                chunksize=ipc_chunksize
            )
            
            # Collect results with progress bar (leave=False to clear when done)
            for chunk_dicts in tqdm(
                results_iter, 
                total=doc_count, 
                desc="      Chunking", 
                disable=not show_progress,
                leave=False,
                ncols=80,
            ):
                all_chunk_dicts.extend(chunk_dicts)
            
        finally:
            pool.close()
            pool.join()
        
        # Clean up shared memory
        _SHARED_DOCS = None
        _SHARED_CONFIG = None
        
        if show_progress:
            elapsed = time.time() - t1
            docs_per_sec = doc_count / elapsed if elapsed > 0 else 0
            tqdm.write(f"      ✓ Chunked in {elapsed:.1f}s ({docs_per_sec:.0f} docs/s)")

        # Convert dicts back to Document objects
        all_chunks = [
            Document(id=cd["id"], text=cd["text"], metadata=cd["metadata"])
            for cd in all_chunk_dicts
        ]

        if show_progress:
            avg = len(all_chunks) / max(doc_count, 1)
            tqdm.write(f"    ✓ {doc_count} docs → {len(all_chunks)} chunks (avg {avg:.1f}/doc)")

        return all_chunks

    def get_info(self) -> Dict[str, Any]:
        """Get chunking configuration info."""
        return {
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
        }
