"""Experiment naming conventions.

Provides functions to generate canonical experiment names from parameters.
Consistent naming is crucial for experiment identification and reproducibility.
"""


def name_direct(model: str, prompt: str, dataset: str) -> str:
    """Generate experiment name for direct (non-RAG) experiments.

    Args:
        model: Model specification (e.g., 'hf:google/gemma-2-2b-it')
        prompt: Prompt style name
        dataset: Dataset name

    Returns:
        Canonical experiment name

    Example:
        >>> name_direct("hf:google/gemma-2-2b-it", "default", "nq")
        'direct_hf_google_gemma22bit_default_nq'
    """
    # Normalize model name: replace special chars
    m = model.replace(":", "_").replace("/", "_").replace("-", "")

    return f"direct_{m}_{prompt}_{dataset}"


def name_rag(
    model: str,
    prompt: str,
    dataset: str,
    retriever: str,
    top_k: int,
    query_transform: str = "none",
    reranker: str = "none",
) -> str:
    """Generate experiment name for RAG experiments.

    Args:
        model: Model specification
        prompt: Prompt style name
        dataset: Dataset name
        retriever: Retriever name
        top_k: Number of documents to retrieve
        query_transform: Query transformation type ('hyde', 'multiquery', 'none')
        reranker: Reranker type ('bge', 'ms-marco', 'none')

    Returns:
        Canonical experiment name

    Example:
        >>> name_rag("hf:gemma-2-2b-it", "default", "nq", "dense_minilm", 5)
        'rag_hf_gemma22bit_dense_minilm_k5_default_nq'
    """
    # Normalize model name
    m = model.replace(":", "_").replace("/", "_").replace("-", "")

    # Build name parts
    parts = ["rag", m, retriever, f"k{top_k}"]

    # Add optional components
    if query_transform and query_transform != "none":
        parts.append(query_transform)

    if reranker and reranker != "none":
        parts.append(reranker)

    parts.extend([prompt, dataset])

    return "_".join(parts)
