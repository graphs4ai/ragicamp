"""Retriever factory for creating retrievers from configuration."""

from typing import Any, Dict, Optional

from ragicamp.core.logging import get_logger
from ragicamp.models.base import LanguageModel
from ragicamp.retrievers import DenseRetriever, HierarchicalRetriever, HybridRetriever, Retriever

logger = get_logger(__name__)


class RetrieverFactory:
    """Factory for creating retrievers from configuration."""

    # Custom retriever registry
    _custom_retrievers: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Register a custom retriever type."""
        def decorator(retriever_class: type) -> type:
            cls._custom_retrievers[name] = retriever_class
            return retriever_class
        return decorator

    @staticmethod
    def create(config: Dict[str, Any]) -> Retriever:
        """Create a retriever from configuration.

        Args:
            config: Retriever configuration dict with 'type' and retriever-specific params

        Returns:
            Instantiated Retriever

        Example:
            >>> config = {"type": "dense", "embedding_model": "all-MiniLM-L6-v2"}
            >>> retriever = RetrieverFactory.create(config)
        """
        retriever_type = config.get("type", "dense")
        config_copy = dict(config)
        config_copy.pop("type", None)

        if retriever_type == "dense":
            return DenseRetriever(**config_copy)
        elif retriever_type == "sparse":
            from ragicamp.retrievers import SparseRetriever
            return SparseRetriever(**config_copy)
        elif retriever_type == "hybrid":
            return HybridRetriever(**config_copy)
        elif retriever_type == "hierarchical":
            return HierarchicalRetriever(**config_copy)
        else:
            raise ValueError(
                f"Unknown retriever type: {retriever_type}. "
                f"Available: dense, sparse, hybrid, hierarchical"
            )

    @staticmethod
    def load(retriever_name: str) -> Retriever:
        """Load a retriever by name, automatically detecting the type.

        Args:
            retriever_name: Name of the retriever to load

        Returns:
            Loaded retriever instance
        """
        from ragicamp.utils.artifacts import get_artifact_manager

        manager = get_artifact_manager()
        retriever_path = manager.get_retriever_path(retriever_name)
        config_path = retriever_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Retriever config not found: {config_path}")

        config = manager.load_json(config_path)
        retriever_type = config.get("type", "dense")

        if retriever_type == "hierarchical":
            return HierarchicalRetriever.load(retriever_name)
        elif retriever_type == "hybrid":
            return HybridRetriever.load(retriever_name)
        else:
            return DenseRetriever.load(retriever_name)


def create_query_transformer(transform_type: str, model: LanguageModel):
    """Create a query transformer.

    Args:
        transform_type: Type of transformer (hyde, multiquery)
        model: Language model to use for transformation

    Returns:
        QueryTransformer instance or None
    """
    if not transform_type or transform_type == "none":
        return None

    if transform_type == "hyde":
        from ragicamp.rag.query_transform.hyde import HyDETransformer
        return HyDETransformer(model)
    elif transform_type == "multiquery":
        from ragicamp.rag.query_transform.multiquery import MultiQueryTransformer
        return MultiQueryTransformer(model, num_queries=3)
    else:
        logger.warning("Unknown query transform type: %s", transform_type)
        return None


def create_reranker(reranker_model: str):
    """Create a reranker.

    Args:
        reranker_model: Reranker model name (bge, ms-marco)

    Returns:
        Reranker instance or None
    """
    if not reranker_model:
        return None

    from ragicamp.rag.rerankers.cross_encoder import CrossEncoderReranker

    # Map short names to full model names
    model_map = {
        "bge": "BAAI/bge-reranker-large",
        "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-large": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    }

    full_model_name = model_map.get(reranker_model, reranker_model)
    return CrossEncoderReranker(model_name=full_model_name)
