"""Base class for query transformers."""

from abc import ABC, abstractmethod
from typing import List


class QueryTransformer(ABC):
    """Base class for query transformation strategies.

    Query transformers modify or expand the original query to improve
    retrieval quality. Examples include:
    - HyDE: Generate hypothetical answers and search with those
    - Multi-query: Rewrite query into multiple variations
    """

    @abstractmethod
    def transform(self, query: str) -> List[str]:
        """Transform a query into one or more search queries.

        Args:
            query: The original user query

        Returns:
            List of transformed queries to search with.
            The original query may or may not be included.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PassthroughTransformer(QueryTransformer):
    """No-op transformer that returns the query unchanged.

    Useful as a baseline or when no transformation is needed.
    """

    def transform(self, query: str) -> List[str]:
        """Return the original query unchanged."""
        return [query]
