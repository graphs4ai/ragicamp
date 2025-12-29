"""Sparse retriever using BM25."""

from typing import Any, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragicamp.retrievers.base import Document, Retriever


class SparseRetriever(Retriever):
    """Sparse retriever using BM25-like TF-IDF scoring.

    Uses scikit-learn's TF-IDF vectorizer for simple sparse retrieval.
    For production BM25, consider using rank-bm25 library.
    """

    def __init__(self, name: str, max_features: int = 10000, **kwargs: Any):
        """Initialize sparse retriever.

        Args:
            name: Retriever identifier
            max_features: Maximum number of features for TF-IDF
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", **kwargs)
        self.documents: List[Document] = []
        self.doc_vectors = None

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents using TF-IDF."""
        self.documents = documents
        texts = [doc.text for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """Retrieve documents using TF-IDF similarity."""
        if len(self.documents) == 0 or self.doc_vectors is None:
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Compute similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        # Get top-k
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Return documents with scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.score = float(similarities[idx])
            results.append(doc)

        return results
