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

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5, **kwargs: Any
    ) -> List[List[Document]]:
        """Retrieve documents for multiple queries using batched TF-IDF.

        Vectorizes all queries at once for efficiency.

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query

        Returns:
            List of document lists, one per query
        """
        import numpy as np

        if len(self.documents) == 0 or self.doc_vectors is None:
            return [[] for _ in queries]

        # Batch vectorize all queries at once
        query_vectors = self.vectorizer.transform(queries)

        # Compute all similarities at once (queries x documents matrix)
        all_similarities = cosine_similarity(query_vectors, self.doc_vectors)

        # Get top-k for each query
        all_results = []
        for i, similarities in enumerate(all_similarities):
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                # Create a copy of the document to avoid modifying the original
                doc = Document(
                    id=self.documents[idx].id,
                    text=self.documents[idx].text,
                    metadata=self.documents[idx].metadata.copy()
                    if self.documents[idx].metadata
                    else {},
                )
                doc.score = float(similarities[idx])
                results.append(doc)

            all_results.append(results)

        return all_results
