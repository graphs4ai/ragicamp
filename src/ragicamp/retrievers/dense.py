"""Dense retriever using vector similarity."""

import pickle
from pathlib import Path
from typing import Any, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.artifacts import get_artifact_manager


class DenseRetriever(Retriever):
    """Dense retriever using neural embeddings and vector similarity.

    Uses sentence transformers to encode queries and documents,
    then performs similarity search using FAISS.
    """

    def __init__(
        self,
        name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        **kwargs: Any,
    ):
        """Initialize dense retriever.

        Args:
            name: Retriever identifier
            embedding_model: Sentence transformer model name
            index_type: FAISS index type (flat, ivf, hnsw)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.embedding_model_name = embedding_model  # Store the model name
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index_type = index_type

        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.documents: List[Document] = []

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents by computing and storing their embeddings."""
        self.documents = documents

        # Compute embeddings
        texts = [doc.text for doc in documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to FAISS index
        if self.index_type == "ivf":
            self.index.train(embeddings.astype("float32"))

        self.index.add(embeddings.astype("float32"))

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """Retrieve documents using dense similarity search."""
        if len(self.documents) == 0:
            return []

        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search
        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)

        # Return documents with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc.score = float(score)
                results.append(doc)

        return results

    def save_index(self, artifact_name: str) -> str:
        """Save the indexed documents and FAISS index.

        Args:
            artifact_name: Name for this retriever artifact (e.g., 'wikipedia_nq_v1')

        Returns:
            Path where the artifact was saved
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Save FAISS index
        faiss.write_index(self.index, str(artifact_path / "index.faiss"))

        # Save documents
        with open(artifact_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Save config
        config = {
            "name": self.name,
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "num_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
        }
        manager.save_json(config, artifact_path / "config.json")

        print(f"✓ Saved retriever to: {artifact_path}")
        return str(artifact_path)

    @classmethod
    def load_index(
        cls, artifact_name: str, embedding_model: Optional[str] = None
    ) -> "DenseRetriever":
        """Load a previously saved retriever index.

        Args:
            artifact_name: Name of the retriever artifact to load
            embedding_model: Optional override for embedding model

        Returns:
            Loaded DenseRetriever instance
        """
        manager = get_artifact_manager()
        artifact_path = manager.get_retriever_path(artifact_name)

        # Load config
        config = manager.load_json(artifact_path / "config.json")

        # Use config embedding model if not overridden
        if embedding_model is None:
            embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")

        # Create retriever
        retriever = cls(
            name=config["name"], embedding_model=embedding_model, index_type=config["index_type"]
        )

        # Load FAISS index
        retriever.index = faiss.read_index(str(artifact_path / "index.faiss"))

        # Load documents
        with open(artifact_path / "documents.pkl", "rb") as f:
            retriever.documents = pickle.load(f)

        print(f"✓ Loaded retriever from: {artifact_path}")
        print(f"  - {len(retriever.documents)} documents indexed")

        return retriever
