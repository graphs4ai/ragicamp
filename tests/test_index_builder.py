"""Tests for indexes/builder module."""

import pytest

from ragicamp.indexes.builder import get_embedding_index_name


class TestGetEmbeddingIndexName:
    """Tests for index naming function."""

    def test_basic_name_generation(self):
        """Test basic index name generation."""
        name = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )

        # Should include corpus short name
        assert "en_" in name
        # Should include model short name (normalized)
        assert "bge" in name.lower()
        # Should include chunk config
        assert "c512" in name
        assert "o50" in name

    def test_same_config_same_name(self):
        """Test that same config produces same name."""
        name1 = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )
        name2 = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )

        assert name1 == name2

    def test_different_config_different_name(self):
        """Test that different config produces different name."""
        name1 = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )
        name2 = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=1024,  # Different chunk size
            chunk_overlap=50,
            corpus_version="20231101.en",
        )

        assert name1 != name2

    def test_different_model_different_name(self):
        """Test that different model produces different name."""
        name_bge = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )
        name_e5 = get_embedding_index_name(
            embedding_model="intfloat/e5-large-v2",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )

        assert name_bge != name_e5

    def test_different_corpus_different_name(self):
        """Test that different corpus produces different name."""
        name_en = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )
        name_simple = get_embedding_index_name(
            embedding_model="BAAI/bge-large-en-v1.5",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.simple",
        )

        assert name_en != name_simple
        assert "en_" in name_en
        assert "simple_" in name_simple

    def test_model_name_normalization(self):
        """Test that model names are normalized (no special chars)."""
        name = get_embedding_index_name(
            embedding_model="some/model-with-dashes",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="20231101.en",
        )

        # Should not contain slashes or dashes in the model part
        # The format is: {corpus}_{model}_c{size}_o{overlap}
        parts = name.split("_")
        model_part = parts[1]  # Second part is model
        assert "-" not in model_part
        assert "/" not in model_part

    def test_corpus_version_without_dot(self):
        """Test corpus version without dot separator."""
        name = get_embedding_index_name(
            embedding_model="test/model",
            chunk_size=512,
            chunk_overlap=50,
            corpus_version="simple",  # No dot
        )

        assert "simple_" in name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
