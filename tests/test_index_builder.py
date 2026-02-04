"""Tests for index building.

Note: The old get_embedding_index_name function was removed during
the clean architecture migration. These tests are skipped until
new index naming tests are added.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Index naming function removed during architecture migration")


class TestGetEmbeddingIndexName:
    """Tests for index naming function - DEPRECATED."""

    def test_placeholder(self):
        """Placeholder test."""
        pass
