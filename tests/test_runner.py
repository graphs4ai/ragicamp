"""Tests for experiment execution runner module."""

import pytest

from ragicamp.execution.runner import ExpSpec, build_specs, _name_direct, _name_rag


class TestExpSpec:
    """Tests for ExpSpec dataclass."""

    def test_exp_spec_defaults(self):
        """Test ExpSpec default values."""
        spec = ExpSpec(
            name="test_exp",
            exp_type="direct",
            model="hf:test/model",
            dataset="nq",
            prompt="default",
        )

        assert spec.name == "test_exp"
        assert spec.exp_type == "direct"
        assert spec.quant == "4bit"
        assert spec.retriever is None
        assert spec.top_k == 5
        assert spec.query_transform is None
        assert spec.reranker is None
        assert spec.batch_size == 8

    def test_exp_spec_with_rag_params(self):
        """Test ExpSpec with RAG parameters."""
        spec = ExpSpec(
            name="rag_exp",
            exp_type="rag",
            model="hf:test/model",
            dataset="hotpotqa",
            prompt="concise",
            retriever="dense_bge",
            top_k=10,
            query_transform="hyde",
            reranker="bge",
            reranker_model="BAAI/bge-reranker-large",
        )

        assert spec.retriever == "dense_bge"
        assert spec.top_k == 10
        assert spec.query_transform == "hyde"
        assert spec.reranker == "bge"
        assert spec.reranker_model == "BAAI/bge-reranker-large"


class TestBuildSpecs:
    """Tests for build_specs function."""

    def test_build_direct_specs(self):
        """Test building direct experiment specs."""
        config = {
            "datasets": ["nq"],
            "direct": {
                "enabled": True,
                "models": ["hf:test/model"],
                "prompts": ["default"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 1
        assert specs[0].exp_type == "direct"
        assert specs[0].model == "hf:test/model"
        assert specs[0].dataset == "nq"

    def test_build_rag_specs(self):
        """Test building RAG experiment specs."""
        config = {
            "datasets": ["hotpotqa"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge", "type": "dense"}],
                "top_k_values": [5],
                "prompts": ["concise"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 1
        assert specs[0].exp_type == "rag"
        assert specs[0].retriever == "dense_bge"

    def test_build_specs_multiple_datasets(self):
        """Test building specs with multiple datasets."""
        config = {
            "datasets": ["nq", "hotpotqa"],
            "direct": {
                "enabled": True,
                "models": ["hf:test/model"],
                "prompts": ["default"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 2
        datasets = [s.dataset for s in specs]
        assert "nq" in datasets
        assert "hotpotqa" in datasets

    def test_build_specs_multiple_top_k(self):
        """Test building specs with multiple top_k values."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [3, 5, 10],
                "prompts": ["default"],
                "quantization": ["4bit"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 3
        top_ks = [s.top_k for s in specs]
        assert 3 in top_ks
        assert 5 in top_ks
        assert 10 in top_ks

    def test_build_specs_with_query_transform(self):
        """Test building specs with query transforms."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [5],
                "prompts": ["default"],
                "quantization": ["4bit"],
                "query_transform": ["none", "hyde"],
            },
        }

        specs = build_specs(config)

        assert len(specs) == 2
        transforms = [s.query_transform for s in specs]
        assert None in transforms  # "none" becomes None
        assert "hyde" in transforms

    def test_build_specs_with_reranker(self):
        """Test building specs with rerankers."""
        config = {
            "datasets": ["nq"],
            "rag": {
                "enabled": True,
                "models": ["hf:test/model"],
                "retrievers": [{"name": "dense_bge"}],
                "top_k_values": [5],
                "prompts": ["default"],
                "quantization": ["4bit"],
                "reranker": {
                    "configs": [
                        {"enabled": False, "name": "none"},
                        {"enabled": True, "name": "bge", "model": "BAAI/bge-reranker-large"},
                    ]
                },
            },
        }

        specs = build_specs(config)

        assert len(specs) == 2
        rerankers = [s.reranker for s in specs]
        assert None in rerankers  # disabled reranker becomes None
        assert "bge" in rerankers

    def test_build_specs_openai_skips_non_4bit(self):
        """Test that OpenAI models skip non-4bit quantization."""
        config = {
            "datasets": ["nq"],
            "direct": {
                "enabled": True,
                "models": ["openai:gpt-4o-mini"],
                "prompts": ["default"],
                "quantization": ["4bit", "8bit"],
            },
        }

        specs = build_specs(config)

        # Should only create 1 spec (4bit), skip 8bit for OpenAI
        assert len(specs) == 1
        assert specs[0].quant == "4bit"


class TestNamingFunctions:
    """Tests for experiment naming functions."""

    def test_name_direct(self):
        """Test direct experiment naming."""
        name = _name_direct("hf:meta-llama/Llama-3.2", "default", "nq", "4bit")

        assert "direct" in name
        assert "nq" in name
        # Should replace special chars
        assert ":" not in name
        assert "/" not in name

    def test_name_direct_non_4bit_suffix(self):
        """Test direct naming includes quantization suffix for non-4bit."""
        name_4bit = _name_direct("hf:test", "default", "nq", "4bit")
        name_8bit = _name_direct("hf:test", "default", "nq", "8bit")

        assert "_8bit" not in name_4bit
        assert "_8bit" in name_8bit

    def test_name_rag(self):
        """Test RAG experiment naming."""
        name = _name_rag("hf:test", "default", "nq", "4bit", "dense_bge", 5)

        assert "rag" in name
        assert "dense_bge" in name
        assert "k5" in name

    def test_name_rag_with_query_transform(self):
        """Test RAG naming with query transform."""
        name = _name_rag("hf:test", "default", "nq", "4bit", "dense", 5, qt="hyde")

        assert "hyde" in name

    def test_name_rag_with_reranker(self):
        """Test RAG naming with reranker."""
        name = _name_rag("hf:test", "default", "nq", "4bit", "dense", 5, rr="bge")

        assert "bge" in name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
