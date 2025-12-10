# RAGiCamp Makefile - Common commands

.PHONY: help install setup test lint format clean

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║                    RAGiCamp - Commands                       ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 SETUP & INSTALLATION"
	@echo "  make install              - Install dependencies"
	@echo "  make install-dev          - Install with dev tools"
	@echo "  make install-all          - Install everything (dev + viz)"
	@echo "  make setup                - Full setup (install + verify)"
	@echo "  make verify-install       - Verify all dependencies are working"
	@echo ""
	@echo "📚 DATASETS"
	@echo "  make download-nq          - Download Natural Questions (validation)"
	@echo "  make download-triviaqa    - Download TriviaQA (validation)"
	@echo "  make download-hotpotqa    - Download HotpotQA (validation)"
	@echo "  make download-all         - Download all datasets (validation splits)"
	@echo "  make list-datasets        - List downloaded datasets"
	@echo "  ℹ️  Advanced: uv run python experiments/scripts/download_datasets.py --help"
	@echo ""
	@echo "🏋️  INDEXING & CORPUS"
	@echo "  make index-wiki-simple    - Index Simple Wikipedia (~200k articles)"
	@echo "  make index-wiki-small     - Quick test (10k articles, full docs)"
	@echo "  make index-wiki-small-chunked - Quick test (10k articles, CHUNKED - better!)"
	@echo "  make list-artifacts       - List saved artifacts"
	@echo "  make clean-artifacts      - Remove all artifacts"
	@echo ""
	@echo "🚀 EVALUATION (Config-Based)"
	@echo "  make eval-baseline-quick  - Quick test (10 examples, GPU)"
	@echo "  make eval-baseline-full   - Full eval (100 examples, GPU, all metrics)"
	@echo "  make eval-baseline-cpu    - CPU evaluation (10 examples, slower)"
	@echo "  make eval-rag             - RAG evaluation (small chunked corpus)"
	@echo "  make eval-rag-wiki-simple - RAG with full Wikipedia Simple (~200k articles)"
	@echo ""
	@echo "🧪 TESTING"
	@echo "  make test                 - Run all tests"
	@echo "  make test-fast            - Run tests (skip slow ones)"
	@echo "  make test-coverage        - Run tests with coverage report"
	@echo "  make test-two-phase       - Test two-phase evaluation"
	@echo "  make test-checkpoint      - Test checkpointing"
	@echo "  make test-config          - Test config validation"
	@echo ""
	@echo "🔧 DEVELOPMENT"
	@echo "  make lint                 - Run linting"
	@echo "  make format               - Format code (black + isort)"
	@echo "  make clean                - Clean generated files"
	@echo ""
	@echo "🔍 CONFIGURATION"
	@echo "  make validate-config CONFIG=path/to/config.yaml"
	@echo "  make validate-all-configs  - Validate all experiment configs"
	@echo "  make create-config OUTPUT=my_exp.yaml [TYPE=baseline|rag|llm_judge]"
	@echo ""
	@echo "💡 TIPS"
	@echo "  - First time? Run: make setup"
	@echo "  - Quick start GPU: make eval-baseline-quick"
	@echo "  - No GPU? Use: make eval-baseline-cpu"
	@echo "  - Edit configs in experiments/configs/ for custom experiments"
	@echo ""

# ============================================================================
# Setup & Installation
# ============================================================================

install:
	@echo "📦 Installing base dependencies..."
	uv sync

install-dev:
	@echo "📦 Installing with dev tools..."
	uv sync --extra dev

install-viz:
	@echo "📦 Installing with visualization tools..."
	uv sync --extra viz

install-all:
	@echo "📦 Installing all dependencies..."
	uv sync --extra dev --extra viz

verify-install:
	@echo "🔍 Verifying installation..."
	@uv run python -c "import torch; print('✓ PyTorch:', torch.__version__)"
	@uv run python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
	@uv run python -c "from bert_score import BERTScorer; print('✓ BERTScore: OK')"
	@uv run python -c "import bleurt; print('✓ BLEURT: OK (checkpoint will auto-download on first use)')"
	@echo ""
	@echo "✅ All dependencies installed correctly!"

setup: install verify-install
	@echo ""
	@echo "✅ Setup complete! You can now run:"
	@echo "   make eval-baseline-quick  - Quick test (10 examples)"
	@echo "   make eval-baseline-full   - Full evaluation (100 examples, all metrics)"
	@echo ""
	@echo "📝 Note: BLEURT checkpoint (~500MB) will auto-download on first evaluation"
	@echo "📖 See docs/guides/CONFIG_BASED_EVALUATION.md for detailed usage"
	@echo ""

# ============================================================================
# Dataset Download & Management
# ============================================================================

download-nq:
	@echo "📚 Downloading Natural Questions (validation split)..."
	@echo "⏱️  This will take a few minutes"
	@echo ""
	uv run python experiments/scripts/download_datasets.py \
		--dataset natural_questions \
		--split validation \
		--output-dir data/datasets

download-triviaqa:
	@echo "📚 Downloading TriviaQA (validation split)..."
	@echo "⏱️  This will take a few minutes"
	@echo ""
	uv run python experiments/scripts/download_datasets.py \
		--dataset triviaqa \
		--split validation \
		--output-dir data/datasets

download-hotpotqa:
	@echo "📚 Downloading HotpotQA (validation split)..."
	@echo "⏱️  This will take a few minutes"
	@echo ""
	uv run python experiments/scripts/download_datasets.py \
		--dataset hotpotqa \
		--split validation \
		--output-dir data/datasets

download-all:
	@echo "📚 Downloading all datasets (validation splits)..."
	@echo "⏱️  This will take 10-15 minutes"
	@echo ""
	uv run python experiments/scripts/download_datasets.py \
		--dataset all \
		--split validation \
		--output-dir data/datasets

list-datasets:
	@echo "📊 Downloaded datasets:"
	@echo ""
	@if [ -d data/datasets ]; then \
		for file in data/datasets/*.json; do \
			if [ -f "$$file" ]; then \
				name=$$(basename $$file); \
				size=$$(du -h "$$file" | cut -f1); \
				count=$$(jq '.info.filtered_size // 0' "$$file" 2>/dev/null || echo "?"); \
				printf "  %-40s %8s    %8s examples\n" "$$name" "$$size" "$$count"; \
			fi \
		done; \
		echo ""; \
		total_size=$$(du -sh data/datasets 2>/dev/null | cut -f1); \
		echo "  Total size: $$total_size"; \
	else \
		echo "  (no datasets downloaded yet)"; \
		echo "  Run: make download-nq"; \
	fi

clean-datasets:
	@echo "🧹 Cleaning downloaded datasets..."
	@if [ -d data/datasets ]; then \
		rm -rf data/datasets/*.json; \
		echo "✓ Datasets removed"; \
	else \
		echo "  (no datasets to clean)"; \
	fi

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "🧪 Running all tests..."
	uv run pytest tests/ -v

test-fast:
	@echo "⚡ Running fast tests (skip slow ones)..."
	uv run pytest tests/ -v -m "not slow"

test-coverage:
	@echo "🧪 Running tests with coverage..."
	uv run pytest tests/ --cov=src/ragicamp --cov-report=html --cov-report=term
	@echo ""
	@echo "✅ Coverage report generated: htmlcov/index.html"
	@echo "   Open with: open htmlcov/index.html (macOS) or xdg-open htmlcov/index.html (Linux)"

test-two-phase:
	@echo "🧪 Testing two-phase evaluation..."
	uv run pytest tests/test_two_phase_evaluation.py -v

test-checkpoint:
	@echo "🧪 Testing checkpoint system..."
	uv run pytest tests/test_checkpointing.py -v

test-config:
	@echo "🧪 Testing config validation..."
	uv run pytest tests/test_config.py -v

test-metrics:
	@echo "🧪 Testing metrics..."
	uv run pytest tests/test_metrics.py -v

test-factory:
	@echo "🧪 Testing factory..."
	uv run pytest tests/test_factory.py -v

test-agents:
	@echo "🧪 Testing agents..."
	uv run pytest tests/test_agents.py -v

test-watch:
	@echo "👀 Running tests in watch mode..."
	uv run pytest-watch tests/ -v

# ============================================================================
# Development
# ============================================================================

lint:
	@echo "🔍 Running linters..."
	@uv run flake8 src/ tests/ || true
	@uv run mypy src/ || true

format:
	@echo "✨ Formatting code..."
	uv run black src/ tests/ experiments/ --line-length 99
	uv run isort src/ tests/ experiments/ --profile black
	@echo "✅ Code formatted!"

clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache dist/ build/ *.egg-info
	@echo "✅ Cleaned!"

clean-outputs:
	@echo "🧹 Cleaning output files..."
	rm -rf outputs/*.json
	@echo "✅ Outputs cleaned!"

clean-all: clean clean-outputs clean-datasets clean-artifacts
	@echo "✅ Everything cleaned!"

# ============================================================================
# Config-Based Evaluation (RECOMMENDED)
# ============================================================================

eval-baseline-quick:
	@echo "🚀 Running baseline evaluation (quick test)"
	@echo "📋 Config: experiments/configs/nq_baseline_gemma2b_quick.yaml"
	@echo "⏱️  ~2-3 minutes on GPU"
	@echo ""
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_baseline_gemma2b_quick.yaml \
		--mode eval

eval-baseline-cpu:
	@echo "🚀 Running baseline evaluation on CPU"
	@echo "📋 Config: experiments/configs/nq_baseline_gemma2b_cpu.yaml"
	@echo "⚠️  CPU is SLOW: ~30-60 minutes for 10 examples"
	@echo "⏳  Loading model... (this may take a few minutes)"
	@echo ""
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_baseline_gemma2b_cpu.yaml \
		--mode eval

eval-baseline-full:
	@echo "🚀 Running baseline evaluation (full - 100 examples, all metrics)"
	@echo "📋 Config: experiments/configs/nq_baseline_gemma2b_full.yaml"
	@echo "⏱️  ~20-25 minutes on GPU"
	@echo ""
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_baseline_gemma2b_full.yaml \
		--mode eval

eval-rag:
	@echo "🔍 Running FixedRAG evaluation (with chunked corpus)"
	@echo "📋 Config: experiments/configs/nq_fixed_rag_gemma2b.yaml"
	@if [ ! -d artifacts/retrievers/wikipedia_small_chunked ]; then \
		echo "⚠️  Chunked Wikipedia index not found. Indexing first..."; \
		$(MAKE) index-wiki-small-chunked; \
	fi
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_fixed_rag_gemma2b.yaml \
		--mode eval

eval-rag-wiki-simple:
	@echo "🔍 Running FixedRAG evaluation (with Wikipedia Simple corpus)"
	@echo "📋 Config: experiments/configs/nq_fixed_rag_wiki_simple.yaml"
	@echo "📚 Corpus: Full Simple Wikipedia (~200k articles)"
	@echo "⏱️  ~15-20 minutes on GPU (100 examples)"
	@echo ""
	@if [ ! -d artifacts/retrievers/wikipedia_simple_chunked_1024_overlap_128 ]; then \
		echo "⚠️  Wikipedia Simple index not found. Indexing first..."; \
		echo "⏱️  This will take 30-60 minutes (one-time setup)"; \
		$(MAKE) index-wiki-simple; \
	fi
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_fixed_rag_wiki_simple.yaml \
		--mode eval

# ============================================================================
# Training & Indexing
# ============================================================================

index-wiki-simple:
	@echo "📚 Indexing Simple English Wikipedia (full ~200k articles)..."
	@echo "⚠️  This will take 30-60 minutes"
	@echo ""
	uv run python experiments/scripts/index_corpus.py \
		--corpus-name wikipedia_simple \
		--corpus-version 20231101.simple \
		--embedding-model all-MiniLM-L6-v2 \
		--artifact-name wikipedia_simple_chunked_1024_overlap_128 \
		--chunk-size 1024 \
		--chunk-overlap 128

index-wiki-small:
	@echo "📚 Quick test: Indexing 10k Simple Wikipedia articles..."
	@echo "⚠️  For TESTING ONLY - use index-wiki-simple for evaluation"
	@echo ""
	uv run python experiments/scripts/index_corpus.py \
		--corpus-name wikipedia_simple \
		--corpus-version 20231101.simple \
		--embedding-model all-MiniLM-L6-v2 \
		--max-docs 10000 \
		--artifact-name wikipedia_small

index-wiki-small-chunked:
	@echo "📚 Indexing 10k Wikipedia articles WITH CHUNKING..."
	@echo "📄 Strategy: recursive (512 chars, 50 overlap)"
	@echo "✨ Better retrieval quality than full-document indexing!"
	@echo ""
	uv run python experiments/scripts/index_corpus.py \
		--corpus-name wikipedia_simple \
		--corpus-version 20231101.simple \
		--embedding-model all-MiniLM-L6-v2 \
		--max-docs 10000 \
		--chunk-strategy recursive \
		--chunk-size 512 \
		--chunk-overlap 50 \
		--artifact-name wikipedia_small_chunked_512_overlap_50

list-artifacts:
	@echo "📦 Saved artifacts:"
	@echo ""
	@if [ -d artifacts/retrievers ]; then \
		echo "Retrievers:"; \
		ls -1 artifacts/retrievers/ 2>/dev/null || echo "  (none)"; \
		echo ""; \
	fi
	@if [ -d artifacts/agents ]; then \
		echo "Agents:"; \
		ls -1 artifacts/agents/ 2>/dev/null || echo "  (none)"; \
	fi

clean-artifacts:
	@echo "🧹 Cleaning artifacts..."
	rm -rf artifacts/
	@echo "✓ Artifacts removed"

# ============================================================================
# Configuration Management
# ============================================================================

validate-config:
	@echo "🔍 Validating configuration..."
	uv run python scripts/validate_config.py $(CONFIG)

validate-all-configs:
	@echo "🔍 Validating all experiment configs..."
	@uv run python scripts/validate_config.py experiments/configs/*.yaml

create-config:
	@echo "📝 Creating config template..."
	@TYPE=$(if $(TYPE),$(TYPE),baseline); \
	OUTPUT=$(if $(OUTPUT),$(OUTPUT),my_experiment.yaml); \
	uv run python scripts/create_config.py $$OUTPUT --type $$TYPE
