# RAGiCamp Makefile
# Simple, config-driven experiments

.PHONY: help env install setup test run index clean

# Default study config
STUDY ?= validation

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                      RAGiCamp Commands                            ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 SETUP"
	@echo "  make install          Install dependencies"
	@echo "  make env              Show how to set env vars"
	@echo ""
	@echo "🚀 RUN STUDIES (config-driven)"
	@echo "  make run                      Run validation study (default)"
	@echo "  make run STUDY=validation     Quick end-to-end test (10 questions)"
	@echo "  make run STUDY=baseline       Full baseline (100 questions)"
	@echo "  make preview                  Preview what would run (dry-run)"
	@echo ""
	@echo "📚 DATA & INDEXES"
	@echo "  make download         Download datasets"
	@echo "  make index            Build 1 test index (500 docs)"
	@echo "  make index-full       Build all indexes for baseline"
	@echo ""
	@echo "🔧 DEV"
	@echo "  make test             Run tests"
	@echo "  make lint             Run linter"
	@echo "  make clean            Clean outputs"
	@echo ""
	@echo "📁 Study configs: conf/study/"
	@echo "   validation.yaml  - Quick pipeline test"
	@echo "   baseline.yaml    - Full baseline experiments"

# ============================================================================
# SETUP
# ============================================================================

install:
	uv sync

env:
	@echo "Run: source ~/setup_env.sh"
	@echo ""
	@echo "Or manually:"
	@echo "  export HF_TOKEN=your_token"
	@echo "  export OPENAI_API_KEY=your_key"

setup: install
	@echo "✓ Installed"
	uv run python -c "import ragicamp; print(f'RAGiCamp {ragicamp.__version__} ready')"

# ============================================================================
# RUN STUDIES
# ============================================================================

run:
	@echo "📊 Running study: $(STUDY)"
	uv run python scripts/experiments/run_study.py conf/study/$(STUDY).yaml

preview:
	@echo "📋 Preview study: $(STUDY)"
	uv run python scripts/experiments/run_study.py conf/study/$(STUDY).yaml --dry-run

# ============================================================================
# DATA & INDEXING
# ============================================================================

download:
	uv run python scripts/data/download.py --all

index:
	@echo "📚 Building test index (500 docs)..."
	uv run python scripts/data/build_all_indexes.py --minimal --max-docs 500

index-full:
	@echo "📚 Building indexes for baseline study..."
	uv run python scripts/data/build_all_indexes.py --standard

index-preview:
	uv run python scripts/data/build_all_indexes.py --standard --dry-run

# ============================================================================
# DEV
# ============================================================================

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

clean:
	rm -rf outputs/
	rm -rf mlruns/
	@echo "✓ Cleaned outputs"

# ============================================================================
# LEGACY (for backward compatibility - will deprecate)
# ============================================================================

quick-test:
	@echo "Use: make run STUDY=validation"
	make run STUDY=validation

baseline:
	uv run python -m ragicamp.cli.run experiment=baseline

rag:
	uv run python -m ragicamp.cli.run experiment=rag
