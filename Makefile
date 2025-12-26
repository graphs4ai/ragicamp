# RAGiCamp Makefile

.PHONY: help install index-simple index-full run-baseline-simple run-baseline-full

# ============================================================================
# HELP
# ============================================================================

help:
	@echo ""
	@echo "RAGiCamp"
	@echo "========"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Install dependencies"
	@echo ""
	@echo "Quick Development (simple):"
	@echo "  make index-simple         Build 1 small index (500 docs)"
	@echo "  make run-baseline-simple  Run quick baseline (10 questions)"
	@echo ""
	@echo "Full Experiments:"
	@echo "  make index-full           Build all indexes for baseline"
	@echo "  make run-baseline-full    Run full baseline (100+ questions)"
	@echo ""
	@echo "Other:"
	@echo "  make download             Download datasets"
	@echo "  make test                 Run tests"
	@echo "  make clean                Clean outputs"
	@echo ""

# ============================================================================
# SETUP
# ============================================================================

install:
	uv sync

download:
	uv run python scripts/data/download.py --all

# ============================================================================
# SIMPLE (quick dev/validation)
# ============================================================================

index-simple:
	@echo "📚 Building simple index (1 index, 500 docs)..."
	uv run python scripts/data/build_all_indexes.py --minimal --max-docs 500

run-baseline-simple:
	@echo "🚀 Running simple baseline (10 questions)..."
	uv run python scripts/experiments/run_study.py conf/study/simple.yaml

# ============================================================================
# FULL (production experiments)
# ============================================================================

index-full:
	@echo "📚 Building full indexes (4 indexes, all docs)..."
	uv run python scripts/data/build_all_indexes.py --standard

run-baseline-full:
	@echo "🚀 Running full baseline (100 questions, all variations)..."
	uv run python scripts/experiments/run_study.py conf/study/full.yaml

# ============================================================================
# DEV
# ============================================================================

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/

clean:
	rm -rf outputs/
	@echo "✓ Cleaned"
