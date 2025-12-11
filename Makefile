# RAGiCamp Makefile
# Run `make help` for available commands

.PHONY: help install setup test lint format clean

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                      RAGiCamp Commands                            ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 SETUP"
	@echo "  make setup            Full setup (install + verify)"
	@echo "  make install          Install dependencies only"
	@echo ""
	@echo "🚀 QUICK START"
	@echo "  make quick-test       Smoke test (10 examples)"
	@echo "  make baseline         DirectLLM evaluation"
	@echo "  make rag              RAG evaluation (needs index)"
	@echo ""
	@echo "📊 BASELINE STUDIES"
	@echo "  make baseline-study-direct   DirectLLM study (2 models × 3 datasets)"
	@echo "  make baseline-study-full     Full study (DirectLLM + RAG)"
	@echo "  make baseline-study-preview  Preview commands (dry-run)"
	@echo ""
	@echo "🔍 RAG STUDIES (run 'make index' first)"
	@echo "  make rag-study-standard  Standard RAG study"
	@echo "  make rag-study-topk      Sweep top_k values"
	@echo "  make rag-study-prompts   Compare prompts"
	@echo ""
	@echo "📚 DATA"
	@echo "  make download-all     Download datasets"
	@echo "  make index            Build retriever index"
	@echo "  make info             Show data info"
	@echo ""
	@echo "🧪 DEV"
	@echo "  make test             Run tests"
	@echo "  make lint             Run linters"
	@echo "  make clean            Clean generated files"
	@echo ""
	@echo "💡 See CHEATSHEET.md for more examples"

# ============================================================================
# SETUP
# ============================================================================

install:
	@echo "📦 Installing dependencies..."
	uv sync

install-dev:
	@echo "📦 Installing with dev tools..."
	uv sync --extra dev

verify:
	@echo "🔍 Verifying installation..."
	@uv run python -c "import torch; print('✓ PyTorch:', torch.__version__)"
	@uv run python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
	@uv run python -c "import hydra; print('✓ Hydra:', hydra.__version__)"
	@uv run python -c "from ragicamp.core import get_logger; print('✓ RAGiCamp: OK')"
	@echo "✅ All dependencies OK!"

setup: install verify
	@echo "✅ Setup complete! Run: make quick-test"

# ============================================================================
# EXPERIMENTS
# ============================================================================

quick-test:
	@echo "🧪 Running quick test..."
	uv run python -m ragicamp.cli.run experiment=quick_test

baseline:
	@echo "🚀 Running baseline..."
	uv run python -m ragicamp.cli.run experiment=baseline

rag:
	@echo "🔍 Running RAG..."
	@if [ ! -d artifacts/retrievers ]; then \
		echo "⚠️  No index found. Run 'make index' first."; \
		exit 1; \
	fi
	uv run python -m ragicamp.cli.run experiment=rag

# Custom run
run:
	uv run python -m ragicamp.cli.run $(ARGS)

# ============================================================================
# BASELINE STUDY (DirectLLM)
# ============================================================================

baseline-study-test:
	@echo "🧪 Quick test..."
	uv run python scripts/experiments/run_baseline_study.py --quick

baseline-study-direct:
	@echo "🚀 DirectLLM baseline study..."
	uv run python scripts/experiments/run_baseline_study.py --direct-only

baseline-study-full:
	@echo "📊 Full baseline study..."
	uv run python scripts/experiments/run_baseline_study.py --direct-only
	uv run python scripts/experiments/run_rag_baseline_study.py --standard

baseline-study-preview:
	@echo "📋 Preview:"
	uv run python scripts/experiments/run_baseline_study.py --direct-only --dry-run

# ============================================================================
# RAG STUDY (requires 'make index' first)
# ============================================================================

rag-study-test:
	uv run python scripts/experiments/run_rag_baseline_study.py --quick

rag-study-topk:
	uv run python scripts/experiments/run_rag_baseline_study.py --sweep-topk

rag-study-prompts:
	uv run python scripts/experiments/run_rag_baseline_study.py --sweep-prompts

rag-study-datasets:
	uv run python scripts/experiments/run_rag_baseline_study.py --compare-datasets

rag-study-models:
	uv run python scripts/experiments/run_rag_baseline_study.py --compare-models

rag-study-standard:
	uv run python scripts/experiments/run_rag_baseline_study.py --standard

rag-study-full:
	uv run python scripts/experiments/run_rag_baseline_study.py --full

rag-study-preview:
	uv run python scripts/experiments/run_rag_baseline_study.py --standard --dry-run

# ============================================================================
# DATA
# ============================================================================

download-nq:
	uv run python scripts/data/download.py --dataset nq

download-triviaqa:
	uv run python scripts/data/download.py --dataset triviaqa

download-hotpotqa:
	uv run python scripts/data/download.py --dataset hotpotqa

download-all:
	@echo "📚 Downloading datasets..."
	uv run python scripts/data/download.py --all

index:
	@echo "📚 Building index..."
	uv run python scripts/data/index.py --preset small

index-full:
	@echo "📚 Building full index..."
	uv run python scripts/data/index.py --preset full

index-test:
	uv run python scripts/data/index.py --preset test

info:
	@uv run python scripts/data/info.py

# ============================================================================
# ANALYSIS
# ============================================================================

compare:
	@uv run python scripts/eval/compare.py outputs/

report:
	@uv run python scripts/eval/report.py outputs/ --format markdown

report-html:
	@uv run python scripts/eval/report.py outputs/ --format html

mlflow-ui:
	@echo "Open http://localhost:5000"
	uv run mlflow ui --backend-store-uri ./mlruns

# ============================================================================
# TESTING & DEV
# ============================================================================

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow"

test-coverage:
	uv run pytest tests/ --cov=src/ragicamp --cov-report=html

lint:
	@uv run ruff check src/ tests/ || true
	@uv run mypy src/ragicamp --ignore-missing-imports || true

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

# ============================================================================
# CLEAN
# ============================================================================

clean:
	@echo "🧹 Cleaning..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov/ dist/ build/ *.egg-info
	@echo "✅ Done!"

clean-outputs:
	rm -rf outputs/*.json outputs/*/

clean-failed:
	@echo "🔍 Finding failed runs..."
	uv run python scripts/eval/cleanup.py outputs/

clean-failed-delete:
	@echo "🗑️  Deleting failed runs..."
	uv run python scripts/eval/cleanup.py outputs/ --delete

clean-artifacts:
	rm -rf artifacts/

clean-all: clean clean-outputs

clean-phi3-cache:
	@echo "Clearing Phi-3 cache (fixes DynamicCache errors)..."
	rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft/Phi*

clean-hf-cache:
	rm -rf ~/.cache/huggingface/modules/transformers_modules/

# ============================================================================
# SHORTCUTS
# ============================================================================

s: setup
t: test
q: quick-test
b: baseline
r: rag
c: compare
i: info
