# RAGiCamp Makefile
# Run `make help` for available commands

.PHONY: help env install setup test lint format clean

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                      RAGiCamp Commands                            ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 SETUP"
	@echo "  make env              Show how to load env vars (HF_TOKEN, OPENAI_API_KEY)"
	@echo "  make setup            Full setup (install + verify)"
	@echo "  make install          Install dependencies only"
	@echo ""
	@echo "🚀 QUICK START"
	@echo "  make quick-test       Smoke test (10 examples)"
	@echo "  make baseline         DirectLLM evaluation"
	@echo "  make rag              RAG evaluation (needs index)"
	@echo ""
	@echo "📊 COMPREHENSIVE STUDY (100 examples, MLflow, JSON)"
	@echo "  make study-quick             Quick verification"
	@echo "  make study-direct            DirectLLM: 2 models × 3 datasets × 3 prompts"
	@echo "  make study-rag               FixedRAG: varies top_k, retrievers, models"
	@echo "  make study-full              Full study (Direct + RAG)"
	@echo "  make study-full-production   Full study, no example cap"
	@echo "  make study-preview           Preview commands (dry-run)"
	@echo ""
	@echo "📊 BASELINE STUDIES (legacy)"
	@echo "  make baseline-study-direct   DirectLLM study (2 models × 3 datasets)"
	@echo "  make baseline-study-full     Full study (DirectLLM + RAG)"
	@echo "  make baseline-study-preview  Preview commands (dry-run)"
	@echo ""
	@echo "🔍 RAG STUDIES (run 'make index' first)"
	@echo "  make rag-study-standard  Standard RAG study"
	@echo "  make rag-study-topk      Sweep top_k values"
	@echo "  make rag-study-prompts   Compare prompts"
	@echo ""
	@echo "📚 DATA & INDEXING"
	@echo "  make download-all     Download datasets"
	@echo "  make index            Build 1 index (Simple Wiki + MiniLM)"
	@echo "  make index-standard   Build 2 indexes (MiniLM + MPNet)"
	@echo "  make index-extended   Build 6 indexes (3 embeddings × 2 chunk sizes)"
	@echo "  make index-all        Build all (including full Wikipedia)"
	@echo "  make index-preview    Preview index builds (dry-run)"
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

env:
	@echo "🔑 Loading environment variables..."
	@echo "Run: source ~/setup_env.sh"
	@echo ""
	@echo "Or copy-paste:"
	@echo "  source ~/setup_env.sh && make quick-test"

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
# COMPREHENSIVE STUDY (NEW - 100 examples, MLflow, structured JSON)
# ============================================================================

study-quick:
	@echo "🧪 Quick study test..."
	uv run python scripts/experiments/run_comprehensive_study.py --quick

study-direct:
	@echo "📊 DirectLLM study (100 examples × 3 datasets × 2 models × 3 prompts)..."
	uv run python scripts/experiments/run_comprehensive_study.py --direct-only

study-rag:
	@echo "📊 FixedRAG study (100 examples, varies top_k, retrievers, models)..."
	uv run python scripts/experiments/run_comprehensive_study.py --rag-only

study-full:
	@echo "📊 Full comprehensive study (Direct + RAG)..."
	uv run python scripts/experiments/run_comprehensive_study.py --full

study-full-production:
	@echo "📊 Full study WITHOUT 100-example cap (production run)..."
	uv run python scripts/experiments/run_comprehensive_study.py --full --no-limit

study-preview:
	@echo "📋 Preview commands (dry-run):"
	uv run python scripts/experiments/run_comprehensive_study.py --full --dry-run

# ============================================================================
# BASELINE STUDY (DirectLLM) - Legacy
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

# Single index (quick setup)
index:
	@echo "📚 Building single index (Simple Wiki + MiniLM + recursive_512)..."
	uv run python scripts/data/build_all_indexes.py --minimal

# Build multiple indexes for RAG study
index-standard:
	@echo "📚 Building standard indexes (4: 2 embeddings × 2 strategies)..."
	@echo "   - MiniLM/MPNet × recursive/paragraph @ 512 chars"
	uv run python scripts/data/build_all_indexes.py --standard

index-extended:
	@echo "📚 Building extended indexes (12: 2 embeddings × 6 chunk configs)..."
	@echo "   - MiniLM/MPNet × recursive/paragraph/fixed @ 256-1024 chars"
	uv run python scripts/data/build_all_indexes.py --extended

index-all:
	@echo "📚 Building ALL indexes (36: 3 embeddings × 6 configs × 2 corpora)..."
	@echo "   - MiniLM/MPNet/E5 × recursive/paragraph/fixed @ various sizes"
	@echo "   - Simple Wikipedia + Full English Wikipedia"
	uv run python scripts/data/build_all_indexes.py --all

index-test:
	@echo "📚 Test index (100 docs only)..."
	uv run python scripts/data/build_all_indexes.py --minimal --max-docs 100

index-preview:
	@echo "📋 Preview index builds (dry-run):"
	uv run python scripts/data/build_all_indexes.py --all --dry-run

# Custom index builds
index-custom:
	@echo "📚 Custom index build example:"
	@echo "   uv run python scripts/data/build_all_indexes.py --embeddings minilm mpnet --chunk-configs recursive_512 paragraph_1024"

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
