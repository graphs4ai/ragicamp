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
	@echo "  make install          Install dependencies"
	@echo "  make setup            Full setup (install + verify)"
	@echo ""
	@echo "🚀 EXPERIMENTS (Hydra - RECOMMENDED)"
	@echo "  make quick-test       Quick smoke test (10 examples)"
	@echo "  make baseline         DirectLLM baseline evaluation"
	@echo "  make rag              RAG evaluation with retrieval"
	@echo "  make run ARGS='...'   Run with custom Hydra args"
	@echo ""
	@echo "📊 BASELINE STUDY (Systematic experiments)"
	@echo "  make baseline-study-test    Quick test (verify setup)"
	@echo "  make baseline-study-direct  Run DirectLLM experiments"
	@echo "  make baseline-study-rag     Run RAG experiments"
	@echo "  make baseline-study-full    Run complete study"
	@echo "  make sweep-prompts          Compare prompt styles"
	@echo "  make sweep-topk             Compare top_k values"
	@echo "  make sweep-datasets         Compare datasets"
	@echo "  make sweep-models           Compare models"
	@echo ""
	@echo "📚 DATA"
	@echo "  make download-all     Download all datasets"
	@echo "  make index            Index corpus (small, for testing)"
	@echo "  make index-full       Index full Wikipedia corpus"
	@echo "  make info             Show data/artifacts info"
	@echo ""
	@echo "📊 ANALYSIS"
	@echo "  make compare          Compare experiment results"
	@echo "  make report           Generate evaluation report"
	@echo ""
	@echo "🧪 TESTING"
	@echo "  make test             Run all tests"
	@echo "  make test-fast        Run fast tests only"
	@echo ""
	@echo "🔧 DEVELOPMENT"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code"
	@echo "  make clean            Clean generated files"
	@echo ""
	@echo "💡 QUICK START"
	@echo "  1. make setup         (first time)"
	@echo "  2. make quick-test    (verify everything works)"
	@echo "  3. make baseline      (run evaluation)"
	@echo ""
	@echo "📖 See CHEATSHEET.md for more examples"
	@echo ""

# ============================================================================
# SETUP
# ============================================================================

install:
	@echo "📦 Installing dependencies..."
	uv sync

install-dev:
	@echo "📦 Installing with dev tools..."
	uv sync --extra dev

install-all:
	@echo "📦 Installing all dependencies..."
	uv sync --extra dev --extra viz

verify:
	@echo "🔍 Verifying installation..."
	@uv run python -c "import torch; print('✓ PyTorch:', torch.__version__)"
	@uv run python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
	@uv run python -c "import hydra; print('✓ Hydra:', hydra.__version__)"
	@uv run python -c "from ragicamp.core import get_logger; print('✓ RAGiCamp core: OK')"
	@echo ""
	@echo "✅ All dependencies installed correctly!"

setup: install verify
	@echo ""
	@echo "✅ Setup complete! Run: make quick-test"

# ============================================================================
# EXPERIMENTS (Hydra-powered)
# ============================================================================

# Quick smoke test (10 examples, fast metrics)
quick-test:
	@echo "🧪 Running quick test..."
	uv run python -m ragicamp.cli.run experiment=quick_test

# Baseline evaluation (DirectLLM, no retrieval)
baseline:
	@echo "🚀 Running baseline evaluation..."
	uv run python -m ragicamp.cli.run experiment=baseline

# RAG evaluation (with retrieval)
rag:
	@echo "🔍 Running RAG evaluation..."
	@if [ ! -d artifacts/retrievers ]; then \
		echo "⚠️  No index found. Run 'make index' first."; \
		exit 1; \
	fi
	uv run python -m ragicamp.cli.run experiment=rag

# Compare models (multi-run)
compare-models:
	@echo "📊 Comparing models..."
	uv run python -m ragicamp.cli.run --multirun \
		model=gemma_2b_4bit,phi3 \
		experiment=baseline \
		evaluation=quick

# Custom run with args
run:
	uv run python -m ragicamp.cli.run $(ARGS)

# Show config without running
show-config:
	uv run python -m ragicamp.cli.run --cfg job $(ARGS)

# ============================================================================
# BASELINE STUDY (Systematic experiments)
# ============================================================================

# Quick test - verify everything works
baseline-study-test:
	@echo "🧪 Running baseline study quick test..."
	uv run python scripts/experiments/run_baseline_study.py --quick

# DirectLLM experiments only (faster, no index needed)
baseline-study-direct:
	@echo "🚀 Running DirectLLM baseline study..."
	uv run python scripts/experiments/run_baseline_study.py --direct-only

# Two-phase baseline: Generate predictions only (Phase 1)
# Saves predictions to disk, no metrics. Safe for unstable environments.
baseline-generate:
	@echo "🚀 Phase 1: Generating predictions..."
	uv run python -m ragicamp.cli.run \
		experiment=baseline_study_direct \
		evaluation.mode=generate

# Two-phase baseline: Compute metrics only (Phase 2)
# Requires predictions from Phase 1. Pass PREDS_PATH=path/to/predictions.json
baseline-evaluate:
	@echo "📊 Phase 2: Computing metrics..."
	@if [ -z "$(PREDS_PATH)" ]; then \
		echo "❌ Error: PREDS_PATH not set. Usage: make baseline-evaluate PREDS_PATH=outputs/.../predictions_raw.json"; \
		exit 1; \
	fi
	uv run python -m ragicamp.cli.run \
		experiment=baseline_study_direct \
		evaluation.mode=evaluate \
		evaluation.predictions_path=$(PREDS_PATH)

# RAG experiments only (needs index)
baseline-study-rag:
	@echo "🔍 Running RAG baseline study..."
	uv run python scripts/experiments/run_baseline_study.py --rag-only

# Full baseline study (DirectLLM + RAG)
baseline-study-full:
	@echo "📊 Running full baseline study..."
	uv run python scripts/experiments/run_baseline_study.py --full

# Dry run - show what would be executed
baseline-study-preview:
	@echo "📋 Preview of baseline study commands:"
	uv run python scripts/experiments/run_baseline_study.py --full --dry-run

# Sweep: Compare prompt styles
sweep-prompts:
	@echo "📝 Sweeping prompt styles..."
	uv run python -m ragicamp.cli.run --multirun \
		experiment=baseline_study_direct \
		prompt=concise,sentence,explained \
		evaluation=quick

# Sweep: Compare top_k values
sweep-topk:
	@echo "🔄 Sweeping top_k values..."
	uv run python -m ragicamp.cli.run --multirun \
		experiment=baseline_study_rag \
		agent.top_k=1,3,5,10 \
		evaluation=quick

# Sweep: Compare datasets
sweep-datasets:
	@echo "📚 Sweeping datasets..."
	uv run python -m ragicamp.cli.run --multirun \
		experiment=baseline_study_direct \
		dataset=nq,triviaqa,hotpotqa \
		evaluation=quick

# Sweep: Compare models
sweep-models:
	@echo "🤖 Sweeping models..."
	uv run python -m ragicamp.cli.run --multirun \
		experiment=baseline_study_direct \
		model=gemma_2b_4bit,phi3 \
		evaluation=quick

# ============================================================================
# LEGACY EXPERIMENTS (old YAML configs - for compatibility)
# ============================================================================

eval-baseline-quick:
	@echo "🚀 Running baseline (legacy config)..."
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_baseline_gemma2b_quick.yaml

eval-baseline-full:
	@echo "🚀 Running full baseline (legacy config)..."
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_baseline_gemma2b_full.yaml

eval-rag-legacy:
	@echo "🔍 Running RAG (legacy config)..."
	uv run python experiments/scripts/run_experiment.py \
		--config experiments/configs/nq_fixed_rag_gemma2b.yaml

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Download datasets
download-nq:
	uv run python scripts/data/download.py --dataset nq

download-triviaqa:
	uv run python scripts/data/download.py --dataset triviaqa

download-hotpotqa:
	uv run python scripts/data/download.py --dataset hotpotqa

download-all:
	@echo "📚 Downloading all datasets..."
	uv run python scripts/data/download.py --all

# Index corpus
index:
	@echo "📚 Indexing corpus (small, for testing)..."
	uv run python scripts/data/index.py --preset small

index-full:
	@echo "📚 Indexing full corpus (this takes a while)..."
	uv run python scripts/data/index.py --preset full

index-test:
	@echo "📚 Indexing tiny corpus (for quick tests)..."
	uv run python scripts/data/index.py --preset test

# Show info
info:
	@uv run python scripts/data/info.py

list-datasets:
	@uv run python scripts/data/info.py --datasets

list-artifacts:
	@uv run python scripts/data/info.py --artifacts

# ============================================================================
# ANALYSIS
# ============================================================================

compare:
	@echo "📊 Comparing experiments..."
	@uv run python scripts/eval/compare.py outputs/

# Compare baseline study results with visualization
compare-baseline:
	@echo "📊 Comparing baseline study results..."
	@uv run python scripts/analysis/compare_baseline.py outputs/

# Compare and export to CSV
compare-csv:
	@echo "📊 Exporting results to CSV..."
	@uv run python scripts/analysis/compare_baseline.py outputs/ --csv outputs/comparison.csv

report:
	@echo "📝 Generating report..."
	@uv run python scripts/eval/report.py outputs/ --format markdown

report-html:
	@echo "📝 Generating HTML report..."
	@uv run python scripts/eval/report.py outputs/ --format html

# ============================================================================
# TESTING
# ============================================================================

test:
	@echo "🧪 Running all tests..."
	uv run pytest tests/ -v

test-fast:
	@echo "⚡ Running fast tests..."
	uv run pytest tests/ -v -m "not slow"

test-coverage:
	@echo "📊 Running tests with coverage..."
	uv run pytest tests/ --cov=src/ragicamp --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

test-core:
	uv run pytest tests/test_config.py tests/test_factory.py tests/test_agents.py -v

# ============================================================================
# DEVELOPMENT
# ============================================================================

lint:
	@echo "🔍 Running linters..."
	@uv run flake8 src/ tests/ --max-line-length=100 || true
	@uv run mypy src/ragicamp --ignore-missing-imports || true

format:
	@echo "✨ Formatting code..."
	uv run black src/ tests/ scripts/ --line-length 100
	uv run isort src/ tests/ scripts/ --profile black
	@echo "✅ Done!"

clean:
	@echo "🧹 Cleaning..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov/ dist/ build/ *.egg-info
	@echo "✅ Cleaned!"

clean-outputs:
	@echo "🧹 Cleaning outputs..."
	rm -rf outputs/*.json outputs/*/
	@echo "✅ Outputs cleaned!"

clean-artifacts:
	@echo "🧹 Cleaning artifacts..."
	rm -rf artifacts/
	@echo "✅ Artifacts cleaned!"

clean-all: clean clean-outputs
	@echo "✅ All cleaned!"

# ============================================================================
# CONFIGURATION
# ============================================================================

validate-config:
	uv run python scripts/utils/validate_config.py $(CONFIG)

validate-all-configs:
	@echo "🔍 Validating all configs..."
	@uv run python scripts/utils/validate_config.py experiments/configs/*.yaml
	@uv run python scripts/utils/validate_config.py conf/experiment/*.yaml

# ============================================================================
# MLFLOW
# ============================================================================

mlflow-ui:
	@echo "🔍 Starting MLflow UI..."
	@echo "Open http://localhost:5000 in your browser"
	uv run mlflow ui --backend-store-uri ./mlruns

# ============================================================================
# SHORTCUTS
# ============================================================================

# Aliases for convenience
s: setup
t: test
q: quick-test
b: baseline
r: rag
c: compare
i: info
