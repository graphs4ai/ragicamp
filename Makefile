# RAGiCamp Makefile

.PHONY: help install index-simple index-full run-baseline-simple run-baseline-full evaluate compare

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
	@echo "  make index-simple            Build 1 small index (500 docs)"
	@echo "  make run-baseline-simple     Run quick baseline (10 questions)"
	@echo "  make run-baseline-simple-hf  HF models only, no OpenAI costs"
	@echo ""
	@echo "Full Experiments:"
	@echo "  make index-full           Build all indexes for baseline"
	@echo "  make run-baseline-full    Run full baseline (100+ questions)"
	@echo ""
	@echo "Re-evaluate and compare:"
	@echo "  make evaluate PATH=outputs/simple METRICS=bertscore,bleurt"
	@echo "  make compare PATH=outputs/simple              # Show comparison table"
	@echo "  make compare PATH=outputs/simple --filter rag # Filter by type"
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

run-baseline-simple-hf:
	@echo "🚀 Running simple baseline - HF only, no OpenAI (10 questions)..."
	uv run python scripts/experiments/run_study.py conf/study/simple_hf.yaml

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
# EVALUATION (re-run metrics on existing predictions)
# ============================================================================

# Re-evaluate with specific metrics: make evaluate PATH=outputs/simple METRICS=bertscore,bleurt
evaluate:
	@echo "📊 Re-evaluating predictions..."
	uv run python scripts/experiments/evaluate_predictions.py $(PATH) --metrics $(or $(METRICS),all)

# Compare results in a table: make compare PATH=outputs/simple
compare:
	@python scripts/eval/compare.py $(or $(PATH),outputs/simple)

# Examples:
#   make evaluate PATH=outputs/simple_hf METRICS=bertscore
#   make evaluate PATH=outputs/simple_hf/direct_hf_google_gemma2bit_default_nq METRICS=llm_judge
#   make evaluate PATH=outputs/simple METRICS=all

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
