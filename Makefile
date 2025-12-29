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
	@echo "  make evaluate DIR=outputs/simple METRICS=bertscore,bleurt"
	@echo "  make compare DIR=outputs/simple              # Show comparison table"
	@echo "  make compare DIR=outputs/simple SORT=f1      # Sort by metric"
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
# COMPREHENSIVE BASELINE (all models, all datasets)
# ============================================================================

# Build all 6 indexes for comprehensive study
index-comprehensive:
	@echo "📚 Building comprehensive indexes (6 variations)..."
	@echo "This will take 2-4 hours for English Wikipedia..."
	uv run python scripts/data/build_all_indexes.py \
		--corpora wiki_simple wiki_en \
		--embeddings minilm mpnet e5 \
		--chunk-configs recursive_512

# Build just Simple Wikipedia indexes (faster, for testing)
index-simple-wiki:
	@echo "📚 Building Simple Wikipedia indexes (3 variations)..."
	uv run python scripts/data/build_all_indexes.py \
		--corpora wiki_simple \
		--embeddings minilm mpnet e5 \
		--chunk-configs recursive_512 recursive_1024

# Run comprehensive baseline study
run-comprehensive:
	@echo "🚀 Running comprehensive baseline (full study)..."
	@echo "This will take 10-20 hours with all models..."
	@echo "Using --skip-existing to resume safely..."
	uv run python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --skip-existing

# Run with timeout per experiment (e.g. 2 hours max)
run-comprehensive-safe:
	@echo "🚀 Running comprehensive baseline with 2h timeout per experiment..."
	uv run python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --skip-existing --timeout 7200

# ============================================================================
# EVALUATION (re-run metrics on existing predictions)
# ============================================================================

# Re-evaluate with specific metrics: make evaluate PATH=outputs/simple METRICS=bertscore,bleurt
evaluate:
	@echo "📊 Re-evaluating predictions..."
	uv run python scripts/experiments/evaluate_predictions.py $(DIR) --metrics $(or $(METRICS),all)

# Compare results in a table: make compare DIR=outputs/simple
compare:
	uv run python scripts/eval/compare.py $(or $(DIR),outputs/simple) $(if $(SORT),--sort $(SORT),)

# Examples:
#   make evaluate PATH=outputs/simple_hf METRICS=bertscore
#   make evaluate PATH=outputs/simple_hf/direct_hf_google_gemma2bit_default_nq METRICS=llm_judge
#   make evaluate PATH=outputs/simple METRICS=all

# ============================================================================
# DEV
# ============================================================================

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src/ragicamp --cov-report=term-missing

lint:
	uv run black --check src/ tests/ scripts/ --line-length 100
	uv run isort --check-only src/ tests/ scripts/ --profile black

format:
	uv run black src/ tests/ scripts/ --line-length 100
	uv run isort src/ tests/ scripts/ --profile black

validate-all-configs:
	@echo "Validating legacy configs..."
	@for config in experiments/configs/*.yaml; do \
		echo "  → $$config"; \
		uv run python scripts/utils/validate_config.py "$$config"; \
	done
	@echo "Validating Hydra configs..."
	uv run python -c "from hydra import compose, initialize_config_dir; from pathlib import Path; conf_dir = Path('conf').absolute(); initialize_config_dir(version_base=None, config_dir=str(conf_dir)).__enter__(); cfg = compose(config_name='config'); print('✅ All configs valid!')"

# Pre-push check (run before pushing to CI)
pre-push: format lint test
	@echo "✅ All checks passed - ready to push!"

clean:
	rm -rf outputs/
	@echo "✓ Cleaned"
