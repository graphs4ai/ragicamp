#!/bin/bash
# Compute LLM-as-Judge scores for all predictions in a study
#
# Usage:
#   ./scripts/compute_llm_judge.sh outputs/comprehensive_baseline
#   ./scripts/compute_llm_judge.sh outputs/comprehensive_baseline gpt-4o
#   ./scripts/compute_llm_judge.sh outputs/comprehensive_baseline gpt-4o-mini 50
#
# Requires: OPENAI_API_KEY environment variable

set -e

STUDY_DIR="${1:-outputs/comprehensive_baseline}"
JUDGE_MODEL="${2:-gpt-4o-mini}"
MAX_CONCURRENT="${3:-50}"  # 50 concurrent API calls

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Find all predictions files
PREDICTIONS=$(find "$STUDY_DIR" -name "*_predictions.json" 2>/dev/null)
TOTAL=$(echo "$PREDICTIONS" | wc -l)

if [ -z "$PREDICTIONS" ]; then
    echo "No predictions files found in $STUDY_DIR"
    exit 1
fi

echo "=============================================="
echo "LLM-as-Judge Evaluation"
echo "=============================================="
echo "Study dir:      $STUDY_DIR"
echo "Judge model:    $JUDGE_MODEL"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Files found:    $TOTAL"
echo "=============================================="
echo ""

# Estimate cost
QUESTIONS_PER_FILE=100
TOTAL_QUESTIONS=$((TOTAL * QUESTIONS_PER_FILE))
if [ "$JUDGE_MODEL" = "gpt-4o-mini" ]; then
    COST_EST="~\$$(echo "scale=2; $TOTAL_QUESTIONS * 0.0001" | bc)"
else
    COST_EST="~\$$(echo "scale=2; $TOTAL_QUESTIONS * 0.01" | bc)"
fi

echo "Estimated cost: $COST_EST for ~$TOTAL_QUESTIONS questions"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Process each file
COUNT=0
FAILED=0

for PRED_FILE in $PREDICTIONS; do
    COUNT=$((COUNT + 1))
    
    # Get experiment name from path
    EXP_NAME=$(basename "$(dirname "$PRED_FILE")")
    OUTPUT_FILE="${PRED_FILE%.json}_llm_judge.json"
    
    echo ""
    echo "[$COUNT/$TOTAL] $EXP_NAME"
    
    # Skip if already computed (check if llm_judge_qa is in predictions file)
    if grep -q '"llm_judge_qa"' "$PRED_FILE" 2>/dev/null; then
        echo "  → Already has llm_judge_qa, skipping"
        continue
    fi
    
    # Compute LLM judge (updates predictions file in-place)
    if uv run ragicamp evaluate "$PRED_FILE" \
        --metrics llm_judge_qa \
        --judge-model "$JUDGE_MODEL" \
        --max-concurrent "$MAX_CONCURRENT" 2>&1; then
        echo "  → Done (updated predictions file)"
    else
        echo "  → FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo "Complete!"
echo "  Processed: $COUNT files"
echo "  Failed:    $FAILED files"
echo "=============================================="

# Aggregate results if we have any
if [ $FAILED -lt $TOTAL ]; then
    echo ""
    echo "Aggregating LLM judge scores..."
    
    # Simple aggregation from summary files
    python3 - "$STUDY_DIR" << 'EOF'
import json
from pathlib import Path
import sys

study_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/comprehensive_baseline"

results = []
for f in Path(study_dir).rglob("*_summary.json"):
    try:
        with open(f) as fp:
            data = json.load(fp)
            exp_name = f.parent.name
            metrics = data.get("overall_metrics", data)
            score = metrics.get("llm_judge_qa")
            if score is not None:
                results.append((exp_name, score))
    except:
        pass

if results:
    results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 by LLM Judge:")
    for name, score in results[:10]:
        print(f"  {score:.4f}  {name[:60]}")
    
    avg = sum(s for _, s in results) / len(results)
    print(f"\nAverage LLM Judge: {avg:.4f}")
else:
    print("No LLM judge scores found yet.")
EOF
fi

