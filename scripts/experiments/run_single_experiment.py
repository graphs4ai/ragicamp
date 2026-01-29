#!/usr/bin/env python3
"""Run a single experiment in isolation (subprocess wrapper).

This script is called by the study runner to isolate experiments from CUDA crashes.
It reuses all the existing abstractions from cli/study.py.

Usage:
    python run_single_experiment.py --spec-json '{"name": "...", ...}' --output-dir outputs/study
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Reuse existing abstractions - NO DUPLICATION
from ragicamp.execution.runner import ExpSpec, run_spec
from ragicamp.cli.study import create_judge_model


def main():
    parser = argparse.ArgumentParser(description="Run single experiment in subprocess")
    parser.add_argument("--spec-json", required=True, help="Experiment spec as JSON string")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--metrics", default="f1,exact_match", help="Comma-separated metrics")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--llm-judge-config", help="LLM judge config as JSON string")
    
    args = parser.parse_args()
    
    spec_dict = json.loads(args.spec_json)
    metrics = args.metrics.split(",")
    
    # Parse llm_judge config if provided
    llm_judge_config = None
    judge_model = None
    if args.llm_judge_config:
        llm_judge_config = json.loads(args.llm_judge_config)
        # Create the judge model in this subprocess
        judge_model = create_judge_model(llm_judge_config)
    
    # Convert dict back to ExpSpec dataclass
    spec = ExpSpec(
        name=spec_dict["name"],
        exp_type=spec_dict["exp_type"],
        model=spec_dict["model"],
        dataset=spec_dict["dataset"],
        prompt=spec_dict["prompt"],
        quant=spec_dict.get("quant", "4bit"),
        retriever=spec_dict.get("retriever"),
        top_k=spec_dict.get("top_k", 5),
        query_transform=spec_dict.get("query_transform"),
        reranker=spec_dict.get("reranker"),
        reranker_model=spec_dict.get("reranker_model"),
        batch_size=spec_dict.get("batch_size", 8),
        min_batch_size=spec_dict.get("min_batch_size", 1),
    )
    
    # Reuse the existing run_spec function with subprocess disabled
    # (we're already in a subprocess, no need to nest)
    status = run_spec(
        spec=spec,
        limit=args.limit,
        metrics=metrics,
        out=args.output_dir,
        judge_model=judge_model,
        llm_judge_config=llm_judge_config,
        force=True,  # Always retry in subprocess
        use_subprocess=False,  # We ARE the subprocess
    )
    
    print(f"\n__RESULT__:{json.dumps({'status': status})}")
    
    if status in ("failed", "crashed"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
