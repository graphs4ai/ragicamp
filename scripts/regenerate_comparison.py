#!/usr/bin/env python
"""Regenerate comparison.json from individual experiment directories.

This script reads all metadata.json and summary files from a study directory
and rebuilds the comparison.json file with the latest metrics (including llm_judge).

Usage:
    python scripts/regenerate_comparison.py outputs/comprehensive_baseline
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def regenerate_comparison(study_dir: Path) -> None:
    """Regenerate comparison.json from individual experiment directories."""
    study_dir = Path(study_dir)
    
    if not study_dir.exists():
        print(f"Error: {study_dir} does not exist")
        return
    
    results = []
    
    # Find all metadata.json files
    metadata_files = list(study_dir.glob("*/metadata.json"))
    print(f"Found {len(metadata_files)} experiments in {study_dir}")
    
    for metadata_path in sorted(metadata_files):
        exp_dir = metadata_path.parent
        
        try:
            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Find summary file
            summary_files = list(exp_dir.glob("*_summary.json"))
            if not summary_files:
                print(f"  Warning: No summary for {exp_dir.name}")
                continue
            
            with open(summary_files[0]) as f:
                summary = json.load(f)
            
            # Merge metrics
            metrics = summary.get("overall_metrics", {})
            if not metrics:
                # Try direct metrics in summary
                metrics = {k: v for k, v in summary.items() 
                          if k in ("f1", "exact_match", "bertscore_f1", "bertscore_precision", 
                                   "bertscore_recall", "bleurt", "llm_judge_qa", "llm_judge_qa_correct",
                                   "llm_judge_qa_incorrect", "num_examples", "num_successful", "num_failures")}
            
            # Build experiment record
            exp = {
                "name": metadata.get("name", exp_dir.name),
                "type": metadata.get("type", "unknown"),
                "model": metadata.get("model", "unknown"),
                "prompt": metadata.get("prompt", "unknown"),
                "dataset": metadata.get("dataset", summary.get("dataset_name", "unknown")),
                "quantization": metadata.get("quantization", "unknown"),
                "retriever": metadata.get("retriever"),
                "top_k": metadata.get("top_k"),
                "batch_size": metadata.get("batch_size", 1),
                "num_questions": metadata.get("num_questions", summary.get("num_examples", 0)),
                "results": metrics,
                "duration": metadata.get("duration", 0.0),
                "throughput_qps": metadata.get("throughput_qps", 0.0),
                "timestamp": metadata.get("timestamp", summary.get("timestamp", "")),
            }
            
            results.append(exp)
            
        except Exception as e:
            print(f"  Error processing {exp_dir.name}: {e}")
    
    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Save comparison.json
    output_path = study_dir / "comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "experiments": results,
            "timestamp": datetime.now().isoformat(),
            "regenerated": True,
        }, f, indent=2)
    
    print(f"\nRegenerated {output_path}")
    print(f"  Total experiments: {len(results)}")
    
    # Show metrics summary
    with_llm = sum(1 for r in results if r.get("results", {}).get("llm_judge_qa") is not None)
    print(f"  With llm_judge: {with_llm}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate comparison.json")
    parser.add_argument("study_dir", type=Path, help="Study output directory")
    args = parser.parse_args()
    
    regenerate_comparison(args.study_dir)


if __name__ == "__main__":
    main()

