#!/usr/bin/env python3
"""Clean up failed/incomplete experiment runs.

Usage:
    python scripts/eval/cleanup.py outputs/           # Preview what would be deleted
    python scripts/eval/cleanup.py outputs/ --delete  # Actually delete
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def is_valid_run(run_dir: Path) -> Tuple[bool, str]:
    """Check if a run directory contains valid results.
    
    Returns (is_valid, reason) tuple.
    """
    # Must have summary file
    summary_files = list(run_dir.glob("*summary*.json"))
    if not summary_files:
        return False, "No summary file"
    
    # Check summary file has metrics
    for summary_file in summary_files:
        try:
            with open(summary_file) as f:
                data = json.load(f)
            
            # Check for metrics
            metrics = data.get("overall_metrics", data.get("metrics", {}))
            if not metrics:
                return False, "No metrics in summary"
            
            # Check for at least basic metrics
            has_f1 = "f1" in metrics or "f1" in data
            has_em = "exact_match" in metrics or "exact_match" in data
            if not (has_f1 or has_em):
                return False, "Missing basic metrics (f1/exact_match)"
            
            # Check num_examples
            num_examples = data.get("num_examples", 0)
            if num_examples == 0:
                return False, "Zero examples processed"
                
        except json.JSONDecodeError:
            return False, "Corrupted JSON"
        except Exception as e:
            return False, f"Error reading: {e}"
    
    return True, "OK"


def find_run_directories(base_path: Path) -> List[Path]:
    """Find all run directories (containing .hydra folder or summary files)."""
    run_dirs = []
    
    # Find directories with .hydra (Hydra output dirs)
    for hydra_dir in base_path.rglob(".hydra"):
        run_dirs.append(hydra_dir.parent)
    
    # Also find directories with summary files but no .hydra
    for summary_file in base_path.rglob("*summary*.json"):
        parent = summary_file.parent
        if parent not in run_dirs:
            # Skip if it's the top-level outputs dir
            if parent != base_path:
                run_dirs.append(parent)
    
    return sorted(set(run_dirs))


def cleanup_outputs(base_path: Path, delete: bool = False, keep_latest: int = 0) -> dict:
    """Clean up failed runs.
    
    Args:
        base_path: Base outputs directory
        delete: If True, actually delete. If False, just preview.
        keep_latest: Keep the N most recent valid runs per config (0 = keep all valid)
    
    Returns:
        Summary dict with counts
    """
    run_dirs = find_run_directories(base_path)
    
    valid_runs = []
    invalid_runs = []
    
    for run_dir in run_dirs:
        is_valid, reason = is_valid_run(run_dir)
        if is_valid:
            valid_runs.append((run_dir, reason))
        else:
            invalid_runs.append((run_dir, reason))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Cleanup Summary: {base_path}")
    print(f"{'='*60}")
    print(f"Total run directories: {len(run_dirs)}")
    print(f"  ✅ Valid:   {len(valid_runs)}")
    print(f"  ❌ Invalid: {len(invalid_runs)}")
    
    if invalid_runs:
        print(f"\n{'='*60}")
        print("Invalid/Failed Runs:")
        print(f"{'='*60}")
        
        total_size = 0
        for run_dir, reason in invalid_runs:
            # Calculate size
            size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
            total_size += size
            size_mb = size / (1024 * 1024)
            
            rel_path = run_dir.relative_to(base_path) if run_dir.is_relative_to(base_path) else run_dir
            print(f"  ❌ {rel_path}")
            print(f"     Reason: {reason} ({size_mb:.2f} MB)")
        
        print(f"\nTotal space to recover: {total_size / (1024*1024):.2f} MB")
        
        if delete:
            print(f"\n🗑️  Deleting {len(invalid_runs)} failed runs...")
            for run_dir, reason in invalid_runs:
                try:
                    shutil.rmtree(run_dir)
                    print(f"  Deleted: {run_dir}")
                except Exception as e:
                    print(f"  Error deleting {run_dir}: {e}")
            print("✅ Cleanup complete!")
        else:
            print(f"\n⚠️  Dry run - no files deleted.")
            print(f"   Run with --delete to remove these directories.")
    else:
        print("\n✅ No failed runs to clean up!")
    
    # Also check for empty date directories
    empty_dirs = []
    for date_dir in base_path.iterdir():
        if date_dir.is_dir() and date_dir.name not in ["multirun"]:
            subdirs = list(date_dir.iterdir())
            if not subdirs:
                empty_dirs.append(date_dir)
    
    if empty_dirs:
        print(f"\n📁 Empty date directories: {len(empty_dirs)}")
        for d in empty_dirs:
            print(f"  {d.name}/")
        if delete:
            for d in empty_dirs:
                d.rmdir()
            print("  Removed empty directories.")
    
    return {
        "total": len(run_dirs),
        "valid": len(valid_runs),
        "invalid": len(invalid_runs),
        "deleted": len(invalid_runs) if delete else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean up failed/incomplete experiment runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview what would be deleted
    python scripts/eval/cleanup.py outputs/
    
    # Actually delete failed runs
    python scripts/eval/cleanup.py outputs/ --delete
    
    # Clean specific date
    python scripts/eval/cleanup.py outputs/2025-12-11/ --delete
        """,
    )
    
    parser.add_argument("path", type=Path, help="Outputs directory to clean")
    parser.add_argument("--delete", "-d", action="store_true", 
                        help="Actually delete failed runs (default: dry run)")
    
    args = parser.parse_args()
    
    if not args.path.exists():
        print(f"Error: {args.path} does not exist")
        sys.exit(1)
    
    cleanup_outputs(args.path, delete=args.delete)


if __name__ == "__main__":
    main()
