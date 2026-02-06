#!/usr/bin/env python3
"""Migrate old prediction format to new format.

Old format:
- metadata.doc_scores: list[float] (just scores)
- No retrieved_docs field

New format:
- retrieved_docs: list[dict] with rank, doc_id, content, score, retrieval_score, retrieval_rank
- metadata: num_docs, top_k

Usage:
    python scripts/migrate_predictions.py outputs/study_name/
    python scripts/migrate_predictions.py outputs/study_name/experiment_name/
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


def is_old_format(prediction: dict) -> bool:
    """Check if prediction uses old format."""
    # Old format has doc_scores in metadata but no retrieved_docs
    metadata = prediction.get("metadata", {})
    has_doc_scores = "doc_scores" in metadata
    has_retrieved_docs = "retrieved_docs" in prediction and len(prediction["retrieved_docs"]) > 0
    return has_doc_scores and not has_retrieved_docs


def migrate_prediction(prediction: dict) -> dict:
    """Migrate a single prediction from old to new format."""
    if not is_old_format(prediction):
        return prediction  # Already new format or not migratable
    
    metadata = prediction.get("metadata", {})
    doc_scores = metadata.pop("doc_scores", [])
    
    # Create retrieved_docs from doc_scores
    # We don't have the original doc content, so we create placeholder entries
    retrieved_docs = []
    for i, score in enumerate(doc_scores):
        retrieved_docs.append({
            "rank": i + 1,
            "doc_id": f"unknown_doc_{i}",
            "content": "[Content not available - migrated from old format]",
            "score": score,
            "retrieval_score": score,
            "retrieval_rank": i + 1,
        })
    
    # Update prediction with new format
    prediction["retrieved_docs"] = retrieved_docs
    
    # Ensure metadata has expected fields
    if "num_docs" not in metadata and doc_scores:
        metadata["num_docs"] = len(doc_scores)
    if "top_k" not in metadata and doc_scores:
        metadata["top_k"] = len(doc_scores)
    
    prediction["metadata"] = metadata
    
    return prediction


def migrate_experiment(exp_path: Path, dry_run: bool = False) -> dict:
    """Migrate predictions for a single experiment."""
    predictions_file = exp_path / "predictions.json"
    
    if not predictions_file.exists():
        return {"status": "skipped", "reason": "no predictions.json"}
    
    try:
        with open(predictions_file) as f:
            data = json.load(f)
    except Exception as e:
        return {"status": "error", "reason": str(e)}
    
    predictions = data.get("predictions", [])
    if not predictions:
        return {"status": "skipped", "reason": "no predictions"}
    
    # Check if migration needed
    old_format_count = sum(1 for p in predictions if is_old_format(p))
    if old_format_count == 0:
        return {"status": "skipped", "reason": "already new format"}
    
    if dry_run:
        return {
            "status": "would_migrate",
            "old_format_count": old_format_count,
            "total_predictions": len(predictions),
        }
    
    # Backup original file
    backup_path = predictions_file.with_suffix(".json.bak")
    shutil.copy2(predictions_file, backup_path)
    
    # Migrate predictions
    migrated_predictions = [migrate_prediction(p) for p in predictions]
    data["predictions"] = migrated_predictions
    data["_migration"] = {
        "migrated_at": datetime.now().isoformat(),
        "from_format": "old_doc_scores",
        "to_format": "retrieved_docs",
        "predictions_migrated": old_format_count,
    }
    
    # Save migrated data
    with open(predictions_file, "w") as f:
        json.dump(data, f, indent=2)
    
    return {
        "status": "migrated",
        "predictions_migrated": old_format_count,
        "total_predictions": len(predictions),
        "backup": str(backup_path),
    }


def migrate_study(study_path: Path, dry_run: bool = False) -> dict:
    """Migrate all experiments in a study."""
    results = {}
    
    for exp_dir in sorted(study_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        result = migrate_experiment(exp_dir, dry_run=dry_run)
        results[exp_dir.name] = result
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Migrate old prediction format to new format")
    parser.add_argument("path", type=Path, help="Path to study or experiment directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    args = parser.parse_args()
    
    path = args.path
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return 1
    
    # Check if this is a single experiment or a study directory
    is_experiment = (path / "predictions.json").exists()
    
    if is_experiment:
        print(f"Migrating single experiment: {path}")
        result = migrate_experiment(path, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    else:
        print(f"Migrating study: {path}")
        if args.dry_run:
            print("(DRY RUN - no changes will be made)")
        
        results = migrate_study(path, dry_run=args.dry_run)
        
        # Summary
        statuses = {}
        for exp_name, result in results.items():
            status = result.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        
        print(f"\n{'='*60}")
        print("MIGRATION SUMMARY")
        print(f"{'='*60}")
        for status, count in sorted(statuses.items()):
            print(f"  {status}: {count}")
        print(f"{'='*60}")
        
        # Show details for migrated experiments
        migrated = [(n, r) for n, r in results.items() if r.get("status") in ("migrated", "would_migrate")]
        if migrated:
            print("\nMigrated experiments:")
            for name, result in migrated:
                print(f"  - {name}: {result.get('predictions_migrated', 0)}/{result.get('total_predictions', 0)} predictions")
    
    return 0


if __name__ == "__main__":
    exit(main())
