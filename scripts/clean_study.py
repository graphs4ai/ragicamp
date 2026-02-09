#!/usr/bin/env python3
"""Diagnose and clean experiment results that were incorrectly run.

Detects experiments that are tainted by known bugs and should be re-run:

1. **Fake reranker** — Name encodes a reranker (e.g. ``_bge_``) but reranking
   was never wired into the agent pipeline.  Results are identical to the
   no-reranker baseline.

2. **Fake query_transform** — Name encodes a query transform (e.g. ``_hyde_``)
   but the feature was never wired.  Results are identical to ``none``.

3. **Duplicate configs** — Multiple directories with different names but
   identical logical configs (same model, dataset, retriever, top_k, prompt).

4. **Incomplete / failed** — Experiments stuck in non-terminal phases.

5. **Metadata-impoverished** — metadata.json missing critical fields
   (model, dataset, retriever), making the experiment un-analysable.

The script has three modes:

- **diagnose** (default): Print a detailed report of all issues found.
- **clean**: Move tainted experiments to ``_tainted/`` archive + delete
  ``optuna_study.db`` so TPE can restart fresh.
- **nuke**: Delete tainted experiments permanently (no archive).

Usage:
    # Dry run — just show what's wrong:
    python scripts/clean_study.py --study outputs/smart_retrieval_slm

    # Clean tainted experiments (archive + remove Optuna DB):
    python scripts/clean_study.py --study outputs/smart_retrieval_slm --clean

    # Nuke tainted experiments (no archive):
    python scripts/clean_study.py --study outputs/smart_retrieval_slm --nuke

    # Also clean incomplete/failed experiments:
    python scripts/clean_study.py --study outputs/smart_retrieval_slm --clean --include-incomplete

    # Keep only experiments matching a specific whitelist:
    python scripts/clean_study.py --study outputs/smart_retrieval_slm --clean --keep-list keep.txt
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================================
# Constants
# ============================================================================

SKIP_DIRS = {
    "_archived_fake_reranked", "_tainted", "analysis",
    "__pycache__", ".ipynb_checkpoints",
}

KNOWN_RERANKER_TOKENS = {
    "bge", "bgev2", "bgev2m3", "bgebase",
    "msmarco", "msmarcolarge",
}

KNOWN_QUERY_TRANSFORMS = {"hyde", "multiquery"}

KNOWN_DATASETS = {"nq", "hotpotqa", "triviaqa", "musique"}

DATASET_ALIASES = {
    "natural_questions": "nq",
}


# ============================================================================
# Data model
# ============================================================================

@dataclass
class ExperimentInfo:
    """Parsed info for a single experiment directory."""
    dir_path: Path
    name: str
    metadata: Optional[dict] = None
    state: Optional[dict] = None
    results: Optional[dict] = None
    issues: list[str] = field(default_factory=list)
    # Parsed config (normalised for dedup)
    config_key: Optional[tuple] = None

    @property
    def phase(self) -> Optional[str]:
        if self.state:
            return self.state.get("phase")
        return None

    @property
    def is_complete(self) -> bool:
        return self.phase == "complete"

    @property
    def has_predictions(self) -> bool:
        return (self.dir_path / "predictions.json").exists()


@dataclass
class DiagnosticReport:
    """Summary of all issues found in a study."""
    total_experiments: int = 0
    clean_experiments: int = 0
    fake_reranker: list[ExperimentInfo] = field(default_factory=list)
    fake_query_transform: list[ExperimentInfo] = field(default_factory=list)
    duplicates: dict[tuple, list[ExperimentInfo]] = field(default_factory=dict)
    incomplete: list[ExperimentInfo] = field(default_factory=list)
    impoverished: list[ExperimentInfo] = field(default_factory=list)

    @property
    def tainted(self) -> set[str]:
        """All experiment names that have at least one issue."""
        names: set[str] = set()
        for exp in self.fake_reranker:
            names.add(exp.name)
        for exp in self.fake_query_transform:
            names.add(exp.name)
        for exps in self.duplicates.values():
            # Keep the best one (most complete, or best f1), taint the rest
            best = _pick_best(exps)
            for exp in exps:
                if exp.name != best.name:
                    names.add(exp.name)
        return names

    @property
    def tainted_including_incomplete(self) -> set[str]:
        names = self.tainted
        for exp in self.incomplete:
            names.add(exp.name)
        for exp in self.impoverished:
            names.add(exp.name)
        return names


# ============================================================================
# Helpers
# ============================================================================

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _get_f1(exp: ExperimentInfo) -> float:
    """Extract F1 score from results or predictions."""
    if exp.results:
        metrics = exp.results.get("metrics", {})
        if isinstance(metrics, dict):
            return metrics.get("f1", 0.0)
    return 0.0


def _pick_best(exps: list[ExperimentInfo]) -> ExperimentInfo:
    """From a list of duplicates, pick the best to keep."""
    # Prefer complete > incomplete
    complete = [e for e in exps if e.is_complete]
    candidates = complete if complete else exps
    # Among those, pick highest F1
    return max(candidates, key=_get_f1)


def _normalise_qt(val: Optional[str]) -> str:
    if val is None or val == "none" or val == "":
        return "none"
    return val.lower()


def _normalise_reranker(val: Optional[str]) -> str:
    if val is None or val == "none" or val == "":
        return "none"
    return val.lower()


def _extract_config_key(exp: ExperimentInfo) -> Optional[tuple]:
    """Build a normalised config fingerprint for deduplication.

    Returns a tuple of (model, dataset, exp_type, retriever, top_k, prompt,
    query_transform_effective, reranker_effective) where the "effective"
    values reflect what the code ACTUALLY did, not what was configured.
    """
    meta = exp.metadata
    if not meta:
        return None

    model = meta.get("model") or "unknown"
    dataset = DATASET_ALIASES.get(meta.get("dataset", ""), meta.get("dataset", "unknown"))
    exp_type = meta.get("type") or "unknown"
    retriever = meta.get("retriever") or "none"
    top_k = meta.get("top_k") or 5
    prompt = meta.get("prompt") or "unknown"
    agent_type = meta.get("agent_type") or "fixed_rag"

    # Effective values: these are what the code ACTUALLY used regardless of config.
    # Both query_transform and reranker were NEVER wired into the agent pipeline
    # in the old code, so the effective value is always "none" for experiments
    # produced before the fix.
    qt_effective = "none"
    rr_effective = "none"

    return (model, dataset, exp_type, retriever, top_k, prompt, qt_effective, rr_effective, agent_type)


def _name_has_reranker(name: str) -> Optional[str]:
    """Check if experiment name encodes a reranker. Return the token or None."""
    parts = name.split("_")
    # Find _k{N}_ segment
    k_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^k\d+$", p):
            k_idx = i
            break

    if k_idx is None:
        return None

    # Tokens between k{N} and the last two parts (prompt, dataset)
    tail = parts[k_idx + 1:]
    if len(tail) < 2:
        return None

    middle = tail[:-2]  # strip prompt + dataset
    for token in middle:
        if token.lower() in KNOWN_RERANKER_TOKENS:
            return token
    return None


def _name_has_query_transform(name: str) -> Optional[str]:
    """Check if experiment name encodes a query transform. Return the token or None."""
    parts = name.split("_")
    k_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^k\d+$", p):
            k_idx = i
            break

    if k_idx is None:
        return None

    tail = parts[k_idx + 1:]
    if len(tail) < 2:
        return None

    middle = tail[:-2]
    for token in middle:
        if token.lower() in KNOWN_QUERY_TRANSFORMS:
            return token
    return None


# ============================================================================
# Diagnosis
# ============================================================================


def scan_study(study_path: Path, include_incomplete: bool = False) -> DiagnosticReport:
    """Scan a study directory and build a diagnostic report."""
    report = DiagnosticReport()
    experiments: list[ExperimentInfo] = []

    for exp_dir in sorted(study_path.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        if exp_dir.name.startswith("."):
            continue

        exp = ExperimentInfo(
            dir_path=exp_dir,
            name=exp_dir.name,
            metadata=_load_json(exp_dir / "metadata.json"),
            state=_load_json(exp_dir / "state.json"),
            results=_load_json(exp_dir / "results.json"),
        )

        report.total_experiments += 1

        # --- Issue 1: Fake reranker ---
        rr_token = _name_has_reranker(exp.name)
        if rr_token:
            exp.issues.append(f"fake_reranker({rr_token})")
            report.fake_reranker.append(exp)

        # --- Issue 2: Fake query_transform ---
        qt_token = _name_has_query_transform(exp.name)
        if qt_token:
            exp.issues.append(f"fake_query_transform({qt_token})")
            report.fake_query_transform.append(exp)

        # --- Issue 4: Incomplete / failed ---
        phase = exp.phase
        if phase and phase not in ("complete",):
            exp.issues.append(f"incomplete({phase})")
            report.incomplete.append(exp)

        # --- Issue 5: Impoverished metadata ---
        if exp.metadata:
            missing_fields = []
            if not exp.metadata.get("model"):
                missing_fields.append("model")
            if not exp.metadata.get("dataset"):
                missing_fields.append("dataset")
            if exp.metadata.get("type") == "rag" and not exp.metadata.get("retriever"):
                missing_fields.append("retriever")
            if missing_fields:
                exp.issues.append(f"missing_metadata({','.join(missing_fields)})")
                report.impoverished.append(exp)
        elif not exp.metadata:
            exp.issues.append("no_metadata")
            report.impoverished.append(exp)

        experiments.append(exp)

    # --- Issue 3: Duplicates (same effective config, different names) ---
    config_groups: dict[tuple, list[ExperimentInfo]] = defaultdict(list)
    for exp in experiments:
        key = _extract_config_key(exp)
        if key:
            exp.config_key = key
            config_groups[key].append(exp)

    for key, exps in config_groups.items():
        if len(exps) > 1:
            report.duplicates[key] = exps
            for exp in exps:
                exp.issues.append(f"duplicate(group_size={len(exps)})")

    report.clean_experiments = sum(
        1 for exp in experiments if not exp.issues
    )

    return report


# ============================================================================
# Reporting
# ============================================================================

_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def print_report(report: DiagnosticReport, include_incomplete: bool = False) -> None:
    """Print a human-readable diagnostic report."""
    print(f"\n{_BOLD}{'=' * 70}")
    print(f"  STUDY DIAGNOSTIC REPORT")
    print(f"{'=' * 70}{_RESET}\n")

    print(f"  Total experiments:  {report.total_experiments}")
    print(f"  {_GREEN}Clean:              {report.clean_experiments}{_RESET}")
    print(f"  {_RED}Fake reranker:      {len(report.fake_reranker)}{_RESET}")
    print(f"  {_RED}Fake query_transform: {len(report.fake_query_transform)}{_RESET}")
    print(f"  {_YELLOW}Duplicate groups:   {len(report.duplicates)} groups "
          f"({sum(len(v) for v in report.duplicates.values())} experiments){_RESET}")
    print(f"  {_YELLOW}Incomplete/failed:  {len(report.incomplete)}{_RESET}")
    print(f"  {_YELLOW}Missing metadata:   {len(report.impoverished)}{_RESET}")

    # --- Fake reranker details ---
    if report.fake_reranker:
        print(f"\n{_BOLD}{_RED}--- FAKE RERANKER (reranker in name but never wired) ---{_RESET}")
        for exp in report.fake_reranker:
            rr = _name_has_reranker(exp.name)
            f1 = _get_f1(exp)
            status = exp.phase or "?"
            print(f"  {_RED}✗{_RESET} {exp.name}")
            print(f"    reranker_token={rr}  f1={f1:.4f}  phase={status}")

    # --- Fake query_transform details ---
    if report.fake_query_transform:
        print(f"\n{_BOLD}{_RED}--- FAKE QUERY_TRANSFORM (qt in name but never wired) ---{_RESET}")
        for exp in report.fake_query_transform:
            qt = _name_has_query_transform(exp.name)
            f1 = _get_f1(exp)
            status = exp.phase or "?"
            print(f"  {_RED}✗{_RESET} {exp.name}")
            print(f"    query_transform={qt}  f1={f1:.4f}  phase={status}")

    # --- Duplicates ---
    if report.duplicates:
        print(f"\n{_BOLD}{_YELLOW}--- DUPLICATE CONFIG GROUPS ---{_RESET}")
        print(f"  (Experiments with the SAME effective config but different names.)")
        print(f"  (query_transform was not wired, so hyde==none in practice.)\n")

        for i, (key, exps) in enumerate(sorted(report.duplicates.items()), 1):
            best = _pick_best(exps)
            model, dataset, exp_type, retriever, top_k, prompt, qt_eff, rr, agent_type = key
            print(f"  {_CYAN}Group {i}{_RESET}: "
                  f"model={model.split(':')[-1].split('/')[-1]}  "
                  f"retriever={retriever}  k={top_k}  "
                  f"prompt={prompt}  dataset={dataset}  "
                  f"agent={agent_type}")
            for exp in exps:
                f1 = _get_f1(exp)
                marker = f"{_GREEN}KEEP{_RESET}" if exp.name == best.name else f"{_RED}TAINT{_RESET}"
                print(f"    [{marker}] {exp.name}  f1={f1:.4f}  phase={exp.phase or '?'}")

    # --- Incomplete / failed ---
    if include_incomplete and report.incomplete:
        print(f"\n{_BOLD}{_YELLOW}--- INCOMPLETE / FAILED ---{_RESET}")
        for exp in report.incomplete:
            f1 = _get_f1(exp)
            print(f"  {_YELLOW}⚠{_RESET} {exp.name}  phase={exp.phase}  f1={f1:.4f}")

    # --- Summary of what would be cleaned ---
    tainted = report.tainted
    if include_incomplete:
        tainted = report.tainted_including_incomplete

    print(f"\n{_BOLD}--- CLEANUP SUMMARY ---{_RESET}")
    print(f"  Experiments to remove:  {_RED}{len(tainted)}{_RESET}")
    print(f"  Experiments to keep:    {_GREEN}{report.total_experiments - len(tainted)}{_RESET}")

    if tainted:
        # Categorize tainted experiments
        fake_rr_names = {e.name for e in report.fake_reranker}
        fake_qt_names = {e.name for e in report.fake_query_transform}
        dup_losers = set()
        for exps in report.duplicates.values():
            best = _pick_best(exps)
            for exp in exps:
                if exp.name != best.name:
                    dup_losers.add(exp.name)

        # Some experiments may be tainted for multiple reasons
        only_rr = fake_rr_names - fake_qt_names - dup_losers
        only_qt = fake_qt_names - fake_rr_names - dup_losers
        only_dup = dup_losers - fake_rr_names - fake_qt_names
        multi = tainted - only_rr - only_qt - only_dup

        if only_rr:
            print(f"    {len(only_rr):3d} fake reranker only")
        if only_qt:
            print(f"    {len(only_qt):3d} fake query_transform only")
        if only_dup:
            print(f"    {len(only_dup):3d} duplicate (worse) only")
        if multi:
            print(f"    {len(multi):3d} multiple issues")
        if include_incomplete:
            inc_only = {e.name for e in report.incomplete} - fake_rr_names - fake_qt_names - dup_losers
            if inc_only:
                print(f"    {len(inc_only):3d} incomplete only")

    print()


# ============================================================================
# Cleanup actions
# ============================================================================


def execute_clean(
    study_path: Path,
    report: DiagnosticReport,
    include_incomplete: bool = False,
    archive: bool = True,
) -> None:
    """Move tainted experiments to archive and remove Optuna DB.

    Args:
        study_path: Path to the study output directory.
        report: DiagnosticReport from scan_study.
        include_incomplete: Also clean incomplete/failed experiments.
        archive: If True, move to _tainted/; if False, delete permanently.
    """
    tainted = report.tainted
    if include_incomplete:
        tainted = report.tainted_including_incomplete

    if not tainted:
        print("Nothing to clean.")
        return

    archive_dir = study_path / "_tainted"
    if archive:
        archive_dir.mkdir(exist_ok=True)

    removed = 0
    for name in sorted(tainted):
        exp_dir = study_path / name
        if not exp_dir.exists():
            continue

        if archive:
            dest = archive_dir / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(exp_dir), str(dest))
            print(f"  archived: {name}")
        else:
            shutil.rmtree(exp_dir)
            print(f"  deleted:  {name}")
        removed += 1

    # Remove Optuna study DB so TPE can reinitialize
    optuna_db = study_path / "optuna_study.db"
    if optuna_db.exists():
        optuna_db.unlink()
        print(f"\n  Removed optuna_study.db (TPE will reinitialize on next run)")

    # Remove study_summary.json (will be regenerated)
    summary = study_path / "study_summary.json"
    if summary.exists():
        summary.unlink()
        print(f"  Removed study_summary.json (will be regenerated)")

    comparison = study_path / "comparison.json"
    if comparison.exists():
        comparison.unlink()
        print(f"  Removed comparison.json (will be regenerated)")

    print(f"\n  Done: {removed} experiments {'archived' if archive else 'deleted'}.")
    remaining = report.total_experiments - removed
    print(f"  Remaining: {remaining} clean experiments on disk.")
    print(f"  Optuna will re-seed from these on next study run.\n")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and clean incorrectly-run experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--study", required=True, type=Path,
        help="Path to study output directory (e.g. outputs/smart_retrieval_slm)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Archive tainted experiments to _tainted/ and remove Optuna DB",
    )
    parser.add_argument(
        "--nuke", action="store_true",
        help="Delete tainted experiments permanently (no archive)",
    )
    parser.add_argument(
        "--include-incomplete", action="store_true",
        help="Also clean incomplete/failed experiments",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output report as JSON (for piping to other tools)",
    )
    args = parser.parse_args()

    study_path = args.study
    if not study_path.is_dir():
        print(f"Error: {study_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = scan_study(study_path, include_incomplete=args.include_incomplete)

    if args.json:
        tainted = report.tainted
        if args.include_incomplete:
            tainted = report.tainted_including_incomplete
        output = {
            "total": report.total_experiments,
            "clean": report.clean_experiments,
            "fake_reranker": [e.name for e in report.fake_reranker],
            "fake_query_transform": [e.name for e in report.fake_query_transform],
            "duplicate_groups": {
                str(k): [e.name for e in v]
                for k, v in report.duplicates.items()
            },
            "incomplete": [e.name for e in report.incomplete],
            "impoverished": [e.name for e in report.impoverished],
            "tainted": sorted(tainted),
            "to_keep": sorted(
                set(e.name for e in [])  # placeholder
                | {e.name for e in []}
            ),
        }
        json.dump(output, sys.stdout, indent=2)
        print()
    else:
        print_report(report, include_incomplete=args.include_incomplete)

    if args.clean:
        confirm = input(f"Archive {len(report.tainted)} experiments and remove Optuna DB? [y/N] ")
        if confirm.lower() == "y":
            execute_clean(study_path, report, args.include_incomplete, archive=True)
        else:
            print("Aborted.")

    elif args.nuke:
        tainted = report.tainted
        if args.include_incomplete:
            tainted = report.tainted_including_incomplete
        confirm = input(
            f"{_RED}PERMANENTLY DELETE {len(tainted)} experiments?{_RESET} "
            f"This cannot be undone. [y/N] "
        )
        if confirm.lower() == "y":
            execute_clean(study_path, report, args.include_incomplete, archive=False)
        else:
            print("Aborted.")


if __name__ == "__main__":
    main()
