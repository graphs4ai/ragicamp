#!/usr/bin/env python3
"""Generate evaluation reports.

Usage:
    python scripts/eval/report.py outputs/
    python scripts/eval/report.py outputs/ --format html
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_hydra_config(result_dir: Path) -> Dict[str, Any]:
    """Load Hydra config from .hydra/config.yaml if available."""
    config_path = result_dir / ".hydra" / "config.yaml"
    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception:
            pass
    return {}


def load_result(path: Path) -> Optional[Dict[str, Any]]:
    """Load a single result file."""
    with open(path) as f:
        data = json.load(f)

    # Skip orchestration logs
    if "total_passed" in data and "results" in data:
        return None

    # Extract metrics
    metrics = {}
    if "overall_metrics" in data:
        for key, value in data["overall_metrics"].items():
            if isinstance(value, (int, float)) and key not in ["num_successful", "num_failures"]:
                metrics[key] = value

    for key in ["exact_match", "f1", "bertscore_f1", "bleurt", "llm_judge_qa"]:
        if key in data and key not in metrics:
            metrics[key] = data[key]

    if "metrics" in data and isinstance(data["metrics"], dict):
        metrics.update(data["metrics"])

    if not metrics:
        return None

    # Load Hydra config
    hydra_config = load_hydra_config(path.parent)

    # Extract experiment parameters
    model_name = "?"
    prompt_style = "?"
    agent_type = "?"

    if hydra_config:
        model_cfg = hydra_config.get("model", {})
        model_name = model_cfg.get("model_name", "?")
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        prompt_cfg = hydra_config.get("prompt", {})
        prompt_style = prompt_cfg.get("style", "?")

        agent_cfg = hydra_config.get("agent", {})
        agent_type = agent_cfg.get("type", "?")
    else:
        # Fallback: try to parse agent_name for older runs
        agent_name = data.get("agent_name", "")
        if "gemma" in agent_name.lower():
            model_name = "gemma-2b"
        elif "llama" in agent_name.lower():
            model_name = "llama3"
        elif "phi" in agent_name.lower():
            model_name = "phi3"

        if "rag" in agent_name.lower():
            prompt_style = "rag"
            agent_type = "fixed_rag"
        elif "direct" in agent_name.lower() or "baseline" in agent_name.lower():
            agent_type = "direct_llm"

    return {
        "path": str(path),
        "name": path.stem,
        "dataset": data.get("dataset_name", "?"),
        "model": model_name,
        "prompt": prompt_style,
        "agent": agent_type,
        "num_examples": data.get("num_examples", "?"),
        "timestamp": data.get("timestamp", "?"),
        "metrics": metrics,
    }


def find_results(base_path: Path, pattern: str = "*summary*.json") -> List[Dict[str, Any]]:
    """Find and load all result files recursively."""
    results = []

    if base_path.is_file():
        r = load_result(base_path)
        if r:
            results.append(r)
        return results

    for path in base_path.rglob(pattern):
        if "baseline_study_summary" in path.name or "rag_baseline_study" in path.name:
            continue
        try:
            r = load_result(path)
            if r:
                results.append(r)
        except Exception:
            pass

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


def generate_markdown_report(results: List[Dict], output_path: Path = None) -> str:
    """Generate a Markdown report."""
    lines = [
        "# RAGiCamp Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total experiments: {len(results)}",
        "",
    ]

    if not results:
        lines.append("No results found.")
        return "\n".join(lines)

    # Collect metrics
    all_metrics = set()
    for r in results:
        all_metrics.update(r["metrics"].keys())

    priority = ["exact_match", "f1", "bertscore_f1", "bleurt", "llm_judge_qa"]
    display_metrics = [m for m in priority if m in all_metrics][:5]

    # Summary table
    lines.extend(
        [
            "## Summary",
            "",
            "| Dataset | Model | Prompt | Agent | N | " + " | ".join(display_metrics) + " |",
            "|" + "|".join(["---"] * (5 + len(display_metrics))) + "|",
        ]
    )

    for r in results:
        model_short = r["model"][:20] if len(r["model"]) > 20 else r["model"]
        row = f"| {r['dataset']} | {model_short} | {r['prompt']} | {r['agent']} | {r['num_examples']} |"
        for metric in display_metrics:
            value = r["metrics"].get(metric)
            if isinstance(value, float):
                row += f" {value:.3f} |"
            elif value is not None:
                row += f" {value} |"
            else:
                row += " - |"
        lines.append(row)

    # Best results per dataset
    lines.extend(["", "## Best Results by Dataset", ""])
    datasets = set(r["dataset"] for r in results if r["dataset"] != "?")
    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r["dataset"] == dataset]
        if not dataset_results:
            continue

        lines.append(f"### {dataset}")
        lines.append("")

        # Best by each metric
        for metric in display_metrics[:3]:
            values = [
                (r, r["metrics"].get(metric)) for r in dataset_results if r["metrics"].get(metric)
            ]
            if values:
                best = max(values, key=lambda x: x[1])
                lines.append(
                    f"- **Best {metric}**: {best[1]:.3f} — {best[0]['model']}/{best[0]['prompt']}"
                )
        lines.append("")

    # Detailed results
    lines.extend(["## Detailed Results", ""])
    for r in results:
        lines.append(f"### {r['dataset']} / {r['model']} / {r['prompt']}")
        lines.append("")
        lines.append(f"- **Agent**: {r['agent']}")
        lines.append(f"- **Examples**: {r['num_examples']}")
        lines.append(f"- **Timestamp**: {r['timestamp']}")
        lines.append(f"- **Path**: `{r['path']}`")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for metric in display_metrics:
            value = r["metrics"].get(metric)
            if isinstance(value, float):
                lines.append(f"| {metric} | {value:.4f} |")
            elif value is not None:
                lines.append(f"| {metric} | {value} |")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def generate_html_report(results: List[Dict], output_path: Path = None) -> str:
    """Generate an HTML report."""
    markdown = generate_markdown_report(results)

    try:
        import markdown as md

        html_content = md.markdown(markdown, extensions=["tables"])
    except ImportError:
        html_content = f"<pre>{markdown}</pre>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAGiCamp Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1400px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #2196F3; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #e3f2fd; }}
        h1 {{ color: #1976D2; }}
        h2 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 5px; }}
        h3 {{ color: #555; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
        ul {{ margin: 10px 0; }}
        .best {{ background-color: #c8e6c9 !important; font-weight: bold; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        print(f"Report saved to: {output_path}")

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("path", type=Path, help="Results directory")
    parser.add_argument("--format", "-f", choices=["markdown", "html"], default="markdown")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--pattern", "-p", default="*summary*.json")

    args = parser.parse_args()

    results = find_results(args.path, args.pattern)

    if not results:
        print("No results found")
        sys.exit(1)

    if args.format == "markdown":
        output_path = args.output or Path("report.md")
        report = generate_markdown_report(results, output_path)
        print(report)
    elif args.format == "html":
        output_path = args.output or Path("report.html")
        generate_html_report(results, output_path)


if __name__ == "__main__":
    main()
