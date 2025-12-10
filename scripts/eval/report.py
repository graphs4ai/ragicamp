#!/usr/bin/env python3
"""Generate evaluation reports.

Usage:
    python scripts/eval/report.py outputs/
    python scripts/eval/report.py outputs/ --format html
    python scripts/eval/report.py outputs/run.json --format markdown
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def load_all_results(directory: Path, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """Load all results from a directory."""
    results = []
    for path in directory.glob(pattern):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_path"] = str(path)
            data["_name"] = path.stem
            results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    return results


def generate_markdown_report(results: List[Dict], output_path: Path = None) -> str:
    """Generate a Markdown report."""
    lines = [
        "# RAGiCamp Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total experiments: {len(results)}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]
    
    # Collect all metrics
    all_metrics = set()
    for r in results:
        if "metrics" in r:
            all_metrics.update(r["metrics"].keys())
        for key in ["exact_match", "f1", "bertscore_f1"]:
            if key in r:
                all_metrics.add(key)
    
    # Create table
    lines.append("| Experiment | Examples | " + " | ".join(sorted(all_metrics)) + " |")
    lines.append("|" + "|".join(["---"] * (2 + len(all_metrics))) + "|")
    
    for r in results:
        name = r.get("_name", "unknown")
        n = r.get("num_examples", r.get("total_examples", "?"))
        
        # Get metrics
        metrics = r.get("metrics", {})
        for key in all_metrics:
            if key in r and key not in metrics:
                metrics[key] = r[key]
        
        row = f"| {name} | {n} |"
        for metric in sorted(all_metrics):
            value = metrics.get(metric)
            if value is not None and isinstance(value, float):
                row += f" {value:.4f} |"
            elif value is not None:
                row += f" {value} |"
            else:
                row += " - |"
        lines.append(row)
    
    lines.extend([
        "",
        "---",
        "",
        "## Details",
        "",
    ])
    
    for r in results:
        name = r.get("_name", "unknown")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Path**: `{r.get('_path', 'unknown')}`")
        lines.append(f"- **Examples**: {r.get('num_examples', '?')}")
        
        if "config" in r:
            config = r["config"]
            if "model" in config:
                lines.append(f"- **Model**: {config['model'].get('model_name', '?')}")
            if "agent" in config:
                lines.append(f"- **Agent**: {config['agent'].get('type', '?')}")
        
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
    
    # Simple HTML wrapper
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAGiCamp Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
<pre>{markdown}</pre>
</body>
</html>
"""
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        print(f"Report saved to: {output_path}")
    
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports",
    )
    
    parser.add_argument(
        "path",
        type=Path,
        help="Results file or directory",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--pattern", "-p",
        default="*summary*.json",
        help="Pattern for finding files (default: *summary*.json)",
    )
    
    args = parser.parse_args()
    
    # Load results
    if args.path.is_file():
        with open(args.path) as f:
            results = [json.load(f)]
        results[0]["_path"] = str(args.path)
        results[0]["_name"] = args.path.stem
    else:
        results = load_all_results(args.path, args.pattern)
    
    if not results:
        print("No results found")
        sys.exit(1)
    
    # Generate report
    if args.format == "markdown":
        output_path = args.output or Path("report.md")
        report = generate_markdown_report(results, output_path)
        if not args.output:
            print(report)
    elif args.format == "html":
        output_path = args.output or Path("report.html")
        generate_html_report(results, output_path)
    elif args.format == "json":
        output_path = args.output or Path("report.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
