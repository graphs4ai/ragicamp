#!/usr/bin/env python3
"""Example of using path utilities to avoid FileNotFoundError."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragicamp.utils.paths import ensure_dir, ensure_output_dirs, safe_write_json


def main():
    """Demonstrate path utilities."""

    print("=" * 70)
    print("Example 1: ensure_dir() - Avoid FileNotFoundError")
    print("=" * 70)

    # Problem: This would fail if 'deep/nested/path' doesn't exist
    # with open("deep/nested/path/file.txt", "w") as f:
    #     f.write("data")  # ❌ FileNotFoundError!

    # Solution: Use ensure_dir first
    ensure_dir("deep/nested/path/file.txt")
    with open("deep/nested/path/file.txt", "w") as f:
        f.write("data")  # ✓ Works!
    print("✓ Created: deep/nested/path/file.txt")

    print("\n" + "=" * 70)
    print("Example 2: safe_write_json() - Write JSON safely")
    print("=" * 70)

    # Combines ensure_dir + json.dump
    data = {"key": "value", "number": 42}
    safe_write_json(data, "another/path/data.json", indent=2)
    print("✓ Created: another/path/data.json")

    print("\n" + "=" * 70)
    print("Example 3: ensure_output_dirs() - Setup all common dirs")
    print("=" * 70)

    # Creates all standard RAGiCamp directories
    ensure_output_dirs()
    print("✓ Created all standard directories:")
    print("  - outputs/")
    print("  - outputs/experiments/")
    print("  - outputs/comparisons/")
    print("  - artifacts/")
    print("  - artifacts/retrievers/")
    print("  - artifacts/agents/")
    print("  - data/")
    print("  - data/datasets/")

    print("\n" + "=" * 70)
    print("Example 4: In your code")
    print("=" * 70)

    print("\n❌ BAD - Can fail with FileNotFoundError:")
    print(
        """
    output_path = "outputs/experiment1/results.json"
    with open(output_path, 'w') as f:
        json.dump(data, f)
    """
    )

    print("\n✅ GOOD - Always works:")
    print(
        """
    from ragicamp.utils import ensure_dir, safe_write_json
    
    # Option 1: Ensure dir then write
    ensure_dir("outputs/experiment1/results.json")
    with open("outputs/experiment1/results.json", 'w') as f:
        json.dump(data, f)
    
    # Option 2: Use safe_write_json (recommended)
    safe_write_json(data, "outputs/experiment1/results.json", indent=2)
    """
    )

    # Cleanup
    import shutil

    for path in ["deep", "another"]:
        if Path(path).exists():
            shutil.rmtree(path)
    print("\n✓ Cleaned up example files")

    print("\n" + "=" * 70)
    print("✅ All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
