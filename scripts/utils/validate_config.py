#!/usr/bin/env python3
"""CLI tool for validating RAGiCamp configuration files.

Usage:
    python validate_config.py config.yaml
    python validate_config.py experiments/configs/*.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragicamp.config import ConfigLoader


def main():
    parser = argparse.ArgumentParser(
        description="Validate RAGiCamp configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single config
  python validate_config.py experiments/configs/nq_baseline_gemma2b_quick.yaml
  
  # Validate multiple configs
  python validate_config.py experiments/configs/*.yaml
  
  # Validate and show summary
  python validate_config.py config.yaml --verbose
        """
    )
    
    parser.add_argument(
        "configs",
        nargs="+",
        help="Configuration file(s) to validate"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    all_valid = True
    valid_count = 0
    invalid_count = 0
    
    print(f"\n🔍 Validating {len(args.configs)} configuration file(s)...\n")
    
    for config_path in args.configs:
        try:
            if ConfigLoader.validate_file(config_path):
                valid_count += 1
                if args.verbose:
                    print()
            else:
                invalid_count += 1
                all_valid = False
        except Exception as e:
            print(f"✗ Error validating {config_path}: {e}\n")
            invalid_count += 1
            all_valid = False
    
    # Summary
    print("=" * 70)
    print(f"Summary: {valid_count} valid, {invalid_count} invalid")
    print("=" * 70)
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()

