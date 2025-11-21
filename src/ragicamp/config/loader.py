"""Configuration loading utilities for RAGiCamp.

This module provides utilities for loading, validating, and merging
experiment configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from ragicamp.config.schemas import ExperimentConfig


class ConfigLoader:
    """Utility class for loading and validating experiment configs."""

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration file.

        Args:
            path: Path to JSON file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def load(path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration file (auto-detect format).

        Args:
            path: Path to config file (.yaml, .yml, or .json)

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If file format not supported
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            return ConfigLoader.load_yaml(path)
        elif suffix == ".json":
            return ConfigLoader.load_json(path)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    @staticmethod
    def validate(config: Dict[str, Any]) -> ExperimentConfig:
        """Validate configuration against schema.

        Args:
            config: Raw configuration dictionary

        Returns:
            Validated ExperimentConfig object

        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            return ExperimentConfig(**config)
        except ValidationError as e:
            print("\n❌ Configuration validation failed:")
            print("\nErrors:")
            for error in e.errors():
                location = " -> ".join(str(x) for x in error["loc"])
                print(f"  • {location}: {error['msg']}")
                if "ctx" in error and "error" in error["ctx"]:
                    print(f"    Details: {error['ctx']['error']}")
            print("\nPlease fix the configuration and try again.\n")
            raise

    @staticmethod
    def load_and_validate(path: Union[str, Path]) -> ExperimentConfig:
        """Load and validate configuration file.

        Args:
            path: Path to config file

        Returns:
            Validated ExperimentConfig object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If configuration is invalid
        """
        config_dict = ConfigLoader.load(path)
        return ConfigLoader.validate(config_dict)

    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations (override takes precedence).

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = dict(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                # Override value
                result[key] = value

        return result

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default experiment configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "model": {
                "type": "huggingface",
                "device": "cuda",
                "load_in_8bit": False,
                "temperature": 0.7,
            },
            "dataset": {
                "split": "validation",
                "filter_no_answer": True,
                "cache_dir": "data/datasets",
            },
            "evaluation": {
                "batch_size": None,
            },
            "output": {
                "save_predictions": False,
                "output_dir": "outputs",
            },
        }

    @staticmethod
    def validate_file(path: Union[str, Path]) -> bool:
        """Validate a configuration file and print results.

        Args:
            path: Path to config file

        Returns:
            True if valid, False otherwise
        """
        try:
            config = ConfigLoader.load_and_validate(path)
            print(f"✓ Configuration is valid: {path}")
            print(f"\nSummary:")

            # Handle evaluate-only mode where agent/model/dataset might be None
            if config.agent:
                print(f"  Agent: {config.agent.type} ({config.agent.name})")
            else:
                print(f"  Agent: None (evaluate mode)")

            if config.model:
                print(f"  Model: {config.model.type} ({config.model.model_name})")
            else:
                print(f"  Model: None (evaluate mode)")

            if config.dataset:
                print(f"  Dataset: {config.dataset.name} ({config.dataset.split})")
            else:
                print(f"  Dataset: None (evaluate mode)")

            print(
                f"  Metrics: {', '.join(m['name'] if isinstance(m, dict) else m for m in config.metrics)}"
            )

            # Show evaluation mode if present
            if config.evaluation:
                print(f"  Mode: {config.evaluation.mode}")
                if config.evaluation.mode == "evaluate" and config.evaluation.predictions_file:
                    print(f"  Predictions file: {config.evaluation.predictions_file}")

            return True
        except (FileNotFoundError, ValidationError, Exception) as e:
            print(f"✗ Configuration is invalid: {path}")
            print(f"  Error: {e}")
            return False


def create_config_template(output_path: Union[str, Path]):
    """Create a template configuration file.

    Args:
        output_path: Where to save the template
    """
    template = {
        "agent": {
            "type": "direct_llm",
            "name": "my_agent",
            "system_prompt": "You are a helpful assistant.",
        },
        "model": {
            "type": "huggingface",
            "model_name": "google/flan-t5-small",
            "device": "cuda",
            "load_in_8bit": False,
        },
        "dataset": {
            "name": "natural_questions",
            "split": "validation",
            "num_examples": 10,
            "filter_no_answer": True,
        },
        "metrics": [
            "exact_match",
            "f1",
        ],
        "evaluation": {
            "batch_size": 4,
        },
        "output": {
            "save_predictions": True,
            "output_path": "outputs/my_experiment.json",
        },
    }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created template configuration: {output_path}")
