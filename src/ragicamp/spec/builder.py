"""Experiment specification builder.

Builds ExperimentSpec objects from YAML configuration files.
This is the main entry point for generating the experiment matrix.
"""

from typing import Any, Dict, List

from ragicamp.spec.experiment import ExperimentSpec
from ragicamp.spec.naming import name_direct, name_rag


def build_specs(config: Dict[str, Any]) -> List[ExperimentSpec]:
    """Build experiment specs from YAML config.

    Generates the full experiment matrix by combining:
    - Models x Datasets x Prompts x Quantizations (for direct)
    - Models x Retrievers x TopK x QueryTransforms x Rerankers x Prompts x Datasets (for RAG)

    Args:
        config: Study configuration dict (loaded from YAML)

    Returns:
        List of ExperimentSpec objects for all experiments

    Example:
        >>> with open("conf/study/my_study.yaml") as f:
        ...     config = yaml.safe_load(f)
        >>> specs = build_specs(config)
        >>> print(f"Generated {len(specs)} experiments")
    """
    specs: List[ExperimentSpec] = []

    # Global settings
    datasets = config.get("datasets", ["nq"])
    batch_size = config.get("batch_size", 8)
    min_batch_size = config.get("min_batch_size", 1)
    metrics = config.get("metrics", ["f1", "exact_match"])

    # Build direct experiment specs
    direct = config.get("direct", {})
    if direct.get("enabled"):
        specs.extend(
            _build_direct_specs(
                direct, datasets, batch_size, min_batch_size, metrics
            )
        )

    # Build RAG experiment specs
    rag = config.get("rag", {})
    if rag.get("enabled"):
        specs.extend(
            _build_rag_specs(rag, datasets, batch_size, min_batch_size, metrics)
        )

    return specs


def _build_direct_specs(
    direct_config: Dict[str, Any],
    datasets: List[str],
    batch_size: int,
    min_batch_size: int,
    metrics: List[str],
) -> List[ExperimentSpec]:
    """Build specs for direct (non-RAG) experiments."""
    specs: List[ExperimentSpec] = []

    models = direct_config.get("models", [])
    prompts = direct_config.get("prompts", ["default"])
    quantizations = direct_config.get("quantization", ["4bit"])

    for model in models:
        for prompt in prompts:
            for quant in quantizations:
                # Skip non-4bit for OpenAI models
                if model.startswith("openai:") and quant != "4bit":
                    continue

                for dataset in datasets:
                    name = name_direct(model, prompt, dataset, quant)
                    specs.append(
                        ExperimentSpec(
                            name=name,
                            exp_type="direct",
                            model=model,
                            dataset=dataset,
                            prompt=prompt,
                            quant=quant,
                            batch_size=batch_size,
                            min_batch_size=min_batch_size,
                            metrics=metrics,
                        )
                    )

    return specs


def _build_rag_specs(
    rag_config: Dict[str, Any],
    datasets: List[str],
    batch_size: int,
    min_batch_size: int,
    metrics: List[str],
) -> List[ExperimentSpec]:
    """Build specs for RAG experiments."""
    specs: List[ExperimentSpec] = []

    models = rag_config.get("models", [])
    retrievers = rag_config.get("retrievers", [])
    top_k_values = rag_config.get("top_k_values", [5])
    prompts = rag_config.get("prompts", ["default"])
    quantizations = rag_config.get("quantization", ["4bit"])

    # Query transform options
    query_transforms = rag_config.get("query_transform", ["none"])
    if not query_transforms:
        query_transforms = ["none"]

    # Reranker configs
    reranker_cfgs = rag_config.get("reranker", {}).get(
        "configs", [{"enabled": False, "name": "none"}]
    )
    if not reranker_cfgs:
        reranker_cfgs = [{"enabled": False, "name": "none"}]

    for model in models:
        for ret_config in retrievers:
            # Handle both string and dict retriever configs
            ret_name = (
                ret_config["name"] if isinstance(ret_config, dict) else ret_config
            )

            for top_k in top_k_values:
                for prompt in prompts:
                    for quant in quantizations:
                        # Skip non-4bit for OpenAI models
                        if model.startswith("openai:") and quant != "4bit":
                            continue

                        for qt in query_transforms:
                            for rr_cfg in reranker_cfgs:
                                rr_name = (
                                    rr_cfg.get("name", "none")
                                    if rr_cfg.get("enabled")
                                    else "none"
                                )
                                rr_model = (
                                    rr_cfg.get("model")
                                    if rr_cfg.get("enabled")
                                    else None
                                )

                                for dataset in datasets:
                                    name = name_rag(
                                        model,
                                        prompt,
                                        dataset,
                                        quant,
                                        ret_name,
                                        top_k,
                                        qt,
                                        rr_name,
                                    )
                                    specs.append(
                                        ExperimentSpec(
                                            name=name,
                                            exp_type="rag",
                                            model=model,
                                            dataset=dataset,
                                            prompt=prompt,
                                            quant=quant,
                                            retriever=ret_name,
                                            top_k=top_k,
                                            query_transform=qt if qt != "none" else None,
                                            reranker=rr_name if rr_name != "none" else None,
                                            reranker_model=rr_model,
                                            batch_size=batch_size,
                                            min_batch_size=min_batch_size,
                                            metrics=metrics,
                                        )
                                    )

    return specs
