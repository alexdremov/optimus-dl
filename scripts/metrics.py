#!/usr/bin/env python3
"""Evaluation script for LLM Baselines models using internal MetricEngine."""

import json
import logging

import hydra
from omegaconf import (
    OmegaConf,
    DictConfig,
)

from optimus_dl.core.log import setup_logging
from optimus_dl.recipe.metrics import (
    MetricsConfig,
    MetricsRecipe,
)

logger = logging.getLogger(__name__)


def pretty_print_results(results: dict) -> None:
    """Pretty print evaluation results.

    Args:
        results: Dictionary with evaluation results {dataset_name: {metric: value}}
    """
    print("\n" + "=" * 80)
    print("INTERNAL METRICS EVALUATION RESULTS")
    print("=" * 80)

    for dataset_name, dataset_results in results.items():
        print(f"\nDATASET: {dataset_name.upper()}")
        print("-" * 40)

        for metric_name, metric_value in dataset_results.items():
            if isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")

    print("\n" + "=" * 80)


@hydra.main(version_base=None, config_path="../configs/metrics", config_name="default")
def evaluate(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration
    """
    setup_logging()

    # Convert to structured config
    metrics_cfg: MetricsConfig = OmegaConf.merge(
        OmegaConf.structured(MetricsConfig), cfg
    )

    logger.info("Starting LLM Baselines Metrics Evaluation")
    if metrics_cfg.common.checkpoint_path:
        logger.info(f"Checkpoint: {metrics_cfg.common.checkpoint_path}")

    # Create recipe and run evaluation
    recipe = MetricsRecipe(metrics_cfg)

    try:
        results = recipe.run()

        # Pretty print results to console
        pretty_print_results(results)

        # Log summary metrics
        logger.info(f"Summary Results: {json.dumps(results, indent=2)}")
        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    evaluate()
