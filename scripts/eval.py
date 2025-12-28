#!/usr/bin/env python3
"""Evaluation script for LLM Baselines models using lm_eval harness."""

import json
import logging

import hydra
from omegaconf import DictConfig

from optimus_dl.core.log import setup_logging
from optimus_dl.recipe.eval import (
    EvalConfig,
    EvalRecipe,
)

logger = logging.getLogger(__name__)


def pretty_print_results(results: dict) -> None:
    """Pretty print evaluation results.

    Args:
        results: Dictionary with evaluation results from lm_eval
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Print overall results
    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\n{task_name.upper()}:")
            print("-" * 40)

            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, int | float):
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")

    # Print task versions and configs if available
    if "versions" in results:
        print(f"Task Versions: {results['versions']}")

    if "config" in results:
        print("Evaluation Config:")
        for key, value in results["config"].items():
            if key != "model_args":  # Skip verbose model args
                print(f"  {key}: {value}")

    print("\n" + "=" * 80)


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")
def evaluate(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration
    """
    setup_logging()

    # Convert to structured config
    from omegaconf import OmegaConf

    eval_cfg: EvalConfig = OmegaConf.merge(OmegaConf.structured(EvalConfig), cfg)

    logger.info("Starting LLM Baselines Evaluation")
    logger.info(f"Checkpoint: {eval_cfg.common.checkpoint_path}")
    logger.info(f"Tasks: {eval_cfg.lm_eval.tasks}")

    # Create recipe and run evaluation
    recipe = EvalRecipe(eval_cfg)

    try:
        results = recipe.run_lm_eval()

        # Pretty print results to console
        pretty_print_results(results)

        # Log summary metrics
        if "results" in results:
            for task_name, task_results in results["results"].items():
                logger.info(f"{task_name}: {json.dumps(task_results, indent=2)}")

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    evaluate()
