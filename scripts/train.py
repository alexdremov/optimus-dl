"""Training script entry point.

This script provides the main entry point for training models using Optimus-DL.
It uses Hydra for configuration management, allowing flexible configuration
via command-line arguments and YAML files.

Usage:
    Basic training::

        python scripts/train.py

    With config overrides::

        python scripts/train.py model=llama optimization.batch_size=64

    Using a different config::

        python scripts/train.py --config-name=train_llama_shakespeare

    Multirun (hyperparameter sweep)::

        python scripts/train.py -m optimization.optimizer.lr=1e-3,1e-4,1e-5
"""

import logging

import hydra

from optimus_dl.core.log import setup_logging
from optimus_dl.core.registry import build
from optimus_dl.recipe.train.base import TrainRecipe

logger = logging.getLogger()


@hydra.main(
    version_base=None, config_path="../configs/train", config_name="train_llama"
)
def train(cfg_raw):
    """Main training function.

    This function is called by Hydra with the resolved configuration. It:
    1. Sets up logging
    2. Builds the training recipe from configuration
    3. Runs the training loop

    Args:
        cfg_raw: Raw configuration object from Hydra. This will be a TrainConfig
            instance with all interpolations resolved.

    Example:
        The function is typically called by Hydra, but can be called directly::

            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.load("configs/train/train_llama.yaml")
            >>> train(cfg)
    """
    setup_logging()
    recipe = build("train_recipe", cfg_raw)
    assert isinstance(recipe, TrainRecipe)
    recipe.run()


if __name__ == "__main__":
    train()
