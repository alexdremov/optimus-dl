#!/usr/bin/env python3
"""Serving script for LLM Baselines models."""

import logging

import hydra
from omegaconf import (
    OmegaConf,
    DictConfig,
)

from optimus_dl.core.log import setup_logging
from optimus_dl.recipe.serve import (
    ServeConfig,
    ServeRecipe,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/serve", config_name="tinyllama")
def serve(cfg: DictConfig) -> None:
    """Main serving function.

    Args:
        cfg: Hydra configuration
    """
    setup_logging()

    # Convert to structured config
    serve_cfg: ServeConfig = OmegaConf.merge(OmegaConf.structured(ServeConfig), cfg)

    logger.info("Starting LLM Baselines Server")

    # Create recipe and run server
    recipe = ServeRecipe(serve_cfg)
    recipe.run()


if __name__ == "__main__":
    serve()
