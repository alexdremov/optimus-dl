#!/usr/bin/env python3
"""Data preparation script for LLM Baselines."""

import logging

import hydra
from omegaconf import (
    OmegaConf,
    DictConfig,
)

from optimus_dl.core.log import setup_logging
from optimus_dl.recipe.pretokenize.config import DataPrepConfig
from optimus_dl.recipe.pretokenize.recipe import DataPrepRecipe

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/prepare_data", config_name="default"
)
def prepare_data(cfg: DictConfig) -> None:
    setup_logging()

    # Merge with structured config to ensure types/defaults
    data_prep_cfg = OmegaConf.merge(OmegaConf.structured(DataPrepConfig), cfg)

    recipe = DataPrepRecipe(data_prep_cfg)
    recipe.run()


if __name__ == "__main__":
    prepare_data()
