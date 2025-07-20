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
    setup_logging()
    recipe = build("train_recipe", cfg_raw)
    assert isinstance(recipe, TrainRecipe)
    recipe.run()


if __name__ == "__main__":
    train()
