import logging
import os

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("cpu_count", os.cpu_count)
