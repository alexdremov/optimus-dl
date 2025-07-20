import ast
import logging
import operator

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
