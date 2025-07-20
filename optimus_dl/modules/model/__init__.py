from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import BaseModel
from .config import ModelConfig

_MODELS_REGISTRY, register_model, build_model = make_registry("model")


bootstrap_module(__name__)
