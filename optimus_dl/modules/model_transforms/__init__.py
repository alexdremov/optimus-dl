from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import BaseModelTransform
from .config import ModelTransformConfig

_MODEL_TRANSFORMS_REGISTRY, register_model_transform, build_model_transform = (
    make_registry("model_transform")
)


bootstrap_module(__name__)
