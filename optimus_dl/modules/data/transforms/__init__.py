from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry
from optimus_dl.modules.data.transforms.base import (
    BaseTransform,
    MapperConfig,
)

_TRANSFORMS_REGISTRY, register_transform, build_transform = make_registry(
    "transform", BaseTransform
)
bootstrap_module(__name__)
