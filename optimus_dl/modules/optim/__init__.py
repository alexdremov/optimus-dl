from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .config import OptimizationConfig

_OPTIMIZER_REGISTRY, register_optimizer, build_optimizer = make_registry("optimizer")


bootstrap_module(__name__)
