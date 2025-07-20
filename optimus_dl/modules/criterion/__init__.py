from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import BaseCriterion
from .config import CriterionConfig

_CRITERIONS_REGISTRY, register_criterion, build_criterion = make_registry("criterion")

bootstrap_module(__name__)
