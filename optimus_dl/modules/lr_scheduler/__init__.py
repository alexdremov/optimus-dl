from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import BaseLRScheduler, BaseLRSchedulerConfig

# Create the lr_scheduler registry
_, register_lr_scheduler, build_lr_scheduler = make_registry(
    "lr_scheduler", BaseLRScheduler
)

bootstrap_module(__name__)
