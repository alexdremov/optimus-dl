from optimus_dl.core.bootstrap import bootstrap_module

from .checkpoint_manager import CheckpointManager, CheckpointManagerConfig
from .load_strategy import LoadStrategy

bootstrap_module(__name__)
