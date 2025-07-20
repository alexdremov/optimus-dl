from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

(
    _TRAIN_RECIPE,
    register_train_recipe,
    build_train_recipe,
) = make_registry("train_recipe")


bootstrap_module(__name__)
