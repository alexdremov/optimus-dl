from optimus_dl.core.registry import make_registry

(
    _dataset_sampling_strategy,
    register_dataset_sampling_strategy,
    build_dataset_sampling_strategy,
) = make_registry("dataset_sampling_strategy")

__all__ = ["register_dataset_sampling_strategy", "build_dataset_sampling_strategy"]
