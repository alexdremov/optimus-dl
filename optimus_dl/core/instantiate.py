"""
Utility functions for configuration instantiation.

These functions are intended to be used as `_target_`s in OmegaConf/Hydra configs
to provide structural and logical operations during object instantiation.
"""

import copy
from collections.abc import (
    Callable,
    Mapping,
)
from typing import Any

from omegaconf import OmegaConf


def merge_dicts(*dicts: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep-merge multiple dictionaries into a single dictionary.
    Later dictionaries in the arguments override earlier ones.

    Usage in config:
        my_kwargs:
          _target_: optimus_dl.core.instantiate.merge_dicts
          _args_:
            - ${defaults.kwargs}
            - {specific_override: 42}
    """
    result = OmegaConf.create({})
    for d in dicts:
        # Convert to config objects to safely merge if they aren't already
        if not OmegaConf.is_config(d):
            d = OmegaConf.create(d)
        result = OmegaConf.merge(result, d)

    # Return as standard dict so it plays nicely with other kwargs
    return OmegaConf.to_container(result, resolve=True)  # type: ignore


def cond(condition: bool, true_val: Any, false_val: Any) -> Any:
    """
    Conditional instantiation.

    Because optimus_dl uses lazy instantiation, the branch that is not
    selected will not be instantiated or evaluated, making this a powerful
    tool for switching out heavy sub-modules like layers or optimizers.

    Usage in config:
        layer:
          _target_: optimus_dl.core.instantiate.cond
          condition: ${use_flash_attn}
          true_val:
            _target_: optimus_dl.modules.FlashAttention
          false_val:
            _target_: optimus_dl.modules.StandardAttention
    """
    if condition:
        return true_val
    else:
        return false_val


def repeat(times: int, item: Any) -> list[Any]:
    """
    Create a list by repeating an item a specific number of times.

    If the item is a dictionary or an OmegaConf config (which often represents
    an object to be instantiated), it will be deep-copied so that each
    repeated element is an independent instance.

    Usage in config:
        layers:
          _target_: optimus_dl.core.instantiate.repeat
          times: ${model.n_layers}
          item:
            _target_: optimus_dl.modules.TransformerLayer
    """
    result = []
    for _ in range(times):
        # Deepcopy ensures that mutable objects (or DictConfigs that get
        # instantiated later) are distinct, separate instances.
        result.append(copy.deepcopy(item))
    return result


def concat_lists(*lists: list[Any]) -> list[Any]:
    """
    Concatenate multiple lists into a single flat list.

    Usage in config:
        all_metrics:
          _target_: optimus_dl.core.instantiate.concat_lists
          _args_:
            - ${metrics.train}
            - ${metrics.val}
    """
    result = []
    for lst in lists:
        if OmegaConf.is_config(lst):
            lst = OmegaConf.to_container(lst, resolve=True)  # type: ignore
        if not isinstance(lst, list):
            raise ValueError(f"Expected a list, but got {type(lst).__name__}")
        result.extend(lst)
    return result


def get_item(key: str, mapping: Mapping[str, Any], default: Any = None) -> Any:
    """
    Retrieve an item from a mapping based on a key.

    Acts as a dynamic switch/router. If the key is not found and no default
    is provided, it raises a KeyError.

    Usage in config:
        optimizer:
          _target_: optimus_dl.core.instantiate.get_item
          key: ${args.optim_type}
          mapping:
            adamw:
              _target_: torch.optim.AdamW
            sgd:
              _target_: torch.optim.SGD
    """
    if OmegaConf.is_config(mapping):
        mapping = OmegaConf.to_container(mapping, resolve=True)  # type: ignore

    if key in mapping:
        return mapping[key]
    if default is not None:
        return default

    raise KeyError(f"Key '{key}' not found in mapping and no default provided.")


def compose(*functions: Callable) -> Callable:
    """
    Compose multiple callables into a single pipeline.

    The functions are applied in the order they are provided (left-to-right).
    i.e., compose(f, g)(x) == g(f(x)).

    This is highly useful for data transformation pipelines.

    Usage in config:
        transform:
          _target_: optimus_dl.core.instantiate.compose
          _args_:
            - _target_: optimus_dl.data.transforms.Tokenize
            - _target_: optimus_dl.data.transforms.Chunk
    """

    def composed(*args: Any, **kwargs: Any) -> Any:
        if not functions:
            # If no functions, act as identity for a single arg
            if len(args) == 1 and not kwargs:
                return args[0]
            raise ValueError(
                "No functions provided to compose and complex arguments passed."
            )

        # Call the first function with all arguments
        result = functions[0](*args, **kwargs)

        # Subsequent functions take the single result of the previous function
        for func in functions[1:]:
            result = func(result)
        return result

    return composed
