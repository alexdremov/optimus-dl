import re
from collections.abc import (
    Callable,
    Iterable,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    cast,
)

import torch
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)

from optimus_dl.core.registry import (
    RegistryConfig,
    RegistryConfigStrict,
)
from optimus_dl.modules.optim import (
    build_optimizer,
    register_optimizer,
)


@dataclass
class CompositeOptimizerConfigEntry(RegistryConfig):
    params_regexp: str = field(
        default=".*",
        metadata={
            "help": "Regular expression to match parameter names for this optimizer."
        },
    )
    optimizer_config: RegistryConfig = field(
        default_factory=RegistryConfig,
        metadata={
            "help": "Configuration for the optimizer to apply to matched parameters."
        },
    )


@dataclass
class CompositeOptimizerConfig(RegistryConfigStrict):
    optimizers: dict[str, CompositeOptimizerConfigEntry] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of named optimizer configurations."},
    )


class CompositeOptimizer(Optimizer):
    """
    A meta-optimizer that wraps multiple PyTorch optimizers and conforms to the
    standard torch.optim.Optimizer interface. Supports named optimizers via dictionary.
    """

    optimizers: dict[str, Optimizer]

    def __init__(
        self, optimizers: Iterable[Optimizer] | dict[str, Optimizer], **kwargs
    ):
        if isinstance(optimizers, dict):
            self.optimizers = cast(dict, optimizers)
        else:
            self.optimizers = {str(i): opt for i, opt in enumerate(optimizers)}

        if not self.optimizers:
            raise ValueError("At least one optimizer must be provided.")

        # Initialize the base class
        groups = []
        for name, opt in self.optimizers.items():
            for group in opt.param_groups:
                group["composite_optimizer_name"] = name
                groups.append(group)

        # groups holds references to all parameter groups across all optimizers,
        # with an added key to identify which optimizer they belong to.
        #
        # changing parameter groups in self.param_groups will affect the original optimizers since they reference the same group dicts.
        super().__init__(groups, {})
        for opt in self.optimizers.values():
            opt.state = self.state

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Performs a single optimization step across all internal optimizers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for opt in self.optimizers.values():
            opt.step(closure=None)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clears the gradients of all optimized torch.Tensors across sub-optimizers.
        """
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restores the state dict and repairs the state dictionary reference aliases.
        """
        super().load_state_dict(state_dict)

        # super().load_state_dict() assigns a new object to self.state.
        # We must re-alias the sub-optimizers to point to this newly loaded state.
        for opt in self.optimizers.values():
            opt.state = self.state
            opt.param_groups.clear()

        # Re-alias param_groups to point to the newly loaded dicts
        for group in self.param_groups:
            name = group.get("composite_optimizer_name")
            assert name and name in self.optimizers, (
                "Each parameter group must have a 'composite_optimizer_name' key that matches a sub-optimizer. "
                f"Missing or invalid name: {name}"
            )
            self.optimizers[name].add_param_group(group)


def get_subgroup(
    params: ParamsT, predicate: Callable[[str, torch.Tensor], bool]
) -> ParamsT:
    """
    Utility function to filter parameters based on a predicate applied to their names.
    Expects params to be an iterable of (name, tensor) tuples.
    """
    if not isinstance(params, list):
        params = list(params)  # Ensure we can iterate multiple times
    is_single_group = False
    if any(isinstance(param, torch.Tensor) for param in params):
        raise ValueError(
            "CompositeOptimizer requires named optimizers with regex matching, not a flat list of parameters."
        )

    if any(isinstance(param, tuple) for param in params):
        assert all(
            isinstance(param, tuple) for param in params
        ), "If using tuples, all parameters must be tuples of (name, tensor)."
        is_single_group = True

    if any(isinstance(param, dict) for param in params):
        assert all(
            isinstance(param, dict) for param in params
        ), "If using dicts, all parameters must be dicts of {name: tensor}."

        for param_group in params:
            assert (
                "params" in param_group
            ), "Each parameter group dict must have a 'params' key."
            assert all(
                isinstance(param, tuple)
                and len(param) == 2
                and isinstance(param[0], str)
                and isinstance(param[1], torch.Tensor)
                for param in param_group["params"]
            ), "If using dicts, each 'params' entry must be a list of (name, tensor) tuples."

    if is_single_group:
        filtered_group = []
        for param_name, param in params:
            if predicate(param_name, param):
                filtered_group.append((param_name, param))
        return filtered_group

    filtered_groups = []
    for param_group in params:
        filtered_params = [
            (param_name, param)
            for param_name, param in param_group["params"]
            if predicate(param_name, param)
        ]
        if not filtered_params:
            continue  # Skip groups that have no matching parameters
        filtered_group = {k: v for k, v in param_group.items() if k != "params"} | {
            "params": filtered_params
        }
        filtered_groups.append(filtered_group)
    return filtered_groups


@register_optimizer("composite", CompositeOptimizerConfig)
def make_composite_optimizer(cfg, params: ParamsT, **kwargs):
    params = list(params)  # Ensure params is a list for multiple iterations

    optimizers = {}
    used_params = set()
    all_params = set()

    is_single_group = any(isinstance(param, tuple) for param in params)
    for param_group in params:
        if is_single_group:
            param_name, _ = param_group
            all_params.add(param_name)
        else:
            for param_name, _ in param_group["params"]:
                all_params.add(param_name)

    for name, entry in cfg.optimizers.items():
        filtered_groups = get_subgroup(
            params=params,
            predicate=lambda param_name, _, entry=entry: re.match(
                entry.params_regexp, param_name
            )
            is not None,
        )
        filtered_groups = list(filtered_groups)  # Ensure we can iterate multiple times
        has_parameters = len(filtered_groups) > 0
        if not has_parameters:
            raise ValueError(
                f"Optimizer '{name}' with regex '{entry.params_regexp}' did not match any parameters."
            )

        if is_single_group:
            for param_name, _ in filtered_groups:
                if param_name in used_params:
                    raise ValueError(
                        f"Parameter '{param_name}' matched by multiple optimizers (currently matched by '{name}')."
                    )
            used_params.update(param_name for param_name, _ in filtered_groups)
        else:
            for param_group in filtered_groups:
                for param_name, _ in param_group["params"]:
                    if param_name in used_params:
                        raise ValueError(
                            f"Parameter '{param_name}' matched by multiple optimizers (currently matched by '{name}')."
                        )
                used_params.update(
                    param_name for param_name, _ in param_group["params"]
                )

        optimizers[name] = build_optimizer(
            entry.optimizer_config,
            params=filtered_groups,
            **kwargs,
        )

    unused_params = all_params - used_params
    if unused_params:
        raise ValueError(
            f"The following parameters were not matched by any optimizer regex: {unused_params}"
        )

    return CompositeOptimizer(optimizers, **kwargs)
