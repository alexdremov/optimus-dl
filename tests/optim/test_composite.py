import torch
import torch.nn as nn

from optimus_dl.modules.optim.adamw import AdamWConfig
from optimus_dl.modules.optim.composite import (
    CompositeOptimizer,
    CompositeOptimizerConfig,
    CompositeOptimizerConfigEntry,
    get_subgroup,
    make_composite_optimizer,
)


class TestCompositeOptimizer:
    def test_get_subgroup_single_group(self):
        model = nn.Linear(10, 5)
        params = list(model.named_parameters())

        # Predicate to match 'weight'
        weight_params = get_subgroup(params, lambda name, _: "weight" in name)
        assert len(weight_params) == 1
        assert weight_params[0][0] == "weight"

        # Predicate to match 'bias'
        bias_params = get_subgroup(params, lambda name, _: "bias" in name)
        assert len(bias_params) == 1
        assert bias_params[0][0] == "bias"

    def test_get_subgroup_multiple_groups(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        params = [
            {"params": list(model[0].named_parameters()), "lr": 0.1},
            {"params": list(model[1].named_parameters()), "lr": 0.01},
        ]

        # Match 'weight'
        weight_groups = get_subgroup(params, lambda name, _: "weight" in name)
        assert len(weight_groups) == 2
        assert len(weight_groups[0]["params"]) == 1
        assert weight_groups[0]["params"][0][0] == "weight"
        assert weight_groups[0]["lr"] == 0.1
        assert weight_groups[1]["lr"] == 0.01

    def test_make_composite_optimizer(self):
        model = nn.Linear(10, 5)

        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-2),
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-3),
                ),
            }
        )

        optimizer = make_composite_optimizer(config, params=model.named_parameters())
        assert isinstance(optimizer, CompositeOptimizer)
        assert "weight_opt" in optimizer.optimizers
        assert "bias_opt" in optimizer.optimizers

        assert optimizer.optimizers["weight_opt"].param_groups[0]["lr"] == 1e-2
        assert optimizer.optimizers["bias_opt"].param_groups[0]["lr"] == 1e-3

    def test_param_groups_reference(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-2),
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-3),
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())

        # Change lr in composite optimizer
        for group in optimizer.param_groups:
            if group["composite_optimizer_name"] == "weight_opt":
                group["lr"] = 5e-2

        # Check if underlying optimizer's lr changed
        assert optimizer.optimizers["weight_opt"].param_groups[0]["lr"] == 5e-2

    def test_state_dict_loading(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-2),
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-3),
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())

        # Step once to initialize state
        loss = model(torch.randn(3, 10)).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()

        # Create a new optimizer and load state
        model2 = nn.Linear(10, 5)
        optimizer2 = make_composite_optimizer(config, params=model2.named_parameters())
        optimizer2.load_state_dict(state_dict)

        # Check if states are linked correctly
        for _opt_name, sub_opt in optimizer2.optimizers.items():
            # The state of sub-optimizer should reference the composite's state
            assert sub_opt.state is optimizer2.state

    def test_param_groups_reference_after_load(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-2),
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-3),
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())

        # Step once
        model(torch.randn(3, 10)).sum().backward()
        optimizer.step()

        state_dict = optimizer.state_dict()

        optimizer2 = make_composite_optimizer(config, params=model.named_parameters())
        optimizer2.load_state_dict(state_dict)

        # Change lr in composite optimizer
        for group in optimizer2.param_groups:
            if group["composite_optimizer_name"] == "weight_opt":
                group["lr"] = 1.0

        # Sub-optimizer should reflect this change
        assert optimizer2.optimizers["weight_opt"].param_groups[0]["lr"] == 1.0
