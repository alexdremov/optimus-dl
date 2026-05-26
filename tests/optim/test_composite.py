import torch
import pytest
import torch.nn as nn

from optimus_dl.modules.optim.adamw import AdamWConfig
from optimus_dl.modules.optim.composite import (
    CompositeOptimizer,
    CompositeOptimizerConfig,
    CompositeOptimizerConfigEntry,
    get_subgroup,
    make_composite_optimizer,
)


@pytest.fixture(
    params=[
        {"_name": "adamw", "lr": 1e-2},
        {"_name": "muon", "lr": 1e-2},
    ]
)
def weight_opt_cfg(request):
    return request.param


@pytest.fixture(
    params=[
        {"_name": "adamw", "lr": 1e-3},
    ]
)
def bias_opt_cfg(request):
    return request.param


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

    def test_make_composite_optimizer(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)

        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=weight_opt_cfg,
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=bias_opt_cfg,
                ),
            }
        )

        optimizer = make_composite_optimizer(config, params=model.named_parameters())
        assert isinstance(optimizer, CompositeOptimizer)
        assert "weight_opt" in optimizer.optimizers
        assert "bias_opt" in optimizer.optimizers

        assert optimizer.optimizers["weight_opt"].param_groups[0]["lr"] == 1e-2
        assert optimizer.optimizers["bias_opt"].param_groups[0]["lr"] == 1e-3

    def test_param_groups_reference(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=weight_opt_cfg,
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=bias_opt_cfg,
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

    def test_state_dict_loading(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=weight_opt_cfg,
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=bias_opt_cfg,
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

    def test_param_groups_reference_after_load(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=weight_opt_cfg,
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=bias_opt_cfg,
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

    def test_overlapping_regexes(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "opt1": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-2),
                ),
                "opt2": CompositeOptimizerConfigEntry(
                    params_regexp=".*weigh.*",  # Overlaps with weight
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-3),
                ),
            }
        )
        with pytest.raises(ValueError, match="matched by multiple optimizers"):
            make_composite_optimizer(config, params=model.named_parameters())

    def test_unmatched_parameters(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "opt1": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",  # bias is unmatched
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-2),
                ),
            }
        )
        with pytest.raises(ValueError, match="not matched by any optimizer regex"):
            make_composite_optimizer(config, params=model.named_parameters())

    def test_empty_matches(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "opt1": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-2),
                ),
                "opt2": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-3),
                ),
                "opt3": CompositeOptimizerConfigEntry(
                    params_regexp=".*nonexistent.*",
                    optimizer_config=AdamWConfig(_name="adamw", lr=1e-3),
                ),
            }
        )
        with pytest.raises(ValueError, match="did not match any parameters"):
            make_composite_optimizer(config, params=model.named_parameters())

    def test_empty_optimizer_dict(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(optimizers={})

        with pytest.raises(ValueError):
            make_composite_optimizer(config, params=model.named_parameters())

        with pytest.raises(
            ValueError, match="At least one optimizer must be provided."
        ):
            make_composite_optimizer(config, params=[])

    def test_step_execution(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=weight_opt_cfg,
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=bias_opt_cfg,
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())

        initial_weight = model.weight.clone().detach()
        initial_bias = model.bias.clone().detach()

        loss = model(torch.randn(3, 10)).sum()
        loss.backward()
        optimizer.step()

        assert not torch.allclose(model.weight, initial_weight)
        assert not torch.allclose(model.bias, initial_bias)

    def test_zero_grad_behavior(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=weight_opt_cfg,
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config=bias_opt_cfg,
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())

        loss = model(torch.randn(3, 10)).sum()
        loss.backward()

        assert model.weight.grad is not None
        assert model.bias.grad is not None

        optimizer.zero_grad(set_to_none=True)

        assert model.weight.grad is None
        assert model.bias.grad is None

    def test_scheduler_integration(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config={"_name": "adamw", "lr": 1.0},
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config={"_name": "adamw", "lr": 0.1},
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=10
        )

        scheduler.step()

        comp_weight_lr = None
        for group in optimizer.param_groups:
            if group["composite_optimizer_name"] == "weight_opt":
                comp_weight_lr = group["lr"]

        sub_weight_lr = optimizer.optimizers["weight_opt"].param_groups[0]["lr"]

        assert comp_weight_lr == sub_weight_lr
        assert comp_weight_lr != 1.0

    def test_complex_parameter_group_inputs(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

        # Add custom keys to param groups
        params = [
            {
                "params": list(model[0].named_parameters()),
                "my_custom_key": "layer1",
                "lr": 0.5,
            },
            {"params": list(model[1].named_parameters()), "my_custom_key": "layer2"},
        ]

        config = CompositeOptimizerConfig(
            optimizers={
                "all_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*",
                    optimizer_config={"_name": "adamw", "lr": 1e-2},
                ),
            }
        )

        optimizer = make_composite_optimizer(config, params=params)

        # Composite should have 2 groups
        assert len(optimizer.param_groups) == 2

        group1 = optimizer.param_groups[0]
        group2 = optimizer.param_groups[1]

        assert group1.get("my_custom_key") == "layer1"
        assert group2.get("my_custom_key") == "layer2"

        assert group1["lr"] == 0.5  # should inherit override
        assert group2["lr"] == 1e-2  # should fallback to config defaults

    def test_nested_composite_optimizer(self):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "nested_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config={
                        "_name": "composite",
                        "optimizers": {
                            "inner_opt": {
                                "params_regexp": ".*",
                                "optimizer_config": {"_name": "adamw", "lr": 1e-2},
                            }
                        },
                    },
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*bias",
                    optimizer_config={"_name": "adamw", "lr": 1e-3},
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())
        assert isinstance(optimizer.optimizers["nested_opt"], CompositeOptimizer)
        assert "inner_opt" in optimizer.optimizers["nested_opt"].optimizers

        # Step
        loss = model(torch.randn(3, 10)).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()

        # Load
        model2 = nn.Linear(10, 5)
        optimizer2 = make_composite_optimizer(config, params=model2.named_parameters())
        optimizer2.load_state_dict(state_dict)

        # Check references through outer -> nested -> inner
        # We manually change the lr on the outermost composite group
        for group in optimizer2.param_groups:
            if group.get("composite_optimizer_name") == "nested_opt":
                group["lr"] = 1.0

        # Now verify it's propagated to the inner-most optimizer inside the nested structure
        nested_opt = optimizer2.optimizers["nested_opt"]
        assert nested_opt.param_groups[0]["lr"] == 1.0

        inner_opt = nested_opt.optimizers["inner_opt"]
        assert inner_opt.param_groups[0]["lr"] == 1.0

    def test_none_regexp_matches_all_remaining(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "weight_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*weight",
                    optimizer_config=weight_opt_cfg,
                ),
                "bias_opt": CompositeOptimizerConfigEntry(
                    params_regexp=None,  # Match remaining (bias)
                    optimizer_config=bias_opt_cfg,
                ),
            }
        )
        optimizer = make_composite_optimizer(config, params=model.named_parameters())
        assert "weight_opt" in optimizer.optimizers
        assert "bias_opt" in optimizer.optimizers

        # Verify weight_opt has weight and bias_opt has bias
        weight_params = [
            p
            for group in optimizer.optimizers["weight_opt"].param_groups
            for p in group["params"]
        ]
        bias_params = [
            p
            for group in optimizer.optimizers["bias_opt"].param_groups
            for p in group["params"]
        ]

        assert len(weight_params) == 1
        assert len(bias_params) == 1
        assert any(p is model.weight for p in weight_params)
        assert any(p is model.bias for p in bias_params)

    def test_none_regexp_only_one_allowed(self, weight_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "opt1": CompositeOptimizerConfigEntry(
                    params_regexp=None,
                    optimizer_config=weight_opt_cfg,
                ),
                "opt2": CompositeOptimizerConfigEntry(
                    params_regexp=None,
                    optimizer_config=weight_opt_cfg,
                ),
            }
        )
        with pytest.raises(
            AssertionError,
            match="Only one optimizer can have params_regexp set to None",
        ):
            make_composite_optimizer(config, params=model.named_parameters())

    def test_none_regexp_no_params_left(self, weight_opt_cfg, bias_opt_cfg):
        model = nn.Linear(10, 5)
        config = CompositeOptimizerConfig(
            optimizers={
                "all_opt": CompositeOptimizerConfigEntry(
                    params_regexp=".*",
                    optimizer_config=weight_opt_cfg,
                ),
                "none_opt": CompositeOptimizerConfigEntry(
                    params_regexp=None,
                    optimizer_config=bias_opt_cfg,
                ),
            }
        )
        # all_opt matches everything, none_opt matches nothing
        with pytest.raises(
            ValueError,
            match="Optimizer 'none_opt' with regex 'None' did not match any parameters",
        ):
            make_composite_optimizer(config, params=model.named_parameters())
