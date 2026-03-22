import math

import omegaconf
import torch
import pytest
from omegaconf import DictConfig

from optimus_dl.core.omegaconf import non_resolving_instantiate


def test_non_resolving_instantiate():
    a = omegaconf.DictConfig(
        {"a": {"_target_": "math.floor", "_args_": [3.2]}, "b": "${.a}"}
    )
    # Default is now lazy=True
    b = non_resolving_instantiate(a)

    # After instantiation, b.a should resolve to 3
    assert b.a == 3
    # b.b should resolve to 3 since it points to b.a
    assert b.b == 3

    # Updating b.a = 10 replaces the resolver string with 10.
    b.a = 10
    assert b.b == 10

    # The original structure should remain a DictConfig
    assert isinstance(b, DictConfig)


def test_non_resolving_instantiate_list():
    a = omegaconf.DictConfig(
        {
            "my_list": [{"_target_": "math.floor", "_args_": [3.2]}, "${..a}"],
            "a": 100,
        }
    )
    b = non_resolving_instantiate(a)
    assert b.my_list[0] == 3
    assert b.my_list[1] == 100

    b.a = 200
    assert b.my_list[1] == 200


def test_complex_linking():
    config = omegaconf.OmegaConf.create(
        {
            "params": {"val": 10.5},
            "module_a": {"_target_": "math.ceil", "_args_": ["${params.val}"]},
            "module_b": {"_target_": "math.floor", "_args_": ["${params.val}"]},
            "summary": {
                "a": "${module_a}",
                "b": "${module_b}",
                "sum": "${eval:'${summary.a} + ${summary.b}'}",
            },
        }
    )

    inst = non_resolving_instantiate(config)

    # Check initial instantiation
    assert inst.module_a == 11
    assert inst.module_b == 10
    assert inst.summary.a == 11
    assert inst.summary.b == 10
    assert inst.summary.sum == 21

    # module_a and module_b are lazy. summary.a still links to module_a.
    inst.params.val = 20.1
    assert inst.module_a == 21
    assert inst.module_b == 20
    assert inst.summary.a == 21
    assert inst.summary.b == 20
    assert inst.summary.sum == 41


def test_nested_instantiation_lazy():
    # Use absolute-style interpolation for nested tests to avoid relative path issues in ghost configs
    config = omegaconf.OmegaConf.create(
        {
            "p": 16,
            "outer": {
                "_target_": "builtins.dict",
                "inner": {"_target_": "math.sqrt", "_args_": ["${p}"]},
            },
            "ref": "${outer.inner}",
        }
    )

    inst = non_resolving_instantiate(config)

    assert inst.outer.inner == 4.0
    assert inst.ref == 4.0

    # Test reactivity
    inst.p = 25
    assert inst.outer.inner == 5.0


def test_cross_file_style_links():
    # Test with lazy=False for strict type checking
    config = omegaconf.OmegaConf.create(
        {
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "params": [
                    {
                        "_target_": "torch.nn.Parameter",
                        "_args_": ["${eval:'torch.randn(10)'}"],
                    }
                ],
                "lr": "${hparams.lr}",
            },
            "hparams": {"lr": 0.01},
        }
    )

    inst = non_resolving_instantiate(config, lazy=False)

    assert isinstance(inst.optimizer, torch.optim.Adam)
    assert inst.optimizer.defaults["lr"] == 0.01


def test_lazy_non_resolving_instantiate():
    config = omegaconf.OmegaConf.create(
        {"val": 10.5, "obj": {"_target_": "math.ceil", "_args_": ["${val}"]}}
    )

    # Instantiate with lazy=True (default)
    inst = non_resolving_instantiate(config)

    # Check initial value
    assert inst.obj == 11

    # Change dependency
    inst.val = 20.1
    # It should re-instantiate on access
    assert inst.obj == 21

    inst.val = 30.5
    assert inst.obj == 31


def test_lazy_complex_objects():
    config = omegaconf.OmegaConf.create(
        {
            "hparams": {"lr": 0.01},
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "params": [
                    {
                        "_target_": "torch.nn.Parameter",
                        "_args_": ["${eval:'torch.randn(5)'}"],
                    }
                ],
                "lr": "${hparams.lr}",
            },
        }
    )

    inst = non_resolving_instantiate(config)

    # optimizer is lazy. Accessing it returns an SGD instance.
    opt1 = inst.optimizer
    assert opt1.defaults["lr"] == 0.01

    # Update lr
    inst.hparams.lr = 0.05
    # Next access should return a NEW instance with new lr
    opt2 = inst.optimizer
    assert opt2.defaults["lr"] == 0.05
    assert opt1 is not opt2


def test_circular_dependency():
    config = omegaconf.OmegaConf.create(
        {
            "a": {"_target_": "builtins.dict", "ref": "${b}"},
            "b": {"_target_": "builtins.dict", "ref": "${a}"},
        }
    )

    inst = non_resolving_instantiate(config)

    with pytest.raises(ValueError, match="Circular dependency detected"):
        print(inst.a)


def test_relative_interpolation_conversion():
    # Test that relative interpolations in lazy nodes are correctly converted to absolute
    # even when the node is moved to ghost storage.
    config = omegaconf.OmegaConf.create(
        {
            "params": {"x": 10},
            "outer": {
                "y": 5,
                "node": {
                    "_target_": "builtins.dict",
                    "val_x": "${...params.x}",  # node(1) -> outer(2) -> root(3)
                    "z": 2,
                    "val_z": "${.z}",  # sibling
                },
            },
        }
    )

    inst = non_resolving_instantiate(config)
    # Check that it works!
    assert inst.outer.node.val_x == 10
    assert inst.outer.node.val_z == 2

    # Test reactivity
    inst.params.x = 20
    assert inst.outer.node.val_x == 20


def test_deep_nested_complex():
    config = omegaconf.OmegaConf.create(
        {
            "a": {
                "b": [
                    {
                        "_target_": "math.sqrt",
                        "_args_": ["${.....c}"],
                    },  # args(1) -> list_elem(2) -> list_b(3) -> dict_a(4) -> root(5)
                    100,
                ]
            },
            "c": 64,
        }
    )

    inst = non_resolving_instantiate(config)
    assert inst.a.b[0] == 8.0
    inst.c = 81
    assert inst.a.b[0] == 9.0


def test_list_of_targets():
    config = omegaconf.OmegaConf.create(
        {
            "models": [
                {"_target_": "math.ceil", "_args_": [1.1]},
                {"_target_": "math.floor", "_args_": [1.9]},
            ]
        }
    )
    inst = non_resolving_instantiate(config)
    assert inst.models[0] == 2
    assert inst.models[1] == 1


def test_absolute_path_to_local_param():
    # A _target_ block at a.b uses an absolute path to its own parameter
    config = omegaconf.OmegaConf.create(
        {"a": {"b": {"_target_": "builtins.dict", "val": 10.5, "ref": "${a.b.val}"}}}
    )

    inst = non_resolving_instantiate(config)
    assert inst.a.b.val == 10.5
    assert inst.a.b.ref == 10.5

    config2 = omegaconf.OmegaConf.create(
        {
            "params": {"v": 10.5},
            "a": {"b": {"_target_": "math.ceil", "_args_": ["${params.v}"]}},
        }
    )
    inst2 = non_resolving_instantiate(config2)
    assert inst2.a.b == 11
    inst2.params.v = 20.1
    assert inst2.a.b == 21


def test_root_isolation():
    # Different root configurations should not interfere with each other's lazy nodes
    cfg1 = omegaconf.OmegaConf.create(
        {"v": 10, "obj": {"_target_": "math.ceil", "_args_": ["${v}"]}}
    )
    cfg2 = omegaconf.OmegaConf.create(
        {"v": 100, "obj": {"_target_": "math.ceil", "_args_": ["${v}"]}}
    )

    inst1 = non_resolving_instantiate(cfg1)
    inst2 = non_resolving_instantiate(cfg2)

    assert inst1.obj == 10
    assert inst2.obj == 100

    inst1.v = 20
    assert inst1.obj == 20
    assert inst2.obj == 100  # cfg2 remains isolated


def test_list_sibling_reference():
    # Absolute path from a sibling list element
    config = omegaconf.OmegaConf.create(
        {
            "data_items": [
                {"_target_": "builtins.dict", "val": 10},
                {"_target_": "builtins.dict", "ref": "${data_items[0].val}"},
            ]
        }
    )

    inst = non_resolving_instantiate(config)
    assert inst.data_items[0].val == 10
    assert inst.data_items[1].ref == 10

    # Reactivity
    inst.data_items[0].val = 20
    assert inst.data_items[1].ref == 20


def test_nested_eval_resolvers():
    # Resolvers using other resolvers as arguments
    config = omegaconf.OmegaConf.create(
        {
            "base": 10,
            "offset": 5,
            "obj": {
                "_target_": "math.ceil",
                "_args_": ["${eval:'${base} + ${eval:\"${offset} * 2\"}'}"],
            },
        }
    )

    inst = non_resolving_instantiate(config)
    # 10 + (5 * 2) = 20
    assert inst.obj == 20

    inst.base = 30
    assert inst.obj == 40

    inst.offset = 10
    # 30 + (10 * 2) = 50
    assert inst.obj == 50


def test_nested_lazy_recursive():
    # _target_ within _target_ where both are handled by non_resolving_instantiate
    config = omegaconf.OmegaConf.create(
        {
            "outer": {
                "_target_": "builtins.dict",
                "inner": {"_target_": "builtins.dict", "val": "${params.v}"},
            },
            "params": {"v": 1},
        }
    )

    inst = non_resolving_instantiate(config)
    assert inst.outer.inner.val == 1

    inst.params.v = 2
    assert inst.outer.inner.val == 2


def test_nested_interpolation_indices():
    # Interpolations inside list indices: ${data[${idx}]}
    # Note: OmegaConf requires interpolations used as keys/indices to resolve to strings.
    config = omegaconf.OmegaConf.create(
        {
            "idx": "0",
            "data": [10, 20],
            "obj": {"_target_": "math.ceil", "_args_": ["${data[${idx}]}"]},
        }
    )

    inst = non_resolving_instantiate(config)
    assert inst.obj == 10

    inst.idx = "1"
    assert inst.obj == 20
