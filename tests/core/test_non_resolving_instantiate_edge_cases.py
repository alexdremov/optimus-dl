import omegaconf

from optimus_dl.core.omegaconf import non_resolving_instantiate


def test_relative_path_child():
    config = omegaconf.OmegaConf.create(
        {"a": {"b": {"c": {"_target_": "builtins.dict", "val": "${.d}", "d": 42}}}}
    )

    inst = non_resolving_instantiate(config)
    assert inst.a.b.c.val == 42
