import omegaconf

from optimus_dl.core.omegaconf import non_resolving_instantiate


def test_brace_in_string():
    config = omegaconf.OmegaConf.create(
        {"a": "}", "obj": {"_target_": "builtins.dict", "val": "${eval:'\"}\"'}"}}
    )

    inst = non_resolving_instantiate(config)
    assert inst.obj.val == "}"


def test_brace_in_string_with_relative_path():
    config = omegaconf.OmegaConf.create(
        {
            "x": "hello",
            "obj": {
                "_target_": "builtins.dict",
                # The string is `} ${x} {`
                # _extract_interpolations might break here if it doesn't handle quotes
                "val": '${eval:\'"} " + "${x}" + " {"\'}',
            },
        }
    )

    inst = non_resolving_instantiate(config)
    assert inst.obj.val == "} hello {"
