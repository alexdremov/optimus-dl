import omegaconf

from optimus_dl.core.omegaconf import non_resolving_instantiate


def test_root_target_returns_wrapper():
    """
    Test to verify that if the root configuration has a `_target_`,
    `non_resolving_instantiate` eagerly instantiates it without a 'root' wrapper.
    """
    config = omegaconf.OmegaConf.create({"_target_": "builtins.dict", "key": "value"})

    # Instantiate with lazy=True
    inst = non_resolving_instantiate(config)

    # Now it should be the eagerly evaluated dict
    assert not isinstance(inst, omegaconf.DictConfig)
    assert isinstance(inst, dict)

    # The output is directly the instantiated dict
    assert inst == {"key": "value"}
