from dataclasses import (
    dataclass,
    field,
)

import omegaconf
import pytest

from optimus_dl.core.registry import (
    RegistryConfig,
    RegistryConfigStrict,
    make_registry,
    validate_and_cast,
)


@dataclass
class SimpleConfig(RegistryConfigStrict):
    param_int: int = 1
    param_float: float = 1.0
    param_str: str = "default"
    param_bool: bool = False


@dataclass
class NestedConfig(RegistryConfigStrict):
    simple: SimpleConfig = field(default_factory=SimpleConfig)
    simple_list: list[SimpleConfig] = field(default_factory=list)
    simple_dict: dict[str, SimpleConfig] = field(default_factory=dict)
    optional_simple: SimpleConfig | None = None


@dataclass
class FlexibleConfig(RegistryConfig):
    known_param: int = 10


def test_primitive_casting():
    """Test that primitives are correctly cast from various types."""

    @dataclass
    class Primitives:
        i: int
        f: float
        s: str
        b: bool

    # Success case
    cfg = {"i": "10", "f": "2.5", "s": 123, "b": "true"}
    casted = validate_and_cast(Primitives, cfg)
    assert casted.i == 10
    assert casted.f == 2.5
    assert casted.s == "123"
    assert casted.b is True

    # Error case
    with pytest.raises(TypeError):
        validate_and_cast(
            Primitives, {"i": "not_an_int", "f": 1.0, "s": "hi", "b": True}
        )


def test_nested_strict_validation():
    """Test that RegistryConfigStrict subclasses validate unknown keys recursively."""
    # Top level unknown key
    with pytest.raises(ValueError) as excinfo:
        validate_and_cast(SimpleConfig, {"param_int": 1, "unknown": "error"})
    assert "Unknown keys" in str(excinfo.value)

    # Nested unknown key
    cfg = {"simple": {"param_int": 2, "unknown_nested": "error"}}
    with pytest.raises(ValueError) as excinfo:
        validate_and_cast(NestedConfig, cfg)
    assert "Unknown keys" in str(excinfo.value)
    assert "simple" in str(excinfo.value)


def test_flexible_registry_config():
    """Test that RegistryConfig (dict subclass) allows and preserves extra keys."""
    cfg = {
        "_name": "test",
        "known_param": 20,
        "extra_param": "precious_data",
        "nested_extra": {"a": 1},
    }
    casted = validate_and_cast(FlexibleConfig, cfg)
    assert isinstance(casted, FlexibleConfig)
    assert casted.known_param == 20
    assert casted["extra_param"] == "precious_data"
    assert casted["nested_extra"] == {"a": 1}
    assert casted._name == "test"


def test_recursive_collections():
    """Test validation and casting inside lists and dicts."""
    cfg = {
        "simple_list": [{"param_int": "10"}, {"param_float": "5.5"}],
        "simple_dict": {"a": {"param_str": "hello"}, "b": {"param_bool": "yes"}},
    }
    casted = validate_and_cast(NestedConfig, cfg)
    assert casted.simple_list[0].param_int == 10
    assert casted.simple_list[1].param_float == 5.5
    assert casted.simple_dict["a"].param_str == "hello"
    assert casted.simple_dict["b"].param_bool is True

    # Ensure they are instances
    assert isinstance(casted.simple_list[0], SimpleConfig)
    assert isinstance(casted.simple_dict["a"], SimpleConfig)


def test_unions_and_optionals():
    """Test handling of Union and Optional types."""

    @dataclass
    class UnionConfig:
        u: int | str
        opt: int | None = None

    # Union prefers first matching type (if castable)
    assert validate_and_cast(UnionConfig, {"u": "123"}).u == 123
    assert validate_and_cast(UnionConfig, {"u": "abc"}).u == "abc"

    # Optional
    assert validate_and_cast(UnionConfig, {"u": 1, "opt": "50"}).opt == 50
    assert validate_and_cast(UnionConfig, {"u": 1, "opt": None}).opt is None


def test_omegaconf_integration():
    """Test that OmegaConf DictConfigs are correctly handled."""
    oc = omegaconf.OmegaConf.create(
        {"param_int": "${eval:'10 + 5'}", "param_str": "dynamic"}
    )
    casted = validate_and_cast(SimpleConfig, oc)
    assert casted.param_int == 15
    assert casted.param_str == "dynamic"
    assert isinstance(casted, SimpleConfig)


def test_integration_with_build():
    """Test that the build function correctly triggers validation and casting."""
    registry, register, build = make_registry("test_build_validation")

    @register("my_component", SimpleConfig)
    class MyComponent:
        def __init__(self, cfg: SimpleConfig):
            self.cfg = cfg
            assert isinstance(cfg, SimpleConfig)

    # 1. Successful build with casting
    obj = build({"_name": "my_component", "param_int": "100"})
    assert obj.cfg.param_int == 100

    # 2. Failed build with unknown key (strict)
    with pytest.raises(
        AssertionError
    ):  # build currently uses AssertionError for key mismatches
        build({"_name": "my_component", "unknown": 1})
