import functools
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import omegaconf

registries = {}


@dataclass
class RegistryConfig(dict[str, Any]):
    _name: str | None = None


C = TypeVar("C", bound=type)
T = TypeVar("T")


def make_registry(registry_name: str, base_class: type | None = None):
    if registry_name in registries:
        registry = registries[registry_name]
    else:
        registries[registry_name] = {}
        registry = registries[registry_name]

    if base_class is None:
        base_class = Any

    def register_arch(name: str, class_name: str, registered_class: type):
        full_name = f"{class_name}-{name}"

        assert (
            full_name not in registry
        ), f"Double registering of {full_name} in {registry_name} registry"

        base_registry = registry.get(class_name, None)
        assert (
            base_registry is not None
        ), f"Base class {class_name} not found in {registry_name} registry"
        assert (
            base_registry[1] is not None
        ), f"Base class {class_name} must have a structured config in {registry_name} registry to register architectures"

        def wrapper(method):
            cfg = method()
            assert cfg is None or isinstance(
                cfg, RegistryConfig
            ), "Configs must be subclasses of RegistryConfig"
            registry[full_name] = (registered_class, cfg)
            return method

        return wrapper

    def register(name: str, cfg: type | None = None):
        assert (
            name not in registry
        ), f"Double registering of {name} in {registry_name} registry"
        assert cfg is None or issubclass(
            cfg, RegistryConfig
        ), "Configs must be subclasses of RegistryConfig"

        def wrapper(registered_class: C) -> C:
            registry[name] = (registered_class, cfg)
            registered_class.register_arch = functools.partial(
                register_arch,
                class_name=name,
                registered_class=registered_class,
            )
            return registered_class

        return wrapper

    @overload
    def build(
        cfg: RegistryConfig,
        cast_to: type[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def build(
        cfg: RegistryConfig,
        cast_to: None = None,
        **kwargs: Any,
    ) -> Any: ...

    def build(
        cfg: RegistryConfig | None, cast_to: type[T] | None = None, **kwargs
    ) -> T | Any | None:
        if cfg is None:
            return None
        if isinstance(cfg, str):
            name = cfg
        else:
            if not omegaconf.OmegaConf.is_config(cfg):
                cfg = omegaconf.OmegaConf.structured(cfg)
            name = cfg._name
        assert name in registry, f"Unknown {name} in {registry_name} registry"
        registered_class, structured_cfg = registry[name]
        if type(structured_cfg) is type:
            structured_cfg = omegaconf.OmegaConf.merge(
                omegaconf.OmegaConf.structured(structured_cfg), structured_cfg()
            )
        if structured_cfg is not None:
            cfg = omegaconf.OmegaConf.merge(
                structured_cfg, omegaconf.OmegaConf.to_container(cfg=cfg, resolve=True)
            )
            obj = registered_class(cfg, **kwargs)
        else:
            obj = registered_class(**kwargs)

        if cast_to is not None:
            assert isinstance(obj, cast_to), f"Expected {cast_to}, got {type(obj)}"
        return obj

    return registry, register, build


@overload
def build(
    registry_name: str,
    cfg: RegistryConfig,
    cast_to: type[T],
    **kwargs: Any,
) -> T: ...


@overload
def build(
    registry_name: str,
    cfg: RegistryConfig,
    cast_to: None = None,
    **kwargs: Any,
) -> Any: ...


def build(
    registry_name: str,
    cfg: RegistryConfig,
    cast_to: type[T] | None = None,
    **kwargs: Any,
) -> T | Any | None:
    assert registry_name in registries, f"Unknown registry {registry_name}"
    _, _, build_fn = make_registry(registry_name)
    obj = build_fn(cfg, **kwargs)
    if cast_to is not None:
        assert isinstance(obj, cast_to), f"Expected {cast_to}, got {type(obj)}"
    return obj
