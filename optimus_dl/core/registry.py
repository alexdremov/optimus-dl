"""Registry system for dependency injection and component management.

This module provides a flexible registry pattern that allows components (models,
optimizers, data loaders, etc.) to be registered and instantiated from configuration.
The registry system is the foundation of Optimus-DL's modular architecture, enabling
easy component swapping and configuration-driven instantiation.

Example:
    Basic usage::

        # Create a registry
        registry, register, build = make_registry("my_component")

        # Register a component
        @register("my_impl", MyConfig)
        class MyImplementation:
            def __init__(self, cfg: MyConfig):
                self.cfg = cfg

        # Build from config
        config = MyConfig(_name="my_impl", param=1)
        obj = build(config)
"""

import functools
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import omegaconf

# Global registry storage: maps registry_name -> {component_name -> (class, config_class)}
registries = {}


@dataclass
class RegistryConfigStrict:
    """Strict configuration base class for registry components.

    This is a minimal configuration class that only requires a `_name` field.
    Use this when you don't need additional configuration fields.

    Attributes:
        _name: The registered name of the component to instantiate.
    """
    _name: str | None = None


@dataclass
class RegistryConfig(RegistryConfigStrict, dict[str, Any]):
    """Flexible configuration class for registry components.

    This extends RegistryConfigStrict to allow arbitrary additional fields
    via dictionary inheritance. Use this when you need custom configuration
    parameters beyond static fields.

    Attributes:
        _name: The registered name of the component to instantiate.
        Additional fields can be added as dictionary keys.

    Example:
        >>> config = RegistryConfig(_name="my_model", n_layers=12, hidden_size=768)
        >>> config["n_layers"]  # 12
    """
    ...


C = TypeVar("C", bound=type)
T = TypeVar("T")


def make_registry(registry_name: str, base_class: type | None = None):
    """Create or retrieve a component registry.

    This function creates a new registry or retrieves an existing one. Each registry
    maintains a mapping of component names to their classes and configuration classes.
    The registry pattern enables dependency injection and configuration-driven
    component instantiation.

    Args:
        registry_name: Unique name for the registry (e.g., "model", "optimizer").
        base_class: Optional base class that all registered components must inherit
            from. Used for type checking. If None, any class can be registered.

    Returns:
        A tuple of (registry_dict, register_decorator, build_function):
        - registry_dict: The actual registry dictionary (for inspection/debugging)
        - register: Decorator function to register components
        - build: Function to build instances from configuration

    Example:
        >>> registry, register, build = make_registry("model", BaseModel)
        >>>
        >>> @register("llama", LlamaConfig)
        >>> class Llama(BaseModel):
        ...     def __init__(self, cfg: LlamaConfig):
        ...         self.cfg = cfg
        >>>
        >>> # Later, build from config
        >>> config = LlamaConfig(_name="llama", n_embd=512)
        >>> model = build(config)

    Note:
        The registry is stored globally. Multiple calls with the same `registry_name`
        will return the same registry instance.
    """
    if registry_name in registries:
        registry = registries[registry_name]
    else:
        registries[registry_name] = {}
        registry = registries[registry_name]

    if base_class is None:
        base_class = Any

    def register_arch(name: str, class_name: str, registered_class: type):
        """Register an architecture variant of a base class.

        This allows registering multiple variants (architectures) of a base class.
        For example, you might have a base "llama" model with variants like
        "llama-7b", "llama-13b", "llama-70b".

        Args:
            name: Name of the architecture variant.
            class_name: Name of the base class this variant belongs to.
            registered_class: The class implementing this architecture.

        Returns:
            A decorator function that takes a config factory method.

        Raises:
            AssertionError: If the architecture is already registered or if the
                base class doesn't exist or doesn't have a structured config.

        Example:
            >>> @register("llama", LlamaConfig)
            >>> class Llama(BaseModel):
            ...     pass
            >>>
            >>> @Llama.register_arch("7b")
            >>> def llama_7b_config():
            ...     return LlamaConfig(n_layers=32, n_embd=4096)
        """
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
            assert (
                cfg is None
                or isinstance(cfg, RegistryConfig)
                or isinstance(cfg, RegistryConfigStrict)
            ), "Configs must be subclasses of RegistryConfig"
            registry[full_name] = (registered_class, cfg)
            return method

        return wrapper

    def register(name: str, cfg: type | None = None):
        """Register a component in the registry.

        This is the main decorator for registering components. It associates a
        component class with a name and optional configuration class.

        Args:
            name: Unique name to register the component under (e.g., "llama", "adamw").
            cfg: Optional configuration class that must be a subclass of
                RegistryConfig or RegistryConfigStrict. If None, the component
                will be instantiated without configuration.

        Returns:
            A decorator function that registers the decorated class.

        Raises:
            AssertionError: If the name is already registered or if cfg is not
                a valid config class.

        Example:
            >>> @register("my_model", MyModelConfig)
            >>> class MyModel:
            ...     def __init__(self, cfg: MyModelConfig):
            ...         self.cfg = cfg
        """
        assert (
            name not in registry
        ), f"Double registering of {name} in {registry_name} registry"
        assert (
            cfg is None
            or issubclass(cfg, RegistryConfig)
            or issubclass(cfg, RegistryConfigStrict)
        ), "Configs must be subclasses of RegistryConfig or RegistryConfigStrict"

        def wrapper(registered_class: C) -> C:
            """Decorator that registers a class in the registry.

            Args:
                registered_class: The class to register.

            Returns:
                The same class (for use as a decorator).
            """
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
        cfg: RegistryConfig | RegistryConfigStrict | dict,
        cast_to: type[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def build(
        cfg: RegistryConfig | RegistryConfigStrict | dict,
        cast_to: None = None,
        **kwargs: Any,
    ) -> Any: ...

    def build(
        cfg: RegistryConfig | RegistryConfigStrict | dict | None,
        cast_to: type[T] | None = None,
        **kwargs,
    ) -> T | Any | None:
        """Build a component instance from configuration.

        This function instantiates a registered component based on its configuration.
        It handles merging default configuration values with provided overrides,
        and supports both structured configs and plain dictionaries.

        Args:
            cfg: Configuration object containing `_name` field specifying which
                component to build. Can be a RegistryConfig, dict, or None.
                If None, returns None. If a string, treats it as the component name.
            cast_to: Optional type to cast the result to. If provided, raises
                AssertionError if the built object is not an instance of this type.
            **kwargs: Additional keyword arguments passed to the component constructor.

        Returns:
            An instance of the registered component, or None if cfg is None.

        Raises:
            AssertionError: If the component name is not found in the registry, or
                if cast_to is provided and the built object is not of that type.

        Example:
            >>> config = RegistryConfig(_name="llama", n_embd=512)
            >>> model = build(config, cast_to=BaseModel)
            >>> assert isinstance(model, Llama)
        """
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
    cfg: RegistryConfig | RegistryConfigStrict | dict,
    cast_to: type[T],
    **kwargs: Any,
) -> T: ...


@overload
def build(
    registry_name: str,
    cfg: RegistryConfig | RegistryConfigStrict | dict,
    cast_to: None = None,
    **kwargs: Any,
) -> Any: ...


def build(
    registry_name: str,
    cfg: RegistryConfig | RegistryConfigStrict | dict,
    cast_to: type[T] | None = None,
    **kwargs: Any,
) -> T | Any | None:
    """Build a component from a named registry.

    This is a convenience function that builds a component from a registry by name.
    It's useful when you know the registry name but don't have the registry
    functions directly available.

    Args:
        registry_name: Name of the registry to build from (e.g., "model", "optimizer").
        cfg: Configuration object with `_name` field specifying the component.
        cast_to: Optional type to cast the result to.
        **kwargs: Additional arguments passed to the component constructor.

    Returns:
        An instance of the registered component.

    Raises:
        AssertionError: If the registry doesn't exist, the component name is not
            found, or if cast_to is provided and the object is not of that type.

    Example:
        >>> config = RegistryConfig(_name="llama", n_embd=512)
        >>> model = build("model", config, cast_to=BaseModel)
    """
    assert registry_name in registries, f"Unknown registry {registry_name}"
    _, _, build_fn = make_registry(registry_name)
    obj = build_fn(cfg, **kwargs)
    if cast_to is not None:
        assert isinstance(obj, cast_to), f"Expected {cast_to}, got {type(obj)}"
    return obj
