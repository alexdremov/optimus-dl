"""OmegaConf custom resolvers for configuration.

This module registers custom resolvers for OmegaConf that enable advanced
configuration features like Python expression evaluation and environment
variable access.

It also provides a custom non-resolving instantiation system that allows
Hydra-style object creation while preserving OmegaConf reactivity and
interpolation integrity.

Lazy reactive Hydra instantiation for OmegaConf trees.

Goal
----
Hydra's `hydra.utils.instantiate(...)` normally resolves interpolations eagerly.
That breaks OmegaConf reactivity in some workflows.

This module implements a lazy alternative:

- Every DictConfig node with `_target_` is replaced by `${_lazy_inst:<id>}`
- Real target config is stored in a resolver-side cache
- On access, `_lazy_inst`:
  1) checks whether external dependencies changed
  2) returns cached instance if not changed
  3) re-instantiates if changed

Key guarantees
--------------
1. Internal references inside a target node remain relative/portable.
2. External references are tracked for cache invalidation.
3. Nested lazy targets continue to work after instantiation.
4. Circular lazy recursion is detected.

Important
---------
- This relies on OmegaConf internals (`_set_parent`, resolver cache usage).
- Not thread-safe by design (shared mutable cache in config object).
"""

import copy
import hashlib
import importlib
import logging
import os
from typing import Any

import hydra
from omegaconf import (
    OmegaConf,
    Container,
    DictConfig,
    ListConfig,
)
from omegaconf._utils import split_key

logger = logging.getLogger(__name__)

# Register custom resolvers for OmegaConf
# These can be used in YAML configs with ${resolver:args} syntax

OmegaConf.register_new_resolver(
    "eval",
    lambda x: eval(
        x,
        {
            "torch": importlib.import_module("torch"),
            "numpy": importlib.import_module("numpy"),
            "math": importlib.import_module("math"),
            "scipy": importlib.import_module("scipy"),
        },
    ),
)
"""Register 'eval' resolver for evaluating Python expressions in configs.

This allows you to use Python expressions in YAML configs:
    batch_size: ${eval:"64 * 2"}  # Results in 128
    seq_len: ${eval:"1024 + 512"}  # Results in 1536

Note: Use with caution as eval() can execute arbitrary code.
"""

OmegaConf.register_new_resolver("cpu_count", os.cpu_count)
"""Register 'cpu_count' resolver for getting CPU count in configs.

This allows you to reference the number of CPU cores in YAML configs:
    num_workers: ${cpu_count:}  # Uses all available CPU cores
    num_workers: ${eval:"${cpu_count:} // 2"}  # Uses half the cores
"""


def hash_resolver(x, max_len=16):
    """Resolver for computing hash of a value repr."""
    x = repr(x)
    return hashlib.sha256(x.encode("utf-8")).hexdigest()[:max_len]


OmegaConf.register_new_resolver("hash", hash_resolver)
"""Register 'hash' resolver for computing hash of a value.

This allows you to compute hashes in YAML configs:
    model_id: ${hash:${model_config}}
"""


def conf_hash_resolver(*args, _root_):
    """Resolver for computing hash of a root config."""
    max_len = 16
    if len(args) > 0:
        assert len(args) == 1, "Only one argument is allowed"
        max_len = int(args[0])
    return hash_resolver(_root_, max_len=max_len)


OmegaConf.register_new_resolver("config_hash", conf_hash_resolver)
"""Register 'config_hash' resolver for computing hash of a root config.

This allows you to compute hashes of root config in YAML configs:
    model_id: model-${config_hash:}
"""


# --- Lazy Instantiation System ---


def _copy_lazy_state(src: Any, dst: Any) -> None:
    """Recursively copy lazy definitions between caches to propagate state to instantiated containers."""
    if isinstance(src, Container) and isinstance(dst, Container):
        src_cache = OmegaConf.get_cache(src)
        dst_cache = OmegaConf.get_cache(dst)
        if "lazy_configs" in src_cache:
            dst_cache["lazy_configs"] = src_cache["lazy_configs"]

        if isinstance(src, DictConfig) and isinstance(dst, DictConfig):
            for k in src.keys():
                if k in dst:
                    _copy_lazy_state(src._get_node(k), dst._get_node(k))
        elif isinstance(src, ListConfig) and isinstance(dst, ListConfig):
            for i in range(min(len(src), len(dst))):
                _copy_lazy_state(src._get_node(i), dst._get_node(i))


def _lazy_inst_resolver(local_key: str, _root_: Any, _parent_: Any) -> Any:
    """Internal resolver for reactive lazy instantiation.

    Reads lazy state directly from the `_parent_` node's resolver cache,
    avoiding any modification to the visible configuration structure.
    """
    cache = OmegaConf.get_cache(_parent_)

    if "lazy_configs" not in cache or str(local_key) not in cache["lazy_configs"]:
        raise ValueError(
            f"Could not find lazy config definition for '{local_key}' in parent. "
            "Ensure the config was initialized with non_resolving_instantiate."
        )

    definition_node, outside_deps, kwargs = cache["lazy_configs"][str(local_key)]

    recursion_tracker = cache.setdefault("_lazy_recursion", set())
    if str(local_key) in recursion_tracker:
        raise ValueError(f"Circular dependency detected for lazy node: {local_key}")

    # Fingerprinting: Check if any external dependencies have changed.
    try:
        if outside_deps:
            temp_deps = OmegaConf.create(outside_deps)
            temp_deps._set_parent(_root_)
            current_fp = str(OmegaConf.to_container(temp_deps, resolve=True))
        else:
            current_fp = ""
    except Exception as e:
        if "recursion" in str(e).lower() or isinstance(e, RecursionError):
            raise ValueError(
                f"Circular dependency detected during dependency resolution for lazy node: {local_key}"
            ) from e
        raise

    lazy_inst_cache = cache.setdefault("_lazy_inst_cache", {})
    if str(local_key) in lazy_inst_cache:
        last_fp, instance = lazy_inst_cache[str(local_key)]
        if last_fp == current_fp:
            return instance

    # Re-instantiate:
    # Re-link definition node to the root so internal absolute interpolations work.
    definition_node._set_parent(_root_)

    recursion_tracker.add(str(local_key))
    try:
        instance = hydra.utils.instantiate(definition_node, **kwargs)
    finally:
        recursion_tracker.remove(str(local_key))

    # Wrap containers to preserve reactivity for nested lookups.
    if isinstance(instance, dict | list):
        instance = OmegaConf.create(instance)

    # Propagate lazy configs to the newly instantiated container so nested targets work.
    if isinstance(instance, Container):
        _copy_lazy_state(definition_node, instance)

    lazy_inst_cache[str(local_key)] = (current_fp, instance)
    return instance


if not OmegaConf.has_resolver("_lazy_inst"):
    OmegaConf.register_new_resolver("_lazy_inst", _lazy_inst_resolver, use_cache=False)


def _split_path(path: str) -> list[str]:
    """Split an OmegaConf path into components using internal utility."""
    if not path:
        return []
    return split_key(path)


def _join_path(parts: list[str]) -> str:
    """Join path components back into an OmegaConf dot-separated string."""
    return ".".join(parts)


def _to_absolute_key(inter_key: str, parent_path: str) -> str:
    """Convert a relative interpolation key to an absolute one."""
    if not inter_key.startswith("."):
        return inter_key

    dots = 0
    temp_key = inter_key
    while temp_key.startswith("."):
        dots += 1
        temp_key = temp_key[1:]

    parts = _split_path(parent_path)

    # OmegaConf relative paths: . is current container, .. is parent container.
    # pop n-1 parts for n dots.
    pop_count = dots - 1
    if pop_count > 0:
        parts = parts[:-pop_count]

    if temp_key:
        parts.extend(_split_path(temp_key))

    return _join_path(parts)


def _get_relative_path(from_val_path: str, to_abs_path: str) -> str:
    """Calculate a relative path (with dots) from one absolute path to another."""
    f_parts = _split_path(from_val_path)
    t_parts = _split_path(to_abs_path)

    common = 0
    for f, t in zip(f_parts, t_parts, strict=False):
        if f == t:
            common += 1
        else:
            break

    # Number of steps to go up from parent container to common prefix
    up_count = (len(f_parts) - 1) - common
    dots = up_count + 1
    rem = t_parts[common:]
    return "." * dots + _join_path(rem)


def _get_interpolations_antlr(node: Any, interpolations: list[str]) -> None:
    if type(node).__name__ == "InterpolationContext":
        interpolations.append(node.getText())
        return

    if hasattr(node, "getChildCount"):
        for i in range(node.getChildCount()):
            _get_interpolations_antlr(node.getChild(i), interpolations)


def _extract_interpolations(s: str) -> list[str]:
    """Extract top-level interpolation blocks using OmegaConf's internal parser."""
    if "${" not in s:
        return []

    try:
        from omegaconf.grammar_parser import parse

        tree = parse(s)
        inters: list[str] = []
        _get_interpolations_antlr(tree, inters)
        return inters
    except Exception:
        # Fallback if parser fails, though it shouldn't for valid omegaconf strings
        return []


def _normalize_and_collect_deps(
    obj: Any, current_path: str, ghost_root_path: str, outside_deps: list[str]
) -> Any:
    """Recursively normalize interpolations and collect outside dependencies.

    Internal paths (within the ghosted node) are made relative to keep the node portable.
    External paths are made absolute and added to the outside dependency list for fingerprinting.
    """
    if isinstance(obj, dict):
        return {
            k: _normalize_and_collect_deps(
                v,
                f"{current_path}.{k}" if current_path else str(k),
                ghost_root_path,
                outside_deps,
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [
            _normalize_and_collect_deps(
                v, f"{current_path}[{i}]", ghost_root_path, outside_deps
            )
            for i, v in enumerate(obj)
        ]
    elif isinstance(obj, str) and "${" in obj:
        parts = _split_path(current_path)
        parent_path = _join_path(parts[:-1])

        def process_inter_block(inter_block: str) -> str:
            # content from ${content}
            content = inter_block[2:-1]

            # Recursively handle nested interpolations in resolver arguments
            if ":" in content:
                resolver_parts = content.split(":", 1)
                name = resolver_parts[0]
                args = resolver_parts[1]

                if name == "_lazy_inst":
                    return f"${{{name}:{args}}}"

                nested_blocks = _extract_interpolations(args)
                for block in nested_blocks:
                    args = args.replace(block, process_inter_block(block))
                return f"${{{name}:{args}}}"

            abs_key = _to_absolute_key(content, parent_path)

            # Check if internal or external
            if (
                abs_key == ghost_root_path
                or abs_key.startswith(ghost_root_path + ".")
                or abs_key.startswith(ghost_root_path + "[")
            ):
                return f"${{{_get_relative_path(current_path, abs_key)}}}"
            else:
                outside_deps.append(f"${{{abs_key}}}")
                return f"${{{abs_key}}}"

        res = obj
        for block in _extract_interpolations(obj):
            res = res.replace(block, process_inter_block(block))
        return res
    return obj


def non_resolving_instantiate(config: Any, lazy: bool = True, **kwargs: Any) -> Any:
    """Custom instantiate function that preserves OmegaConf interpolations.

    It selectively instantiates nodes with a '_target_' using hydra.utils.instantiate,
    without eagerly resolving the entire configuration tree.
    """
    if isinstance(config, DictConfig | ListConfig):
        config_copy = copy.deepcopy(config)
        OmegaConf.set_struct(config_copy, False)
        OmegaConf.set_readonly(config_copy, False)
        config_copy._set_flag("allow_objects", True)

        if not lazy:

            def _walk_and_instantiate(node: Any) -> Any:
                if isinstance(node, DictConfig):
                    if "_target_" in node:
                        return hydra.utils.instantiate(node, **kwargs)
                    for key in list(node.keys()):
                        if not OmegaConf.is_interpolation(node, key):
                            node[key] = _walk_and_instantiate(node[key])
                    return node
                elif isinstance(node, ListConfig):
                    for i in range(len(node)):
                        if not OmegaConf.is_interpolation(node, i):
                            node[i] = _walk_and_instantiate(node[i])
                    return node
                return node

            return _walk_and_instantiate(config_copy)

        def _walk_and_ghost(
            node: Any,
            path: str,
            parent_node: Any,
            local_key: str | int,
            target_deps: list[str] | None,
        ) -> Any:
            if isinstance(node, DictConfig):
                if "_target_" in node:
                    # Bottom-Up
                    my_deps: list[str] = []
                    node_copy = copy.deepcopy(node)
                    for k in list(node_copy.keys()):
                        if not OmegaConf.is_interpolation(node_copy, k):
                            node_copy[k] = _walk_and_ghost(
                                node_copy[k],
                                f"{path}.{k}" if path else str(k),
                                node_copy,
                                k,
                                my_deps,
                            )

                    raw_data = OmegaConf.to_container(node_copy, resolve=False)
                    norm_data = _normalize_and_collect_deps(
                        raw_data, path, path, my_deps
                    )
                    norm_node = OmegaConf.create(norm_data)

                    # Store definition in the parent container's resolver cache
                    cache = OmegaConf.get_cache(parent_node)
                    lazy_configs = cache.setdefault("lazy_configs", {})

                    # Propagate caches from node_copy to norm_node to preserve nested targets
                    _copy_lazy_state(node_copy, norm_node)

                    # Deduplicate dependencies while preserving order
                    unique_deps = list(dict.fromkeys(my_deps))
                    lazy_configs[str(local_key)] = (norm_node, unique_deps, kwargs)

                    if target_deps is not None:
                        target_deps.extend(unique_deps)

                    return f"${{_lazy_inst:{local_key}}}"

                for key in list(node.keys()):
                    if OmegaConf.is_interpolation(node, key):
                        continue
                    child_path = f"{path}.{key}" if path else str(key)
                    node[key] = _walk_and_ghost(
                        node[key], child_path, node, key, target_deps
                    )
                return node
            elif isinstance(node, ListConfig):
                for i in range(len(node)):
                    if OmegaConf.is_interpolation(node, i):
                        continue
                    child_path = f"{path}[{i}]"
                    node[i] = _walk_and_ghost(node[i], child_path, node, i, target_deps)
                return node
            return node

        if isinstance(config_copy, DictConfig) and "_target_" in config_copy:
            # Eagerly instantiate the root node if it's a target
            for k in list(config_copy.keys()):
                if k != "_target_" and not OmegaConf.is_interpolation(config_copy, k):
                    config_copy[k] = _walk_and_ghost(
                        config_copy[k], str(k), config_copy, k, None
                    )
            return hydra.utils.instantiate(config_copy, **kwargs)
        else:
            _walk_and_ghost(config_copy, "", None, "", None)
            return config_copy

    elif isinstance(config, dict):
        if "_target_" in config:
            return hydra.utils.instantiate(config, **kwargs)
        return {
            k: non_resolving_instantiate(v, lazy=lazy, **kwargs)
            for k, v in config.items()
        }
    elif isinstance(config, list):
        return [non_resolving_instantiate(v, lazy=lazy, **kwargs) for v in config]
    return config
