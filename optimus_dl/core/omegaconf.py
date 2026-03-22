"""OmegaConf custom resolvers for configuration.

This module registers custom resolvers for OmegaConf that enable advanced
configuration features like Python expression evaluation and environment
variable access.

It also provides a custom non-resolving instantiation system that allows
Hydra-style object creation while preserving OmegaConf reactivity and
interpolation integrity.
"""

import copy
import hashlib
import importlib
import logging
import os
from typing import Any

import hydra
from omegaconf import (
    DictConfig,
    ListConfig,
    OmegaConf,
)

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


# Use a global store for ghost nodes, keyed by (root_id, ghost_key)
_GLOBAL_GHOST_CONFIGS: dict[tuple[int, str], DictConfig] = {}
_GLOBAL_GHOST_DEPS: dict[tuple[int, str], list[str]] = {}
_GLOBAL_LAZY_INST_CACHE: dict[tuple[int, str], tuple[str, Any]] = {}
_GLOBAL_RECURSION_TRACKER: set[tuple[int, str]] = set()


def _lazy_inst_resolver(ghost_key: str, _root_: Any) -> Any:
    """Internal resolver for reactive lazy instantiation."""
    root_id = id(_root_)
    cache_key = (root_id, ghost_key)

    if cache_key in _GLOBAL_RECURSION_TRACKER:
        raise ValueError(f"Circular dependency detected for lazy node: {ghost_key}")

    if cache_key not in _GLOBAL_GHOST_CONFIGS:
        raise ValueError(f"Could not find lazy config node for {ghost_key}")

    node = _GLOBAL_GHOST_CONFIGS[cache_key]
    outside_deps = _GLOBAL_GHOST_DEPS.get(cache_key, [])

    # Fingerprinting: Check external dependencies
    try:
        if outside_deps:
            # Create a temporary container to resolve dependencies against the root
            temp_deps = OmegaConf.create(outside_deps)
            temp_deps._set_parent(_root_)
            current_fp = str(OmegaConf.to_container(temp_deps, resolve=True))
        else:
            current_fp = ""
    except Exception as e:
        if "recursion" in str(e).lower() or isinstance(e, RecursionError):
            raise ValueError(
                f"Circular dependency detected during resolution: {ghost_key}"
            ) from e
        raise

    if cache_key in _GLOBAL_LAZY_INST_CACHE:
        last_fp, instance = _GLOBAL_LAZY_INST_CACHE[cache_key]
        if last_fp == current_fp:
            return instance

    # Re-instantiate
    node._set_parent(_root_)
    _GLOBAL_RECURSION_TRACKER.add(cache_key)
    try:
        # We NO LONGER clean the config. If the user provided extra keys,
        # it's their responsibility to ensure the target accepts them.
        # This preserves interpolations that might depend on those keys.
        instance = hydra.utils.instantiate(node)
    finally:
        _GLOBAL_RECURSION_TRACKER.remove(cache_key)

    if isinstance(instance, (dict, list)):
        instance = OmegaConf.create(instance)

    _GLOBAL_LAZY_INST_CACHE[cache_key] = (current_fp, instance)
    return instance


# Register the internal lazy instantiation resolver
if not OmegaConf.has_resolver("_lazy_inst"):
    OmegaConf.register_new_resolver("_lazy_inst", _lazy_inst_resolver, use_cache=False)


def _split_path(path: str) -> list[str]:
    """Split an OmegaConf path into components, handling both . and [index]."""
    parts = []
    curr = ""
    for char in path:
        if char == ".":
            if curr:
                parts.append(curr)
            curr = ""
        elif char == "[":
            if curr:
                parts.append(curr)
            curr = "["
        elif char == "]":
            curr += "]"
            parts.append(curr)
            curr = ""
        else:
            curr += char
    if curr:
        parts.append(curr)
    return parts


def _join_path(parts: list[str]) -> str:
    """Join path components back into an OmegaConf dot-separated string."""
    res = ""
    for i, p in enumerate(parts):
        if p.startswith("["):
            res += p
        else:
            if i > 0:
                res += "."
            res += p
    return res


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

    # OmegaConf relative paths: . is current container, .. is parent container
    # pop n-1 parts for n dots.
    pop_count = dots - 1
    if pop_count > 0:
        parts = parts[:-pop_count]

    if temp_key:
        parts.append(temp_key)

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

    # up_count is number of steps to go up from parent container to common prefix
    up_count = (len(f_parts) - 1) - common
    dots = up_count + 1
    rem = t_parts[common:]
    return "." * dots + _join_path(rem)


def _extract_interpolations(s: str) -> list[str]:
    """Robustly extract all top-level interpolation blocks from a string, handling nesting."""
    inters = []
    stack = []
    for i in range(len(s)):
        if s[i : i + 2] == "${":
            stack.append(i)
        elif s[i] == "}" and stack:
            start = stack.pop()
            if not stack:
                inters.append(s[start : i + 1])
    return inters


def _normalize_and_collect_deps(
    obj: Any, current_path: str, ghost_root_path: str, outside_deps: list[str]
) -> Any:
    """Recursively normalize interpolations and collect outside dependencies."""
    if isinstance(obj, dict):
        return {
            k: _normalize_and_collect_deps(
                v,
                f"{current_path}.{k}" if current_path else k,
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
        # parent_path is the container of this value
        parent_path = _join_path(parts[:-1])

        def process_inter_block(inter_block: str) -> str:
            # content from ${content}
            content = inter_block[2:-1]

            # Recursively handle nested interpolations in resolver arguments
            if ":" in content:
                resolver_parts = content.split(":", 1)
                name = resolver_parts[0]
                args = resolver_parts[1]

                # Robust extraction of nested blocks within args
                nested_blocks = _extract_interpolations(args)
                for block in nested_blocks:
                    args = args.replace(block, process_inter_block(block))
                return f"${{{name}:{args}}}"

            # Resolve absolute key from root
            abs_key = _to_absolute_key(content, parent_path)

            # Normalization: keep internal dependencies relative, external absolute
            if (
                abs_key == ghost_root_path
                or abs_key.startswith(ghost_root_path + ".")
                or abs_key.startswith(ghost_root_path + "[")
            ):
                return f"${{{_get_relative_path(current_path, abs_key)}}}"
            else:
                outside_deps.append(f"${{{abs_key}}}")
                return f"${{{abs_key}}}"

        # Replace all top-level interpolations in the string with normalized versions
        res = obj
        for block in _extract_interpolations(obj):
            res = res.replace(block, process_inter_block(block))
        return res
    return obj


def non_resolving_instantiate(config: Any, lazy: bool = True, **kwargs: Any) -> Any:
    """Custom instantiate function that preserves OmegaConf interpolations.

    It selectively instantiates nodes with a '_target_' using hydra.utils.instantiate,
    without eagerly resolving the entire configuration tree.

    Unlike standard Hydra instantiation, this preserves reactivity:
    - If a target node depends on '${params.lr}', and 'params.lr' is updated
      after instantiation, the object will be automatically re-instantiated
      on the next access (if lazy=True).
    - Interpolations pointing TO an instantiated node (e.g., 'ref: ${model}')
      remain live and will reflect the current state of the model.

    EXAMPLES:
        # Reactive behavior
        cfg = OmegaConf.create({'val': 10, 'obj': {'_target_': 'math.ceil', '_args_': ['${val}']}})
        inst = non_resolving_instantiate(cfg)
        print(inst.obj) # 10
        inst.val = 20.1
        print(inst.obj) # 21 (Re-instantiated!)

        # Interpolation preservation
        cfg = OmegaConf.create({'a': {'_target_': 'dict', 'x': 1}, 'b': '${a.x}'})
        inst = non_resolving_instantiate(cfg)
        inst.a.x = 100
        print(inst.b) # 100

    COUNTER-EXAMPLES:
        # standard instantiate (EAGER)
        cfg = ...
        inst = hydra.utils.instantiate(cfg)
        inst.val = 20.1
        print(inst.obj) # 10 (STALE - it was replaced by a fixed int)

    Args:
        config: The configuration node to instantiate.
        lazy: If True, defers instantiation and supports reactive re-instantiation.
              Defaults to True.
        **kwargs: Additional keyword arguments to pass to hydra.utils.instantiate.

    Returns:
        The instantiated object or a new configuration object with targets instantiated.
    """
    if isinstance(config, (DictConfig, ListConfig)):
        config_copy = copy.deepcopy(config)
        OmegaConf.set_struct(config_copy, False)
        OmegaConf.set_readonly(config_copy, False)
        config_copy._set_flag("allow_objects", True)

        root_id = id(config_copy)

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

        def _walk_and_ghost(node: Any, path: str) -> Any:
            if isinstance(node, DictConfig):
                if "_target_" in node:
                    # RATIONALE: To move a node to internal storage while preserving
                    # its relative interpolations, we must normalize them.
                    outside_deps: list[str] = []
                    raw_data = OmegaConf.to_container(node, resolve=False)
                    norm_data = _normalize_and_collect_deps(
                        raw_data, path, path, outside_deps
                    )
                    norm_node = OmegaConf.create(norm_data)

                    # Move to internal ghost storage.
                    ghost_key = (
                        path.replace(".", "_").replace("[", "_").replace("]", "_")
                    )
                    if not ghost_key:
                        ghost_key = "root"

                    _GLOBAL_GHOST_CONFIGS[(root_id, ghost_key)] = norm_node
                    _GLOBAL_GHOST_DEPS[(root_id, ghost_key)] = outside_deps

                    # Replace node with our reactive resolver.
                    return f"${{_lazy_inst:{ghost_key}}}"

                for key in list(node.keys()):
                    if key == "_lazy_configs":
                        continue
                    if OmegaConf.is_interpolation(node, key):
                        continue
                    child_path = f"{path}.{key}" if path else key
                    node[key] = _walk_and_ghost(node[key], child_path)
                return node
            elif isinstance(node, ListConfig):
                for i in range(len(node)):
                    if OmegaConf.is_interpolation(node, i):
                        continue
                    child_path = f"{path}[{i}]"
                    node[i] = _walk_and_ghost(node[i], child_path)
                return node
            return node

        return _walk_and_ghost(config_copy, "")

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
