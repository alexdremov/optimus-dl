"""OmegaConf custom resolvers for configuration.

This module registers custom resolvers for OmegaConf that enable advanced
configuration features like Python expression evaluation and environment
variable access.
"""

import hashlib
import logging
import os

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Register custom resolvers for OmegaConf
# These can be used in YAML configs with ${resolver:args} syntax

OmegaConf.register_new_resolver("eval", eval)
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
