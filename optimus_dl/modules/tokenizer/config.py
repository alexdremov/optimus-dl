from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfig


@dataclass
class BaseTokenizerConfig(RegistryConfig):
    add_bos: bool = True
    add_eos: bool = True
