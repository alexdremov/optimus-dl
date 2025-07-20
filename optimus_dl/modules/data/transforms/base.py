from dataclasses import dataclass

import torchdata.nodes


class BaseTransform:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def build(self, source: torchdata.nodes.BaseNode) -> torchdata.nodes.BaseNode:
        raise NotImplementedError


@dataclass
class MapperConfig:
    num_workers: int = 4
    in_order: bool = True
    method: str = "thread"
    snapshot_frequency: int = 32
    prebatch: int = 32
