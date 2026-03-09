import random
from dataclasses import dataclass
from typing import Any

from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)


@dataclass
class SkipInterleavedTransformConfig(RegistryConfigStrict):
    """Configuration for skip_interleaved.

    Attributes:
        skip_count: Number of items to skip before producing a new one. First item is always produced.
    """

    skip_count: int = 1


class SkipInterleavedTransformNode(BaseNode):
    def __init__(
        self, node: BaseNode, cfg: SkipInterleavedTransformConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node

        self.counter = 0

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        if initial_state:
            self.counter = initial_state.get("counter", 0)
            self.node.reset(initial_state.get("source_state"))
        else:
            self.counter = 0
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        return {
            "source_state": self.node.state_dict(),
            "counter": self.counter,
        }

    def next(self):
        if self.counter > 0:
            for _ in range(self.cfg.skip_count):
                next(self.node)
                self.counter += 1

        # Fetch the actual item we want to produce
        value = next(self.node)
        self.counter += 1

        return value


@register_transform("skip_interleaved", SkipInterleavedTransformConfig)
class SkipInterleavedTransform(BaseTransform):
    """Transform that deterministically skips a fixed number of data items.

    This is useful for downsampling a dataset or creating interleaved subsets.
    It guarantees that the first item is always produced, followed by skipping
    exactly `skip_count` items before producing the next one.

    Args:
        cfg: Configuration containing the `skip_count` parameter.
    """

    def __init__(self, cfg: SkipInterleavedTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        """Wrap the source node with a SkipInterleavedTransformNode."""
        return SkipInterleavedTransformNode(source, self.cfg)


@dataclass
class SkipRandomTransformConfig(RegistryConfigStrict):
    """Configuration for skip_random.

    Attributes:
        probability: Probability (0.0 to 1.0) that any given item is skipped.
    """

    probability: float = 0.5


class SkipRandomTransformNode(BaseNode):
    def __init__(
        self, node: BaseNode, cfg: SkipRandomTransformConfig, seed: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.seed = seed

        # Use an isolated RNG instance so we don't pollute/rely on global state
        self.rng = random.Random(seed)

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        if initial_state:
            if "rng_state" in initial_state:
                self.rng.setstate(initial_state["rng_state"])
            self.node.reset(initial_state.get("source_state"))
        else:
            # Hard reset back to the initial seed
            self.rng = random.Random(self.seed)
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        return {
            "source_state": self.node.state_dict(),
            "rng_state": self.rng.getstate(),
        }

    def next(self):
        while True:
            # Fetch the next item. (Will naturally raise StopIteration if the source is empty)
            value = next(self.node)

            # Roll the dice: if less than probability, loop again and discard this value
            if self.rng.random() < self.cfg.probability:
                continue

            # Otherwise, keep and yield it
            return value


@register_transform("skip_random", SkipRandomTransformConfig)
class SkipRandomTransform(BaseTransform):
    """Transform that randomly skips data items.

    This adds stochastic sub-sampling to the data pipeline. Note that since
    skipping is probabilistic, the final dataset length will vary slightly
    unless exactly replicated with the same random seed.

    Args:
        cfg: Configuration with the skip probability and seed.
    """

    def __init__(self, cfg: SkipRandomTransformConfig, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.seed = seed

    def build(self, source: BaseNode) -> BaseNode:
        """Wrap the source node with a SkipRandomTransformNode."""
        return SkipRandomTransformNode(source, self.cfg, seed=self.seed)
