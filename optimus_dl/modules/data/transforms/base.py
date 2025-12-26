"""Base transform classes for data pipeline.

This module defines the base classes for data transforms, which are components
that process data as it flows through the pipeline. Transforms can be chained
together to create complex data processing pipelines.
"""

from dataclasses import dataclass

import torchdata.nodes


class BaseTransform:
    """Base class for all data transforms.

    All data transforms in Optimus-DL should inherit from this class. Transforms
    take a data source (BaseNode) and return a new BaseNode that applies the
    transformation. Transforms can be chained together using CompositeTransform.

    Subclasses should implement:

    - `build()`: Apply the transform to a data source and return a new node

    Example:
        ```python
        @register_transform("tokenize", TokenizeConfig)
        class TokenizeTransform(BaseTransform):
            def __init__(self, cfg: TokenizeConfig, **kwargs):
                super().__init__(**kwargs)
                self.tokenizer = build_tokenizer(cfg.tokenizer_config)

            def build(self, source: BaseNode) -> BaseNode:
                def tokenize_fn(item):
                    return {"input_ids": self.tokenizer.encode(item["text"])}
                return source.map(tokenize_fn)

        ```"""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the transform.

        Args:
            *args: Positional arguments (typically unused, for compatibility).
            **kwargs: Keyword arguments passed from the data builder.
        """
        pass

    def build(self, source: torchdata.nodes.BaseNode) -> torchdata.nodes.BaseNode:
        """Apply the transform to a data source.

        This method takes a data source node and returns a new node that applies
        the transformation. The transformation is applied lazily as data flows
        through the pipeline.

        Args:
            source: The data source node to transform.

        Returns:
            A new BaseNode that applies the transformation.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Example:
            ```python
            transform = TokenizeTransform(cfg)
            transformed_source = transform.build(raw_source)
            # transformed_source now yields tokenized data

            ```"""
        raise NotImplementedError


@dataclass
class MapperConfig:
    """Configuration for map operations in data transforms.

    This configuration is used by transforms that apply map operations to data.
    It controls parallelism, ordering, and batching behavior.

    Attributes:
        num_workers: Number of worker processes/threads for parallel processing.
        in_order: If True, preserve the order of items. If False, allow out-of-order
            processing for better performance.
        method: Parallelization method: "thread" (threading) or "process" (multiprocessing).
        snapshot_frequency: How often to snapshot the iterator state for checkpointing.
        prebatch: Number of items to batch together before processing (for efficiency).
    """

    num_workers: int = 4
    in_order: bool = True
    method: str = "thread"
    snapshot_frequency: int = 32
    prebatch: int = 32
