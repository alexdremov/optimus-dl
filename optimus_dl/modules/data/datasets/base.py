"""Base dataset class for data sources.

This module defines the BaseDataset class that all data sources must inherit from.
It provides integration with torchdata's pipeline system and checkpointing support.
"""

import torchdata.nodes


class BaseDataset(torchdata.nodes.BaseNode):
    """Base class for all dataset implementations.

    All data sources in Optimus-DL should inherit from this class. It provides:
    - Integration with torchdata's pipeline system
    - Checkpointing support for resuming data iteration
    - Configuration storage

    Subclasses should implement:
    - The data iteration logic (inherited from torchdata.nodes.BaseNode)
    - Optionally override `load_state_dict()` for custom checkpointing

    Example:
        >>> @register_dataset("my_dataset", MyDatasetConfig)
        >>> class MyDataset(BaseDataset):
        ...     def __init__(self, cfg: MyDatasetConfig, **kwargs):
        ...         super().__init__(cfg, **kwargs)
        ...         self.data = load_data(cfg.data_path)
        ...
        ...     def __iter__(self):
        ...         for item in self.data:
        ...             yield item
    """

    def __init__(self, cfg, **kwargs):
        """Initialize the base dataset.

        Args:
            cfg: Configuration object for this dataset.
            **kwargs: Additional keyword arguments passed from the data builder.
        """
        super().__init__()
        self.cfg = cfg

    def load_state_dict(self, state_dict: dict) -> None:
        """Load dataset state from checkpoint.

        This method restores the dataset's iteration state, allowing training
        to resume from the same position in the dataset. The default implementation
        uses torchdata's `reset()` method.

        Args:
            state_dict: Dictionary containing the dataset's saved state.
                Typically includes iteration position, random state, etc.

        Note:
            Subclasses can override this to handle custom state restoration.
            The state_dict is typically saved by the checkpoint manager.
        """
        self.reset(state_dict)
