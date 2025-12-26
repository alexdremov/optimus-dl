"""Checkpoint loading strategy configuration.

This module defines the LoadStrategy class that controls which components of a
checkpoint are loaded. This is useful for fine-tuning scenarios where you might
want to load model weights but not optimizer state, or for resuming training with
different configurations.
"""

from dataclasses import dataclass, field


@dataclass
class LoadStrategy:
    """Configuration for selective checkpoint loading.

    This class controls which components are loaded from a checkpoint. It's
    particularly useful for:
    - Fine-tuning: Load model weights but not optimizer state
    - Resuming with different configs: Load model but reset optimizer
    - Evaluation: Load only model weights
    - Debugging: Load specific components to isolate issues

    All fields default to True, meaning by default everything is loaded.

    Attributes:
        load_model: If True, load model weights. Set to False to skip model loading.
        load_optimizer: If True, load optimizer state. Set to False to start with
            fresh optimizer state (useful for fine-tuning).
        load_scheduler: If True, load learning rate scheduler state. Set to False
            to restart scheduler from beginning.
        load_data_sources: If True, restore data source state (e.g., dataset reader
            position). Set to False to restart from beginning of dataset.
        load_dataloaders: If True, restore full dataloader state including shuffling
            state. Set to False to restart dataloaders.
        load_metrics: If True, restore accumulated metrics. Set to False to start
            with fresh metrics.
        load_iteration: If True, resume iteration count from checkpoint. Set to False
            to restart iteration counting (useful when fine-tuning).
        extra_ignore_keys: Optional list of state dict keys to ignore during loading.

    Example:
        ```python
        # Fine-tuning: load model only
        strategy = LoadStrategy(
            load_model=True,
            load_optimizer=False,
            load_scheduler=False,
            load_iteration=False,
        )
        
        # Resume training: load everything
        strategy = LoadStrategy()  # All defaults to True
        
        # Evaluation: model only
        strategy = LoadStrategy(
            load_model=True,
            load_optimizer=False,
            load_scheduler=False,
            load_data_sources=False,
            load_dataloaders=False,
            load_metrics=False,
            load_iteration=False,
        )
    
        ```"""

    load_model: bool = field(
        default=True, metadata={"help": "Whether to load model weights."}
    )
    load_optimizer: bool = field(
        default=True, metadata={"help": "Whether to load optimizer state."}
    )
    load_scheduler: bool = field(
        default=True,
        metadata={"help": "Whether to load learning rate scheduler state."},
    )
    load_data_sources: bool = field(
        default=True,
        metadata={"help": "Whether to load data source state (e.g. dataset position)."},
    )
    load_dataloaders: bool = field(
        default=True, metadata={"help": "Whether to load full dataloader state."}
    )
    load_metrics: bool = field(
        default=True, metadata={"help": "Whether to load accumulated metrics."}
    )
    load_iteration: bool = field(
        default=True, metadata={"help": "Whether to resume the iteration count."}
    )
    extra_ignore_keys: list[str] | None = field(
        default=None,
        metadata={
            "help": "List of specific keys to ignore in the checkpoint state dict."
        },
    )
