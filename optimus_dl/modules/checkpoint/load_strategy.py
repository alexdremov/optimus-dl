from dataclasses import dataclass, field


@dataclass
class LoadStrategy:
    """Strategy for loading checkpoints."""

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
