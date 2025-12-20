from dataclasses import dataclass


@dataclass
class LoadStrategy:
    load_model: bool = True
    load_optimizer: bool = True
    load_scheduler: bool = True
    load_data_sources: bool = True
    load_dataloaders: bool = True
    load_metrics: bool = True
    load_iteration: bool = True
    extra_ignore_keys: list[str] | None = None
