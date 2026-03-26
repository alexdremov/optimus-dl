from dataclasses import (
    dataclass,
    field,
)
from typing import Any

from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfig


@dataclass
class DataPipelineConfig:
    source: RegistryConfig = field(
        default=MISSING,
        metadata={"description": "Config for the dataset source"},
    )
    transform: RegistryConfig | None = field(
        default=None,
        metadata={"description": "Config for the dataset transforms"},
    )
    profile: bool = field(
        default=False,
        metadata={"description": "Whether to profile this data pipeline"},
    )
    report_freq: int = field(
        default=100,
        metadata={
            "description": "Frequency of profiling report in number of iterations"
        },
    )


@dataclass
class EvalDataPipelineConfig(DataPipelineConfig):
    eval_freq: int | None = field(
        default=None,
        metadata={
            "description": "Frequency of evaluation in number of training steps specifically for this dataset. If None, use the global eval_freq."
        },
    )

    eval_iterations: int | None = field(
        default=None,
        metadata={
            "description": "Max number of iterations of validation data for this dataset. If None, use the global eval_iterations."
        },
    )

    eval_guaranteed_same_batches: bool | None = field(
        default=None,
        metadata={
            "description": "Whether it is guaranteed that each DP rank sees the same batches count. If None, use the global eval_guaranteed_same_batches."
        },
    )


@dataclass
class DataConfig:
    train_datasets: DataPipelineConfig = field(
        default=MISSING,
        metadata={
            "description": "Config for the training batches: dataset and transforms"
        },
    )
    eval_datasets: dict[str, EvalDataPipelineConfig] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Config for the evaluation batches: dataset and transforms. "
                "The key is the name of the dataset, which will be used to identify the dataset in the metrics. "
                "The value is the config for the dataset and transforms."
            )
        },
    )

    scratch: Any = field(
        default=None,
        metadata={
            "description": "Any data whatsoever to be used in dataset configs with config interpolations like ${data.scratch.my_config} to reduce duplication"
        },
    )
