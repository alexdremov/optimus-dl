"""Distributed model transforms for training."""

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP

from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.distributed.mesh import MeshCollective
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import register_model_transform
from optimus_dl.modules.model_transforms.base import BaseModelTransform
from optimus_dl.modules.model_transforms.config import ModelTransformConfig

logger = logging.getLogger(__name__)


class BaseDistributedTransform(BaseModelTransform):
    """Base class for distributed model transforms.

    Provides common access to the collective and device information.
    """

    def __init__(
        self,
        cfg: ModelTransformConfig,
        collective: Collective,
        device: torch.device,
        **kwargs: Any,
    ):
        super().__init__(cfg, **kwargs)
        self.collective = collective
        self.device = device


@dataclass
class DDPTransformConfig(ModelTransformConfig):
    """Configuration for Distributed Data Parallel (DDP).

    Attributes:
        find_unused_parameters: Whether to traverse the graph to find unused
            parameters during backward.
        gradient_as_bucket_view: If True, uses views for gradient buckets to
            save memory.
        static_graph: Whether the computation graph is static across iterations.
    """

    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False


class DDPWrappedModel(DDP, BaseModel):
    """A wrapper for DDP that implements the BaseModel interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_parameter_groups(self):
        """Delegate parameter grouping to the inner module."""
        return self.module.make_parameter_groups()

    def fully_shard(self, **fsdp_kwargs):
        """Delegate sharding to the inner module."""
        return self.module.fully_shard(**fsdp_kwargs)

    def accumulation_context(self, is_last_microbatch):
        """Context manager for gradient accumulation (disables synchronization)."""
        return nullcontext() if is_last_microbatch else self.no_sync()


@register_model_transform("ddp", DDPTransformConfig)
class DDPTransform(BaseDistributedTransform):
    """Transform that wraps a model with Distributed Data Parallel.

    DDP replicates the model on each device and synchronizes gradients during
    the backward pass.

    Args:
        cfg: DDP configuration.
        collective: Distributed collective.
        device: Target compute device.
    """

    def __init__(
        self,
        cfg: DDPTransformConfig,
        collective: Collective,
        device: torch.device,
        **kwargs: Any,
    ):
        super().__init__(cfg, collective, device, **kwargs)

        self.collective = collective

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply DDP wrapping to the model."""
        if self.collective.world_size <= 1:
            logger.info("Single rank detected, skipping DDP wrapping")
            return model

        logger.info("Wrapping model with DDP")

        # Move model to device
        model = model.to(self.device)

        # Wrap with DDP
        ddp_model = DDPWrappedModel(
            model,
            process_group=self.collective.process_group,
            device_ids=(
                [self.collective.local_rank] if self.device.type == "cuda" else None
            ),
            find_unused_parameters=self.cfg.find_unused_parameters,
            gradient_as_bucket_view=self.cfg.gradient_as_bucket_view,
            static_graph=self.cfg.static_graph,
        )

        return ddp_model


@dataclass
class MixedPrecisionConfig:
    """Configuration for FSDP mixed precision policy.

    Attributes:
        param_dtype: Datatype for parameter storage (e.g., 'bfloat16').
        reduce_dtype: Datatype for gradient reduction (e.g., 'float32').
        output_dtype: Datatype for forward pass outputs.
        cast_forward_inputs: If True, automatically casts inputs to param_dtype.
    """

    # Parameter storage dtype (e.g., float16, bfloat16, float32)
    param_dtype: str | None = None
    # Gradient reduction dtype (e.g., float16, bfloat16, float32)
    reduce_dtype: str | None = None
    # Output dtype for forward pass (e.g., float16, bfloat16, float32)
    output_dtype: str | None = None
    # Whether to cast forward inputs to the specified dtype
    cast_forward_inputs: bool = True


@dataclass
class OffloadConfig:
    """Configuration for FSDP offloading policy.

    Attributes:
        cpu_offload: If True, offloads parameters to CPU memory.
        pin_memory: If True, pins CPU memory for faster transfers.
    """

    # Whether to enable CPU offloading
    cpu_offload: bool = False
    # Whether to pin memory for CPU offloaded parameters (only relevant if cpu_offload=True)
    pin_memory: bool = True


@dataclass
class FullyShardTransformConfig(ModelTransformConfig):
    """Configuration for FSDP2 (fully_shard) transform.

    Attributes:
        reshard_after_forward: Whether to discard parameters after forward pass.
        mixed_precision: Mixed precision policy configuration.
        offload: CPU offloading policy configuration.
        use_hybrid_sharding: If True, uses Hybrid Sharding (shard within node,
            replicate across nodes).
        sync_grad_accum: If True, always synchronizes gradients during accumulation.
    """

    # Whether to reshard parameters after forward pass
    reshard_after_forward: bool | int = False
    # Mixed precision configuration
    mixed_precision: MixedPrecisionConfig | None = None
    # Offloading configuration
    offload: OffloadConfig | None = None
    # Whether to use hybrid sharding (HSDP): shard within nodes, replicate across nodes
    use_hybrid_sharding: bool = True

    sync_grad_accum: bool = False


@register_model_transform("fully_shard", FullyShardTransformConfig)
class FullyShardTransform(BaseDistributedTransform):
    """Transform that wraps a model with FSDP2 (Fully Sharded Data Parallel).

    FSDP2 shards model parameters, gradients, and optimizer states across ranks,
    enabling the training of models much larger than the memory of a single GPU.

    Args:
        cfg: FSDP2 configuration.
        collective: Distributed collective (MeshCollective required).
        device: Target compute device.
    """

    def __init__(
        self,
        cfg: FullyShardTransformConfig,
        collective: Collective,
        device: torch.device,
        **kwargs: Any,
    ):
        super().__init__(cfg, collective, device, **kwargs)
        self.mesh = None
        if self.collective.world_size > 1:
            self.mesh = self._create_hybrid_mesh()

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply FSDP2 sharding to the model."""
        if self.collective.world_size <= 1:
            logger.info("Single rank detected, skipping FSDP2 wrapping")
            return model

        logger.info("Wrapping model with FSDP2 (fully_shard)")

        # Move model to device
        model = model.to(self.device)

        # Configure FSDP2 options
        fsdp_kwargs = {}

        # Add mesh if available
        if self.mesh is not None:
            fsdp_kwargs["mesh"] = self.mesh

        # Set reshard_after_forward
        fsdp_kwargs["reshard_after_forward"] = self.cfg.reshard_after_forward

        # Configure mixed precision policy
        if self.cfg.mixed_precision is not None:
            mp_config = self.cfg.mixed_precision

            # Convert string dtype names to torch dtypes
            param_dtype = (
                self._str_to_dtype(mp_config.param_dtype)
                if mp_config.param_dtype
                else None
            )
            reduce_dtype = (
                self._str_to_dtype(mp_config.reduce_dtype)
                if mp_config.reduce_dtype
                else None
            )
            output_dtype = (
                self._str_to_dtype(mp_config.output_dtype)
                if mp_config.output_dtype
                else None
            )

            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                output_dtype=output_dtype,
                cast_forward_inputs=mp_config.cast_forward_inputs,
            )
            fsdp_kwargs["mp_policy"] = mp_policy
            logger.info(
                f"Configured mixed precision: param={param_dtype}, reduce={reduce_dtype}, output={output_dtype}"
            )

        # Configure offloading policy
        if self.cfg.offload is not None and self.cfg.offload.cpu_offload:
            offload_policy = CPUOffloadPolicy(pin_memory=self.cfg.offload.pin_memory)
            fsdp_kwargs["offload_policy"] = offload_policy
            logger.info(
                f"Configured CPU offloading with pin_memory={self.cfg.offload.pin_memory}"
            )

        # Apply fully_shard to the model
        model.fully_shard(**fsdp_kwargs)
        fsdp_model = fully_shard(model, **fsdp_kwargs)

        @contextmanager
        def accumulation_context(is_last_microbatch):
            """Context manager for FSDP gradient accumulation."""
            if self.cfg.sync_grad_accum:
                fsdp_model.set_requires_gradient_sync(True)
                yield
                return

            if is_last_microbatch:
                fsdp_model.set_requires_gradient_sync(True)
            else:
                fsdp_model.set_requires_gradient_sync(False)
            yield
            fsdp_model.set_requires_gradient_sync(True)

        # The return type will be the FSDP-wrapped model
        fsdp_model.accumulation_context = accumulation_context
        return fsdp_model

    def _str_to_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype name to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
            "double": torch.float64,
            "half": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        if dtype_str not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype_str}. Supported dtypes: {list(dtype_map.keys())}"
            )

        return dtype_map[dtype_str]

    def _create_hybrid_mesh(self):
        """Create a hybrid sharding mesh (HSDP) from the collective's DP mesh."""
        if not isinstance(self.collective, MeshCollective):
            raise ValueError("Hybrid sharding requires MeshCollective")

        mesh = self.collective.dp_mesh
        if not self.cfg.use_hybrid_sharding:
            mesh = mesh._flatten()
        return mesh
