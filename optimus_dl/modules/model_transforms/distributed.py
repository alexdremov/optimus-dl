"""Distributed model transforms for training."""

import logging
from contextlib import (
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from optimus_dl.core.dtype import str_to_dtype
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

        assert isinstance(
            self.collective, MeshCollective
        ), "DDPTransform requires a MeshCollective for distributed mode"
        assert (
            self.collective.dp_mesh.shape[0] == self.collective.dp_world_size
        ), "DP mesh size must match world size for DDPTransform (no sharding is possible with DDP)"

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

    Supports standard FP16/BF16 and FP8 (E4M3, E5M2) formats.
    When using FP8 with Transformer Engine, the FP8 recipe handles quantization
    and these dtypes can be set to None to let FSDP2 use defaults.

    Attributes:
        param_dtype: Datatype for parameter storage (e.g., 'bfloat16', 'float8_e4m3fn').
        reduce_dtype: Datatype for gradient reduction (e.g., 'float32', 'float8_e5m2').
        output_dtype: Datatype for forward pass outputs.
        cast_forward_inputs: If True, automatically casts inputs to param_dtype.
        fp8_enabled: If True, indicates that FP8 is being used (for logging/debugging).
        fp8_format: FP8 format string (e4m3, e5m2, hybrid) if using FP8.
    """

    # Parameter storage dtype (e.g., float16, bfloat16, float32, float8_e4m3fn)
    param_dtype: str | None = None
    # Gradient reduction dtype (e.g., float16, bfloat16, float32, float8_e5m2)
    reduce_dtype: str | None = None
    # Output dtype for forward pass (e.g., float16, bfloat16, float32)
    output_dtype: str | None = None
    # Whether to cast forward inputs to the specified dtype
    cast_forward_inputs: bool = True
    # FP8-specific: Whether FP8 is enabled
    fp8_enabled: bool = False
    # FP8-specific: FP8 format (e4m3, e5m2, hybrid)
    fp8_format: str | None = None


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

            # Check if FP8 is enabled
            if mp_config.fp8_enabled:
                logger.info(
                    f"FP8 mode enabled with format={mp_config.fp8_format}. "
                    f"Mixed precision dtypes will be handled by Transformer Engine. "
                    f"FSDP2 will use default dtypes."
                )

            # Convert string dtype names to torch dtypes
            # Note: For FP8, these may be None as TE handles quantization
            param_dtype = (
                str_to_dtype(mp_config.param_dtype) if mp_config.param_dtype else None
            )
            reduce_dtype = (
                str_to_dtype(mp_config.reduce_dtype) if mp_config.reduce_dtype else None
            )
            output_dtype = (
                str_to_dtype(mp_config.output_dtype) if mp_config.output_dtype else None
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

    def _create_hybrid_mesh(self):
        """Create a hybrid sharding mesh (HSDP) from the collective's DP mesh."""
        if not isinstance(self.collective, MeshCollective):
            raise ValueError("Hybrid sharding requires MeshCollective")

        mesh = self.collective.dp_mesh
        if not self.cfg.use_hybrid_sharding:
            mesh = mesh._flatten()
        return mesh
