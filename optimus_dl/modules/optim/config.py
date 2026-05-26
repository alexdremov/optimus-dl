"""
General optimizer config.

This module provides configuration dataclasses for AMP (Automatic Mixed Precision)
and optimization settings, including FP8 (Floating Point 8-bit) support.
"""

from dataclasses import (
    dataclass,
    field,
)

from optimus_dl.core.registry import RegistryConfig


@dataclass
class Fp8Config:
    """Configuration for FP8 (Floating Point 8-bit) training.

    FP8 training uses 8-bit floating point formats (E4M3, E5M2) to reduce
    memory usage and increase training throughput. This configuration
    controls how FP8 is applied during training.

    Based on Megatron-LM's SOTA implementation (https://github.com/NVIDIA/Megatron-LM).

    Attributes:
        enabled: If True, use FP8 for training. If False, fall back to standard AMP.
        format: The FP8 format to use. Options:
            - "hybrid": Use E4M3 for forward pass, E5M2 for backward pass (recommended)
            - "e4m3": Use E4M3 for all tensors
            - "e5m2": Use E5M2 for all tensors (pure E5M2 not supported by TE DelayedScaling)
        recipe: FP8 quantization recipe. Options:
            - "delayed": Delayed scaling (default, recommended for training)
            - "tensorwise": Per-tensor current scaling (TE >= 2.2.0)
            - "blockwise": Block-wise scaling (TE >= 2.3.0)
            - "mxfp8": MXFP8 block scaling (Blackwell only, TE >= 2.1.0)
        param: If True, keep parameters in FP8 precision to save memory.
            Parameters are cast to FP8 after initialization and during all-gather.
        margin: Margin for amax computation. Larger values may help with numerical stability.
        amax_history_len: Length of amax history to maintain for delayed scaling.
        amax_compute_algo: Algorithm for computing amax. Options: "max" or "most_recent".
        reduce_amax: Whether to reduce amax across distributed group.
        wgrad: If False, override FP8 config and compute weight gradients in higher precision.
        fp8_dpa: Whether to enable FP8 dot product attention.
        fp8_mha: Whether to enable FP8 multi-head attention.
        first_last_layers_bf16: If True, retain first and last N layers in BF16.
            This can improve numerical stability.
        num_layers_at_start_in_bf16: Number of layers at start to keep in BF16.
        num_layers_at_end_in_bf16: Number of layers at end to keep in BF16.
        tp_only_amax_red: If True, reduce FP8 AMAX only in TP or TP-CP domain.
    """

    enabled: bool = False
    format: str = "hybrid"  # hybrid, e4m3 (e5m2 not supported by TE DelayedScaling)
    recipe: str = "delayed"  # delayed, tensorwise, blockwise, mxfp8
    param: bool = False
    margin: int = 0
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"  # max, most_recent
    reduce_amax: bool = True
    wgrad: bool = True
    fp8_dpa: bool = False
    fp8_mha: bool = False
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 1
    num_layers_at_end_in_bf16: int = 1
    tp_only_amax_red: bool = False


@dataclass
class AmpConfig:
    """Configuration for Automatic Mixed Precision (AMP) training.

    AMP enables training with reduced precision (FP16, BF16, FP8) to improve
    performance and reduce memory usage while maintaining model accuracy.

    Attributes:
        enabled: If True, use mixed precision training.
        dtype: The primary dtype for mixed precision. Options:
            - "torch.float16" / "fp16" / "float16"
            - "torch.bfloat16" / "bf16" / "bfloat16"
            - "torch.float8_e4m3fn" / "fp8_e4m3" / "e4m3" (requires FP8 config)
            - "torch.float8_e5m2" / "fp8_e5m2" / "e5m2" (requires FP8 config)
        enable_scaler: If True, use gradient scaler (recommended for FP16).
            Automatically disabled for BF16 and FP8.
        init_scale: Initial scale for gradient scaler.
        growth_factor: Factor by which to increase scale when no overflow occurs.
        backoff_factor: Factor by which to decrease scale when overflow occurs.
        growth_interval: Number of iterations between scale updates.
        fp8: FP8-specific configuration. Only used when dtype is FP8.
    """

    enabled: bool = False
    dtype: str = "torch.bfloat16"

    enable_scaler: bool = '${eval: \'"${.dtype}" == "torch.float16"\'}'
    init_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    # FP8-specific configuration
    fp8: Fp8Config = field(default_factory=Fp8Config)


@dataclass
class OptimizationConfig:
    """Configuration for training optimization.

    This includes optimizer settings, gradient accumulation, clipping,
    and mixed precision (AMP) configuration.

    Attributes:
        optimizer: Configuration for the optimizer (AdamW, SGD, etc.).
        iterations: Total number of training iterations/steps.
        acc_steps: Number of steps to accumulate gradients before updating weights.
        clip_grad_norm: Maximum norm for gradient clipping. If None, no clipping.
        amp: AMP configuration for mixed precision training.
    """

    optimizer: RegistryConfig

    iterations: int = field(default=1000, metadata={"description": "Total train steps"})
    acc_steps: int = field(
        default=1, metadata={"description": "Steps to accumulate gradient"}
    )
    clip_grad_norm: float | None = field(
        default=None, metadata={"description": "Clip gradient norm"}
    )
    amp: AmpConfig = field(default_factory=AmpConfig)
