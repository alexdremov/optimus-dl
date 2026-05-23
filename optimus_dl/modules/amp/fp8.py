"""FP8 (Floating Point 8-bit) training support for Optimus-DL.

This module provides integration with NVIDIA's Transformer Engine for FP8 training,
enabling state-of-the-art low-precision training with E4M3 and E5M2 formats.

Key Concepts:
- E4M3: 4 exponent bits, 3 mantissa bits - ideal for forward pass (weights/activations)
- E5M2: 5 exponent bits, 2 mantissa bits - ideal for backward pass (gradients)
- Hybrid: E4M3 for forward, E5M2 for backward (recommended for most use cases)

Example:
    from optimus_dl.modules.amp import FP8Recipe, create_fp8_recipe, is_transformer_engine_available

    if is_transformer_engine_available():
        fp8_recipe = create_fp8_recipe("hybrid", amax_history_len=16)
    else:
        # Fall back to standard AMP
        fp8_recipe = None
"""

import importlib.util
import logging
from collections.abc import Iterator
from contextlib import (
    contextmanager,
)
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
)

import torch

logger = logging.getLogger(__name__)


class Fp8Format(Enum):
    """FP8 format enumeration."""

    HYBRID = "hybrid"  # E4M3 forward, E5M2 backward (recommended)
    E4M3 = "e4m3"  # E4M3 for all tensors
    E5M2 = "e5m2"  # E5M2 for all tensors


def is_transformer_engine_available() -> bool:
    """Check if NVIDIA Transformer Engine is available.

    Returns:
        True if Transformer Engine can be imported, False otherwise.
    """
    try:
        # Check if it's a real package, not a meta/stub package
        if importlib.util.find_spec("transformer_engine.common.recipe") is None:
            return False

        return True
    except (ImportError, RuntimeError, AttributeError):
        return False


def get_transformer_engine_version() -> str | None:
    """Get the Transformer Engine version if available.

    Returns:
        The version string, or None if Transformer Engine is not installed.
    """
    if not is_transformer_engine_available():
        return None
    try:
        import transformer_engine

        return transformer_engine.__version__
    except AttributeError:
        return None


def get_fp8_format_from_string(format_str: str) -> Fp8Format:
    """Convert a string representation to an Fp8Format enum.

    Args:
        format_str: String representation of the FP8 format.

    Returns:
        The corresponding Fp8Format enum value.

    Raises:
        ValueError: If the format string is not recognized.
    """
    format_str = format_str.lower()
    if format_str in ("hybrid", "hybrid_fp8", "fp8_hybrid"):
        return Fp8Format.HYBRID
    elif format_str in ("e4m3", "fp8_e4m3", "e4m3fn", "float8_e4m3fn"):
        return Fp8Format.E4M3
    elif format_str in ("e5m2", "fp8_e5m2", "e5m2fn", "float8_e5m2"):
        return Fp8Format.E5M2
    else:
        raise ValueError(
            f"Unknown FP8 format: {format_str}. "
            f"Supported formats: hybrid, e4m3, e5m2"
        )


@dataclass
class Fp8RecipeConfig:
    """Configuration for creating an FP8 recipe.

    This configuration is used to create a Transformer Engine FP8 recipe
    for use in training loops.

    Attributes:
        format: The FP8 format to use (hybrid, e4m3, e5m2).
        margin: Margin for amax computation.
        amax_history_len: Length of amax history.
        amax_compute_algo: Algorithm for computing amax ("max" or "most_recent").
        reduce_amax: Whether to reduce amax across distributed group.
        fp8_dpa: Whether to enable FP8 dot product attention.
        fp8_mha: Whether to enable FP8 multi-head attention.
    """

    format: str = "hybrid"
    margin: int = 0
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"
    reduce_amax: bool = True
    fp8_dpa: bool = False
    fp8_mha: bool = False


class FP8Recipe:
    """FP8 recipe wrapper for Transformer Engine.

    This class wraps Transformer Engine's FP8 recipe (DelayedScaling) and
    provides a clean interface for use in Optimus-DL training loops.

    The recipe handles:
    - Automatic quantization/dequantization of tensors
    - Scaling factor management (amax)
    - Format selection (E4M3 for forward, E5M2 for backward)
    - Numerical stability features

    Example:
        # Create a recipe
        recipe = FP8Recipe(format="hybrid", amax_history_len=16)

        # Use in training loop
        with recipe.autocast():
            output = model(input)
            loss = criterion(output, target)

        with recipe.backward():
            loss.backward()
    """

    def __init__(
        self,
        format: str | Fp8Format = Fp8Format.HYBRID,
        margin: int = 0,
        amax_history_len: int = 1024,
        amax_compute_algo: str = "max",
        reduce_amax: bool = True,
        fp8_dpa: bool = False,
        fp8_mha: bool = False,
    ):
        """Initialize FP8 recipe.

        Args:
            format: FP8 format (hybrid, e4m3, e5m2).
            margin: Margin for amax computation.
            amax_history_len: Length of amax history.
            amax_compute_algo: Algorithm for computing amax ("max" or "most_recent").
            reduce_amax: Whether to reduce amax across distributed group.
            fp8_dpa: Whether to enable FP8 dot product attention.
            fp8_mha: Whether to enable FP8 multi-head attention.

        Raises:
            ImportError: If Transformer Engine is not available.
            ValueError: If format is not recognized or is pure E5M2 (not supported).
        """
        if not is_transformer_engine_available():
            raise ImportError(
                "Transformer Engine is required for FP8 training. "
                "Install it with: pip install transformer-engine"
            )

        self.format = (
            get_fp8_format_from_string(format) if isinstance(format, str) else format
        )
        self.margin = margin
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.reduce_amax = reduce_amax
        self.fp8_dpa = fp8_dpa
        self.fp8_mha = fp8_mha

        # Import Transformer Engine components
        from transformer_engine.common.recipe import (
            DelayedScaling,
            Format,
        )

        # Map our format to TE format
        if self.format == Fp8Format.HYBRID:
            te_format = Format.HYBRID
        elif self.format == Fp8Format.E4M3:
            te_format = Format.E4M3
        elif self.format == Fp8Format.E5M2:
            # E5M2 is not supported by DelayedScaling recipe
            raise ValueError(
                "Pure E5M2 format is not supported by Transformer Engine's DelayedScaling recipe. "
                "Please use 'hybrid' or 'e4m3' format instead."
            )
        else:
            raise ValueError(f"Unsupported FP8 format: {self.format}")

        # Create the Transformer Engine recipe
        self._recipe = DelayedScaling(
            fp8_format=te_format,
            margin=margin,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
            reduce_amax=reduce_amax,
            fp8_dpa=fp8_dpa,
            fp8_mha=fp8_mha,
        )

        logger.info(
            f"Created FP8 recipe with format={self.format.value}, "
            f"amax_history_len={amax_history_len}, "
            f"amax_compute_algo={amax_compute_algo}"
        )

    @property
    def recipe(self):
        """Get the underlying Transformer Engine recipe."""
        return self._recipe

    @contextmanager
    def autocast(self) -> Iterator[None]:
        """Context manager for FP8 autocast (forward pass).

        This context manager handles automatic quantization of inputs to FP8
        and dequantization of outputs.

        Example:
            with fp8_recipe.autocast():
                output = model(input)

        Yields:
            None
        """
        with self._recipe:
            yield

    @contextmanager
    def backward(self) -> Iterator[None]:
        """Context manager for FP8 backward pass.

        This context manager prepares gradients for the backward pass
        and handles any necessary scaling.

        Example:
            with fp8_recipe.backward():
                loss.backward()

        Yields:
            None
        """
        with self._recipe:
            yield

    def prepare_for_backward(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Prepare inputs for backward pass.

        Args:
            *args: Input tensors to prepare.
            **kwargs: Additional keyword arguments.

        Returns:
            Prepared tensors.
        """
        return self._recipe.prepare_for_backward(*args, **kwargs)

    def prepare(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Prepare inputs for forward pass.

        Args:
            *args: Input tensors to prepare.
            **kwargs: Additional keyword arguments.

        Returns:
            Prepared tensors.
        """
        return self._recipe.prepare(*args, **kwargs)


def create_fp8_recipe(
    format: str | Fp8Format = Fp8Format.HYBRID,
    margin: int = 0,
    amax_history_len: int = 1024,
    amax_compute_algo: str = "max",
    reduce_amax: bool = True,
    fp8_dpa: bool = False,
    fp8_mha: bool = False,
) -> FP8Recipe | None:
    """Create an FP8 recipe if Transformer Engine is available.

    This is a convenience function that returns an FP8Recipe if Transformer
    Engine is available, or None if it is not (allowing for graceful fallback).

    Args:
        format: FP8 format (hybrid, e4m3, e5m2).
        margin: Margin for amax computation.
        amax_history_len: Length of amax history.
        amax_compute_algo: Algorithm for computing amax ("max" or "most_recent").
        reduce_amax: Whether to reduce amax across distributed group.
        fp8_dpa: Whether to enable FP8 dot product attention.
        fp8_mha: Whether to enable FP8 multi-head attention.

    Returns:
        FP8Recipe instance if Transformer Engine is available, None otherwise.

    Example:
        fp8_recipe = create_fp8_recipe("hybrid", amax_history_len=1024)
        if fp8_recipe is not None:
            # Use FP8 training
            with fp8_recipe.autocast():
                output = model(input)
        else:
            # Fall back to standard training
            output = model(input)
    """
    if not is_transformer_engine_available():
        logger.warning(
            "Transformer Engine not available. FP8 training will be disabled. "
            "Install with: pip install transformer-engine"
        )
        return None

    return FP8Recipe(
        format=format,
        margin=margin,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
        reduce_amax=reduce_amax,
        fp8_dpa=fp8_dpa,
        fp8_mha=fp8_mha,
    )


@contextmanager
def fp8_autocast(
    device: torch.device,
    fp8_recipe: FP8Recipe | None,
    enabled: bool = True,
    dtype: torch.dtype | None = None,
) -> Iterator[None]:
    """Unified autocast context manager that handles FP8 and standard AMP.

    This context manager provides a unified interface for both FP8 (via
    Transformer Engine) and standard AMP (via PyTorch autocast).

    Args:
        device: The target device for autocast.
        fp8_recipe: Optional FP8Recipe instance. If provided and enabled,
            uses FP8 autocast. Otherwise falls back to standard AMP.
        enabled: Whether to enable autocast.
        dtype: The dtype for standard AMP autocast. If None, uses float16.

    Yields:
        None

    Example:
        # With FP8
        fp8_recipe = create_fp8_recipe("hybrid")
        with fp8_autocast(device, fp8_recipe, dtype=torch.bfloat16):
            output = model(input)

        # Without FP8 (falls back to AMP)
        with fp8_autocast(device, None, dtype=torch.bfloat16):
            output = model(input)
    """
    if not enabled:
        yield
        return

    if fp8_recipe is not None and device.type == "cuda":
        # Use FP8 autocast
        with fp8_recipe.autocast():
            yield
    else:
        # Use standard AMP autocast
        if dtype is None:
            dtype = torch.float16
        with torch.autocast(device.type, dtype=dtype, enabled=True):
            yield


@contextmanager
def fp8_backward(
    fp8_recipe: FP8Recipe | None,
    enabled: bool = True,
) -> Iterator[None]:
    """Unified backward context manager for FP8.

    Args:
        fp8_recipe: Optional FP8Recipe instance.
        enabled: Whether to enable FP8 backward handling.

    Yields:
        None

    Example:
        with fp8_backward(fp8_recipe):
            loss.backward()
    """
    if not enabled or fp8_recipe is None:
        yield
        return

    with fp8_recipe.backward():
        yield
