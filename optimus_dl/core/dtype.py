"""Dtype utilities for Optimus-DL.

This module provides functions for converting between string representations
and PyTorch dtypes, including support for FP8 formats (E4M3, E5M2).
"""

import torch


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string representation of a dtype to a torch.dtype.

    Supports standard PyTorch dtypes (float32, float16, bfloat16, int8, etc.)
    and FP8 formats (float8_e4m3fn, float8_e5m2).

    Args:
        dtype_str: String representation of the dtype. Can be:
            - Standard names: "float32", "float16", "bfloat16", "int8", etc.
            - Short names: "fp32", "fp16", "bf16", "i8", etc.
            - Full torch names: "torch.float32", "torch.float16", etc.
            - FP8 formats: "float8_e4m3fn", "float8_e5m2", "fp8_e4m3", "fp8_e5m2"
            - FP8 short names: "e4m3", "e5m2"

    Returns:
        The corresponding torch.dtype.

    Raises:
        ValueError: If the dtype string is not supported.

    Example:
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("torch.bfloat16")
        torch.bfloat16
        >>> str_to_dtype("float8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("e5m2")
        torch.float8_e5m2
    """
    dtype_str = dtype_str.lower()

    # Standard dtypes
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    # Short names
    mapping |= {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp64": torch.float64,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
        "u8": torch.uint8,
        "c64": torch.complex64,
        "c128": torch.complex128,
    }

    # FP8 formats (E4M3: 4 exponent bits, 3 mantissa bits)
    if hasattr(torch, "float8_e4m3fn"):
        mapping["float8_e4m3fn"] = torch.float8_e4m3fn
        mapping["fp8_e4m3fn"] = torch.float8_e4m3fn
        mapping["e4m3fn"] = torch.float8_e4m3fn
        mapping["e4m3"] = torch.float8_e4m3fn
        mapping["float8_e4m3"] = torch.float8_e4m3fn
        mapping["fp8_e4m3"] = torch.float8_e4m3fn

    # FP8 formats (E5M2: 5 exponent bits, 2 mantissa bits)
    if hasattr(torch, "float8_e5m2"):
        mapping["float8_e5m2"] = torch.float8_e5m2
        mapping["fp8_e5m2"] = torch.float8_e5m2
        mapping["e5m2fn"] = torch.float8_e5m2
        mapping["e5m2"] = torch.float8_e5m2
        mapping["float8_e5m2fn"] = torch.float8_e5m2
        mapping["fp8_e5m2fn"] = torch.float8_e5m2

    # Full torch names
    mapping |= {f"torch.{k}": v for k, v in mapping.items()}

    if dtype_str not in mapping:
        raise ValueError(
            f"Unsupported dtype string: {dtype_str}, supported: {sorted(mapping.keys())}"
        )
    return mapping[dtype_str]


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    """Check if a dtype is an FP8 format.

    Args:
        dtype: The PyTorch dtype to check.

    Returns:
        True if the dtype is FP8 (E4M3 or E5M2), False otherwise.
    """
    # Build tuple of available FP8 dtypes to avoid AttributeError on PyTorch without FP8
    fp8_dtypes = []
    if hasattr(torch, "float8_e4m3fn"):
        fp8_dtypes.append(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        fp8_dtypes.append(torch.float8_e5m2)
    return dtype in tuple(fp8_dtypes)


def get_fp8_format_from_dtype(dtype: torch.dtype) -> str | None:
    """Get the FP8 format string from a dtype.

    Args:
        dtype: The PyTorch dtype to check.

    Returns:
        The FP8 format string ("e4m3fn" or "e5m2"), or None if not FP8.
    """
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return "e4m3fn"
    elif hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return "e5m2"
    return None
