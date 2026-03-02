import numpy as np
import torch

Sclarable = float | int | torch.Tensor | np.ndarray


def get_item(number: Sclarable) -> float | int | bool:
    """Extract the scalar value from a tensor or array."""
    if isinstance(number, torch.Tensor):
        assert number.numel() == 1, "Tensor must be a scalar."
        return number.item()
    elif isinstance(number, np.ndarray):
        assert number.size == 1, "Array must be a scalar."
        return number.item()
    else:
        return number


def safe_round(number: Sclarable, ndigits: int | None) -> float | int:
    """Safely round a number, handling various numeric types.

    This function handles rounding for Python numbers, PyTorch tensors, and
    NumPy arrays. It recursively handles nested types (e.g., single-element
    tensors) until it reaches a roundable Python number.

    Args:
        number: The number to round. Can be a Python number, PyTorch tensor,
            or NumPy array.
        ndigits: Number of decimal places to round to. If None, returns the
            number unchanged.

    Returns:
        Rounded number as float or int (depending on whether rounding occurred).

    Example:
        ```python
        safe_round(3.14159, 2)  # 3.14
        safe_round(torch.tensor(3.14159), 2)  # 3.14
        safe_round(3.14159, None)  # 3.14159 (no rounding)

        ```"""
    number = get_item(number)
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    return number
