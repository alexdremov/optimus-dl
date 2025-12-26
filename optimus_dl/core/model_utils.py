"""Model utility functions.

This module provides helper functions for working with PyTorch models, such as
counting parameters, analyzing model structure, and other common operations.
"""

import torch


def get_num_parameters(model: torch.nn.Module) -> int:
    """Count the total number of parameters in a model.

    This function counts all parameters in the model, including both trainable
    and non-trainable parameters. It uses a set to handle cases where parameters
    might be shared across modules (though this is rare in practice).

    Args:
        model: PyTorch model to count parameters for.

    Returns:
        Total number of parameters (sum of all parameter tensor sizes).

    Example:
        ```python
        model = Llama(LlamaConfig(n_embd=512, n_head=8, n_layer=12))
        num_params = get_num_parameters(model)
        print(f"Model has {num_params:,} parameters")
        # Model has 123,456,789 parameters
        ```
    """
    params = set()
    for param in model.parameters():
        params.add(param)
    return sum(param.numel() for param in params)
