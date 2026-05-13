import torch


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string representation of a dtype to a torch.dtype."""
    dtype_str = dtype_str.lower()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    mapping |= {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
        "u8": torch.uint8,
    }
    mapping |= {f"torch.{k}": v for k, v in mapping.items()}
    if dtype_str not in mapping:
        raise ValueError(
            f"Unsupported dtype string: {dtype_str}, supported: {list(mapping.keys())}"
        )
    return mapping[dtype_str]
