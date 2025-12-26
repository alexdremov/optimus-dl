# Models

The `optimus_dl.modules.model` package contains the core model architectures implemented in the framework. These models are designed to be modular, configurable, and compatible with various distributed training strategies like Tensor Parallelism.

See the full [Model API Reference](../reference/modules/model/) for detailed documentation on all classes and functions.

## Available Models

Below is a list of the primary model implementations available in Optimus-DL.

- [`llama2`](../reference/modules/model/llama2/): A highly optimized implementation of the Llama 2 and Llama 3 architectures, including support for Grouped-Query Attention (GQA) and Rotary Position Embeddings (RoPE).
- [`gpt2`](../reference/modules/model/gpt2/): A classic implementation of the GPT-2 architecture, often used for baselining and research.
- [`qwen`](../reference/modules/model/qwen/): An implementation of the Qwen architecture, featuring Q/K LayerNorm for improved attention mechanics.
- [`base`](../reference/modules/model/base/): Defines the base interface for all models, ensuring consistent API for training and evaluation.
- [`config`](../reference/modules/model/config/): Provides structured `dataclass` configurations for all supported models, validated with `hydra-zen`.

Llama and Qwen support loading pretrained weights from HuggingFace.
