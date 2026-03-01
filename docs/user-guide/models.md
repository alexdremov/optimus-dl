# Models

The `optimus_dl.modules.model` package contains the core model architectures implemented in the framework. These models are designed to be modular, configurable, and compatible with various distributed training strategies like Tensor Parallelism.

See the full [Model API Reference](../reference/modules/model/index.md) for detailed documentation on all classes and functions.

## Available Models

Below is a list of the primary model implementations available in Optimus-DL.

- [`llama2`](../reference/modules/model/llama2.md): A highly optimized implementation of the Llama 2 and Llama 3 architectures, including support for Grouped-Query Attention (GQA) and Rotary Position Embeddings (RoPE).
- [`gpt2`](../reference/modules/model/gpt2.md): A classic implementation of the GPT-2 architecture, often used for baselining and research.
- [`qwen3`](../reference/modules/model/qwen3.md): An implementation of the Qwen architecture (supporting Qwen 2 and 2.5), featuring Q/K Normalization for improved attention stability.
- [`olmo3`](../reference/modules/model/olmo3.md): An implementation of the OLMo3 architecture, featuring alternating sliding window and full attention, YaRN RoPE, and SwiGLU MLP.
- [`base`](../reference/modules/model/base.md): Defines the base interface for all models, ensuring consistent API for training and evaluation.
- [`config`](../reference/modules/model/config.md): Provides structured `dataclass` configurations for all supported models, validated with `hydra-zen`.

Llama, Qwen, and OLMo support loading pretrained weights from HuggingFace via the `preset_hf*` model types.
