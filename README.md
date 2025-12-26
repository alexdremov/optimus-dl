# Optimus-DL

**Optimus-DL** is a modular, high-performance research framework for training Large Language Models (LLMs) and other deep learning models. It leverages modern PyTorch features (AMP, DDP, Compile) and a flexible, composition-based architecture.

## Key Features

- **Modular "Recipe" Architecture**: Clean separation between model definitions, data pipelines, and training logic.
- **Hydra-based Configuration**: Hierarchical, type-safe, and easily conveniently override-able configurations.
- **Universal Metrics System**: Lazy evaluation and automatic distributed aggregation of metrics.
- **Modern PyTorch**: Built-in support for Mixed Precision (AMP), FSDP2, Tensor Parallelism, Sequence Parallelism, and `torch.compile`.
- **Efficient Kernels**: Integrated support for [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) for memory-efficient and fast RMSNorm, SwiGLU, and CrossEntropy.
- **Registry System**: easy dependency injection and component swapping via a centralized registry.

The core idea of making everything modular and replacable is to make research experiments easy to implement cleanly.

## Supported Models

Optimus-DL includes highly optimized implementations of:

- **Llama 2 / 3**: Full support for GQA, RoPE, and various sharding strategies.
- **Qwen**: Support for Qwen-style attention (Q/K Norm) and architectures.
- **GPT-2**: Classic architecture for baselining.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd optimus-dl

# Install in editable mode with dependencies
pip install -e .
```

### Training

Training is orchestrated via `scripts/train.py` using Hydra configs.

```bash
# Run with default configuration
python scripts/train.py

# Override specific parameters
python scripts/train.py model=gpt2 optimization.batch_size=64 common.use_gpu=true

# Your own config
python scripts/train.py --config-name=train_llama
```

### Writing Train Configs

This project uses [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management. Configurations are hierarchical and composable, allowing you to mix and match models, datasets, and training strategies.

#### Structure & Interpolation

Configs are located in `configs/train/`. A typical training config composes defaults (model, optimizer, scheduler) and then overrides specific parameters.

We use a special `args` section as a "scratch space" for high-level variables. These are referenced throughout the config using OmegaConf's interpolation syntax `${...}`. This ensures consistency (e.g., setting `seq_len` in one place updates both the model and the data pipeline).

```yaml
_name: base
args:
  name: my-experiment
  batch_size: 64
  seq_len: 1024

# ... later in the config ...
optimization:
  iterations: ${args.iterations}

data:
  scratch:
    base_transforms:
      _name: compose
      transforms:
        # ...
        - _name: flat_batcher
          batch_size: ${args.batch_size} # Interpolated from args
          seq_len: ${args.seq_len}
```

#### Data Pipelines & `data.scratch`

The `data` section typically defines `train_datasets` and `eval_datasets`. To avoid repeating complex transform chains, we define them in `data.scratch` and reference them via interpolation.

```yaml
data:
  scratch:
    # Define the transform chain once
    my_transform:
      _name: compose
      transforms:
        - _name: tokenize
          tokenizer_config: {_name: tiktoken, name: gpt2}
        - _name: to_device

  train_datasets:
    source:
      _name: loop
      inner: {_name: preset_dataset, split: train}
    # Reference the transform
    transform: ${data.scratch.my_transform}
```

### Hydra & Omegaconf Extra Quick Guide

Here are some power-user features you'll likely use:

- **Overriding Defaults**: You can swap out entire components from the command line.

```bash
# Switch the model to GPT-2 and optimizer to SGD
python scripts/train.py model=gpt2 optimization/optimizer=sgd
```

- **Multirun (`-m`)**: Run multiple experiments sequentially with a sweep.

```bash
# Run 3 experiments with different learning rates
python scripts/train.py -m optimization.optimizer.lr=1e-3,1e-4,1e-5
```

- **Interpolation**: Reference other config values dynamically.
    - `${layout.param}`: Standard interpolation.
    - `${oc.env:VAR_NAME}`: Read from environment variable `VAR_NAME`.
    - `${.relative_param}`: Relative path interpolation.
    - `${eval:expression}`: Evaluate a Python expression. For example, `${eval:"'string' + '_suffix'"}` or `${eval:"int(100 * 0.5)"}`. This is defined in `optimus_dl/core/omegaconf.py`.

- **Debugging**: See the resolved configuration without running the code.

```bash
# Print the full config structure
python scripts/train.py --config-name=train_llama -c job
```

### Framework Internals

Understanding these core components is crucial for advanced usage and research extensions.

#### Registry System

The framework relies heavily on a registry pattern to decouple configuration from implementation. This allows you to swap components (models, optimizers, schedulers) just by changing the `_name` field in the config.

- **Location**: `optimus_dl/core/registry.py`
- **Usage**:

```python
from optimus_dl.core.registry import make_registry

# Create a new registry
registry, register, build = make_registry("my_component")

@register("my_impl")
class MyImplementation:
    def __init__(self, param): ...

# Build from config
obj = build(RegistryConfig(_name="my_impl", param=1))
```

#### Data Pipeline

Data loading is split into **Sources** and **Transforms**.

- **Source**: Yields raw items (e.g., text, examples).
- **Transforms**: A chain of operations (Tokenize -> Chunk -> Shuffle -> Batch -> ToDevice).

This design allows for highly reusable data processing pipelines. Complex transform chains are often defined in `data.scratch` and referenced in dataset configs.

#### Checkpointing

We use PyTorch's **Distributed Checkpoint (DCP)** API for efficient, sharded saving/loading of large models.

- **Structure**: Checkpoints are directories containing sharded tensor data and a metadata file.
- **Manager**: `CheckpointManager` handles the complexity of saving model, optimizer, scheduler, and dataloader states.
- **Auto-Resume**: The training loop automatically detects the latest checkpoint in the output directory and resumes from it.

**LoadStrategy**:
For fine-tuning or experiments, you might want to load only parts of a checkpoint. The `LoadStrategy` class (`optimus_dl/modules/checkpoint/load_strategy.py`) controls this.

- `load_model` (bool): Load model weights.
- `load_optimizer` (bool): Load optimizer state.
- `load_scheduler` (bool): Load learning rate scheduler state.
- `load_data_sources` (bool): Load data source state (e.g. readers position).
- `load_dataloaders` (bool): Load full dataloader state.
- `load_metrics` (bool): Load accumulated metrics.
- `load_iteration` (bool): Resume iteration count.
- `extra_ignore_keys` (list): Specific keys to ignore in the checkpoint state dict.

### Advanced Usage

#### Model Transforms

Optimus-DL applies transformations to the model after initialization but before training. This is where distributed wrappers and compilation happen.

- **Config**: `model_transforms` list in `train.yaml`.
- **Common Transforms**:
    - `ddp`: Standard DistributedDataParallel.
    - `fully_shard`: PyTorch FSDP2 (Fully Sharded Data Parallel). Supports mixed precision, CPU offloading, and mesh sharding.
    - `compile`: `torch.compile` for graph optimization.

```yaml
model_transforms:
  - _name: fully_shard
    mixed_precision:
      param_dtype: bfloat16
      reduce_dtype: float32
  - _name: compile
```

#### Evaluation with `lm_eval`

The framework integrates with the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) for standardized benchmarks.

- **Script**: `scripts/eval.py`
- **Config**: `configs/eval/default.yaml`

```bash
# Evaluate a checkpoint on Hellaswag and MMLU
python scripts/eval.py \
    common.checkpoint_path=outputs/my-run/checkpoint_00010000 \
    lm_eval.tasks=[hellaswag,mmlu] \
    lm_eval.batch_size=8
```

More advanced:
```bash
python scripts/eval.py --config-name quick_pretrained \
          common.checkpoint_path=null ++common.model._name=preset_hfllama2 ++common.model.hf_model_name=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
          lm_eval.tasks=[hellaswag,mmlu] \
          lm_eval.batch_size=4
```

#### Serving Models

Optimus-DL provides a simple serving script for deploying trained models as an OpenAI-compatible API endpoint. This uses `scripts/serve.py`.

- **Script**: `scripts/serve.py`
- **Config**: `configs/serve/`

```bash
# Serve a TinyLlama model
python scripts/serve.py --config-name=tinyllama
```

Make requests:
```bash
curl -X POST http://127.0.0.1:8000//v1/chat/completions \
-d '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], "max_tokens": 100, "temperature": 0.01}'
```

```bash
curl -X POST http://localhost:8000/v1/completions -d '{"prompt": "All:", "max_tokens": 50, "temperature": 0.01}'
```

## Project Structure

- `optimus_dl/`: Main package source code.
    - `core/`: Fundamental utilities (logging, registry, device management).
    - `modules/`: Pluggable components (models, optimizers, data loaders).
    - `recipe/`: Orchestration logic (training loops, evaluation).
- `configs/`: Hierarchical Hydra configuration files.
- `scripts/`: Entry points.

## Development

The project enforces strict code quality standards.

```bash
# Run tests
pytest

# Format code
black .
isort .
ruff check --fix .
```

## License

MIT License.
