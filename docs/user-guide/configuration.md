# Configuration Guide

Optimus-DL uses [Hydra](https://hydra.cc/) for its configuration system, enabling flexible, hierarchical, and composable setups. All training configurations are located in the `configs/` directory.

## Core Concepts

### 1. Hierarchical Structure
Configurations are built in layers. A main training config (e.g., `train_llama.yaml`) specifies defaults for different components like the model, optimizer, and data.

**Example from `train_llama.yaml`:**
```yaml
defaults:
  - defaults.yaml
  - lr_scheduler: wsd
  - model: llama2
  - criterion: cross_entropy
  - loggers: basic
  - optimization/amp: bfloat16
  - _self_
```
Each item in the `defaults` list points to another YAML file, allowing you to mix and match components easily.

### 2. The `args` Section
We use a special `args` section as a "scratch space" for high-level variables that are reused throughout the configuration. This is the single source of truth for important parameters like batch size, sequence length, and vocabulary size.

```yaml
args:
  name: llama-finetune
  batch_size: 64
  seq_len: 1024
  vocab_size: 32000
```

These values are then referenced in other parts of the config using OmegaConf's interpolation syntax (`${...}`).

### 3. Interpolation
Interpolation is key to keeping the configuration DRY (Don't Repeat Yourself).

```yaml
model:
  vocab_size: ${args.vocab_size} # From args

data:
  train_datasets:
    transform:
      _name: flat_batcher
      batch_size: ${args.batch_size} # From args
      seq_len: ${args.seq_len}       # From args
```

You can also evaluate simple expressions:
```yaml
args:
  global_batch_size: 128
  num_devices: 8

# Calculate per-device batch size
per_device_batch_size: ${eval:"int(${args.global_batch_size} / ${args.num_devices})"}
```

## Key Configuration Sections

### `model`
This section defines the model architecture. The `_name` key determines which model to build (e.g., `llama2`, `gpt2`). Other parameters are specific to the model, such as the number of layers, hidden size, and number of attention heads.

```yaml
model:
  _name: llama2
  vocab_size: ${args.vocab_size}
  n_layer: 12
  n_head: 12
  hidden_dim: 768
```

### `data`
This section defines the entire data pipeline, including training and evaluation datasets. It typically contains:
- `train_datasets` and `eval_datasets`: Define the data sources and transforms.
- `scratch`: A reusable space to define complex transform chains that can be referenced via interpolation.

```yaml
data:
  scratch:
    # Define a reusable transform chain
    my_transform:
      _name: compose
      transforms:
        - _name: tokenize
          tokenizer_config:
            _name: tiktoken
            name: gpt2
        - _name: chunk_tokens
          max_seq_len: ${args.seq_len}
        - _name: shuffle
          seed: 42
          buffer_size: 8096
        - _name: flat_batcher
          batch_size: ${args.batch_size}
          seq_len: ${args.seq_len}
        - _name: prefetch
        - _name: to_device

  train_datasets:
    source:
      _name: loop
      inner:
        _name: preset_slimpajama6b
        split: train
    transform: ${data.scratch.my_transform} # Reference the chain
```

### `optimization`
This section controls the optimization process, including the optimizer, learning rate scheduler, and gradient clipping.

```yaml
# Optimization settings
optimization:
  iterations: ${args.iterations}
  acc_steps: 1          # Gradient accumulation steps
  clip_grad_norm: 5.0   # Gradient clipping norm

  optimizer:
    _name: adamw
    lr: 5e-4            # Base learning rate
    weight_decay: 1e-1
    betas: [0.9, 0.99]  # Adam beta parameters
    eps: 1e-8           # Adam epsilon
```

## Command-Line Overrides

One of Hydra's most powerful features is the ability to override any configuration value from the command line.

```bash
# Override the learning rate and batch size
python scripts/train.py \
  optimization.optimizer.lr=0.01 \
  args.batch_size=32

# Swap out the entire model for GPT-2
python scripts/train.py model=gpt2
```

This makes experimentation fast and easy without needing to modify the underlying configuration files.

