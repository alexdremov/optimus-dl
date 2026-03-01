# Optimizers & Schedulers

Optimus-DL leverages PyTorch's standard optimizers and provides additional customized implementations and configurations in the `optimus_dl.modules.optim` and `optimus_dl.modules.lr_scheduler` packages. The framework is designed to easily integrate with different optimization strategies through the Hydra configuration system.

For a complete list of available optimizers and their configurations, see the [Optimizer API Reference](../reference/modules/optim/index.md).

## Key Optimizers

While any PyTorch optimizer can be used, Optimus-DL provides convenient configs and wrappers for the following:

- [`adamw`](../reference/modules/optim/adamw.md): The AdamW optimizer, a standard choice for training transformer models, with options for weight decay and other hyperparameters. It is the default optimizer for most training recipes.
- [`muon`](../reference/modules/optim/muon.md): The Muon optimizer, a momentum-based optimizer with Newton-Schulz iterations for improved convergence.
- [`soap`](../reference/modules/optim/soap.md): The SOAP (Shampoo with Adam in the Preconditioner subspace) optimizer, a second-order optimizer that uses Shampoo preconditioning combined with Adam updates for improved training efficiency.

## Learning Rate Schedulers

Optimus-DL provides a set of learning rate schedulers tailored for large-scale language model training.

- **`wsd` (Warmup-Stable-Decay)**: A popular scheduler for pre-training that consists of three phases:
    1.  **Warmup**: Linear increase from `base_lr / init_div_factor` to `base_lr`.
    2.  **Stable**: Constant learning rate at `base_lr`.
    3.  **Decay (Cooldown)**: Final decay to a target LR. Supports multiple shapes: linear, cosine, exponential, and piecewise linear.
- **`cosine_annealing`**: Standard cosine annealing with optional warmup.
- **`linear_warmup`**: Simple linear warmup followed by a constant or decaying learning rate.

### WSD Configuration Example

```yaml
lr_scheduler:
  _name: wsd
  warmup_steps: 2000
  fract_decay: 0.1
  decay_type: cosine
  final_lr_factor: 0.1
```

## Optimizer Configuration

Optimizers are configured through the Hydra configuration system. Each optimizer has its own set of hyperparameters that can be tuned for your specific use case.

### AdamW Configuration Example

```yaml
optimization:
  optimizer:
    _name: adamw
    lr: 5e-4
    weight_decay: 1e-1
    betas: [0.9, 0.99]
    eps: 1e-8
    fused: true
```
