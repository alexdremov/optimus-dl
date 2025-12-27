# Optimizers

Optimus-DL leverages PyTorch's standard optimizers and provides additional customized implementations and configurations in the `optimus_dl.modules.optim` package. The framework is designed to easily integrate with different optimization strategies through the Hydra configuration system.

For a complete list of available optimizers and their configurations, see the [Optimizer API Reference](../reference/modules/optim/index.md).

## Key Optimizers

While any PyTorch optimizer can be used, Optimus-DL provides convenient configs and wrappers for the following:

- [`adamw`](../reference/modules/optim/adamw.md): The AdamW optimizer, a standard choice for training transformer models, with options for weight decay and other hyperparameters. It is the default optimizer for most training recipes.
- [`muon`](../reference/modules/optim/muon.md): The Muon optimizer, a momentum-based optimizer with Newton-Schulz iterations for improved convergence.
- [`soap`](../reference/modules/optim/soap.md): The SOAP (Shampoo with Adam in the Preconditioner subspace) optimizer, a second-order optimizer that uses Shampoo preconditioning combined with Adam updates for improved training efficiency.

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

### Muon Configuration Example

```yaml
optimization:
  optimizer:
    _name: muon
    lr: 1e-3
    weight_decay: 0.1
    momentum: 0.95
    nesterov: true
```

### SOAP Configuration Example

```yaml
optimization:
  optimizer:
    _name: soap
    lr: 3e-3
    betas: [0.95, 0.95]
    weight_decay: 0.01
    precondition_frequency: 10
    max_precond_dim: 10000
```

## Choosing an Optimizer

- **AdamW**: Best default choice for most transformer and deep learning models. Well-tested and reliable with good convergence properties.
- **Muon**: Suitable for models where momentum-based optimization with Newton-Schulz iterations can provide faster convergence. Good for certain types of neural networks.
- **SOAP**: Advanced second-order optimizer that can provide better training efficiency for large models, especially when preconditioning can help with optimization landscape. May require more memory due to preconditioner storage.
