# Optimizers

Optimus-DL leverages PyTorch's standard optimizers and provides additional customized implementations and configurations in the `optimus_dl.modules.optim` package. The framework is designed to easily integrate with different optimization strategies through the Hydra configuration system.

For a complete list of available optimizers and their configurations, see the [Optimizer API Reference](../reference/modules/optim/).

## Key Optimizers

While any PyTorch optimizer can be used, Optimus-DL provides convenient configs and wrappers for the following:

- [`adamw`](../reference/modules/optim/adamw/): The AdamW optimizer, a standard choice for training transformer models, with options for weight decay and other hyperparameters. It is the default optimizer for most training recipes.
- [`config`](../reference/modules/optim/config/): Provides structured `dataclass` configurations for all supported optimizers.
