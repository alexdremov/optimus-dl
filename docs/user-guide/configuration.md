# Configuration

Optimus-DL uses [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management. Configurations are hierarchical and composable.

## Structure

Configs are located in `configs/train/`. A typical training config composes defaults (model, optimizer, scheduler) and then overrides specific parameters.

We use a special `args` section as a "scratch space" for high-level variables. These are referenced using OmegaConf's interpolation syntax `${...}`.

```yaml
_name: base
args:
  name: my-experiment
  batch_size: 64
  seq_len: 1024

optimization:
  iterations: ${args.iterations}

data:
  scratch:
    base_transforms:
      _name: compose
      transforms:
        - _name: flat_batcher
          batch_size: ${args.batch_size}
          seq_len: ${args.seq_len}
```

## Data Pipelines

Data loading definitions often use a `data.scratch` section to define transform chains that are referenced by `train_datasets` and `eval_datasets`.

```yaml
data:
  scratch:
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
    transform: ${data.scratch.my_transform}
```

## Advanced Interpolation

*   `${layout.param}`: Standard interpolation.
*   `${oc.env:VAR_NAME}`: Read from environment variable `VAR_NAME`.
*   `${.relative_param}`: Relative path interpolation.
*   `${eval:expression}`: Evaluate a Python expression.

## Overriding Defaults

You can swap out entire components from the command line:

```bash
# Switch the model to GPT-2 and optimizer to SGD
python scripts/train.py model=gpt2 optimization/optimizer=sgd
```
