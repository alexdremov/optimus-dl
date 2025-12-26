# Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/alexdremov/optimus-dl
cd optimus-dl

# Install in editable mode with dependencies
pip install -e .
```

## Training

Training is orchestrated via `scripts/train.py` using Hydra configs.

```bash
# Run with default configuration
python scripts/train.py

# Override specific parameters
python scripts/train.py model=gpt2 optimization.batch_size=64 common.use_gpu=true

# Your own config
python scripts/train.py --config-name=train_llama
```

## Evaluation

The framework integrates with the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

```bash
# Evaluate a checkpoint on Hellaswag and MMLU
python scripts/eval.py \
    common.checkpoint_path=outputs/my-run/checkpoint_00010000 \
    lm_eval.tasks=[hellaswag,mmlu] \
    lm_eval.batch_size=8
```

## Serving

Serve trained models as an OpenAI-compatible API endpoint.

```bash
# Serve a TinyLlama model
python scripts/serve.py --config-name=tinyllama
```


