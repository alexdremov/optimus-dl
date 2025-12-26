# Quick Start Guide

This guide will walk you through the essential steps to get Optimus-DL up and running, from installation to starting your first training job.

## 1. Installation

First, clone the repository and install the framework in editable mode. This allows you to easily modify the code and have your changes reflected immediately.

```bash
# Clone the repository
git clone https://github.com/alexdremov/optimus-dl
cd optimus-dl

# Install in editable mode with dependencies
pip install -e .
```

Alternatively, you can use docker: https://hub.docker.com/repository/docker/alexdremov/optimus-dl/general

## 2. Running a Training Job

The easiest way to start is to run a training job with one of the provided default configurations. The `train_llama.yaml` config is a good starting point.

Training is orchestrated by the `scripts/train.py` script.

```bash
# Run with the default Llama configuration
python scripts/train.py --config-name=train_llama

# multi-gpu training
torchrun --nproc_per_node=gpu scripts/train.py --config-name=train_llama
```

This command will:
1. Load the `train_llama.yaml` configuration.
2. Build the model, data pipeline, optimizer, and other components.
3. Start the training loop, which will log progress and save checkpoints to the `outputs/` directory.

## 3. Customizing Your Run

You can easily override any parameter from the configuration file directly on the command line. This is perfect for quick experiments.

```bash
# Override the batch size and use a different model
python scripts/train.py \
    --config-name=train_llama \
    model=gpt2 \
    args.batch_size=32
```

This will start a new training run using the `gpt2` model configuration and a batch size of 32, while keeping all other settings from `train_llama.yaml`.

## 4. Evaluation

Once you have a trained checkpoint, you can evaluate it on standard benchmarks using the integrated [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

```bash
# Evaluate a checkpoint on Hellaswag and MMLU
python scripts/eval.py \
    common.checkpoint_path=outputs/my-run/checkpoint_00010000 \
    lm_eval.tasks=[hellaswag,mmlu] \
    lm_eval.batch_size=8
```

## 5. Serving the Model

Optimus-DL also includes a simple server to deploy your trained models as an OpenAI-compatible API endpoint.

```bash
# Serve a pre-configured TinyLlama model
python scripts/serve.py --config-name=tinyllama
```

You can then send requests to the running server:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'
```

## What's Next?

You've successfully run your first training job! Here's where to go next:

- **Dive into Configuration**: Learn how to create your own comprehensive training workflows in the **[Configuration Guide](configuration.md)**.
- **Explore the Components**: See what's available for **[Models](models.md)**, **[Data Pipelines](data.md)**, and **[Optimizers](optimizers.md)**.
- **Browse the API**: For in-depth details, head to the **[API Reference](../reference/index.md)**.
