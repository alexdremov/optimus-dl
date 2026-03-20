# Metrics Loggers

Optimus-DL supports multiple logging backends to track training progress, visualize metrics, and store experiment metadata. Loggers are configured in the `loggers` section of the training configuration.

The framework uses a registry-based system for loggers, allowing you to easily swap or combine different logging backends.

## Available Loggers

### 1. JSONL Logger (`jsonl`)
Writes metrics to JSON Lines files, which are easy to parse and process with standard tools.

**Key Features:**
- Automatic file rotation based on size.
- Separate files for different metric groups (e.g., `train`, `eval`).
- Exports the full experiment configuration to JSON and YAML for reproducibility.

**Configuration Example:**
```yaml
loggers:
  _name: jsonl
  output_dir: "logs"
  filename: "metrics.jsonl"
  flush_every: 10
```

### 2. Weights & Biases Logger (`wandb`)
Integrates with [Weights & Biases](https://wandb.ai/) for interactive experiment tracking and visualization.

**Key Features:**
- Support for online and offline modes.
- Automatic run resumption using stored `run_id`.
- Logs the training logs directory as a W&B artifact for later inspection (model checkpoints are not logged automatically).
- Grouping and job type tagging.

**Configuration Example:**
```yaml
loggers:
- _name: wandb
  project: llm-baselines
  mode: online
  enabled: true
```

### 3. MLflow Logger (`mlflow`)
Integrates with [MLflow](https://mlflow.org/) for experiment tracking, supporting both local and remote tracking servers.

**Key Features:**
- **Asynchronous Logging**: High-performance logging using a background thread (native in MLflow 3.x).
- **System Metrics**: Automatic tracking of CPU, GPU, and Memory usage.
- **Environment Tagging**: Automatically logs Python version, platform, and hostname.
- **Config Persistence**: Saves the full experiment configuration as an artifact and as parameters (with automatic truncation and data-loss prevention for long strings).
- **Run Resumption**: Supports resuming existing runs using the stored `run_id`.

**Configuration Example:**
```yaml
loggers:
- _name: mlflow
  tracking_uri: "https://mlflow.domain.com"
  name: "llama-training"
```

## Configuring Multiple Loggers

You can configure multiple loggers by using a list in your configuration.

## Logging Custom Metrics

All metrics computed by the [Metrics Engine](metrics.md) are automatically passed to all enabled loggers. Each logger is responsible for formatting and sending these metrics to its respective backend.
