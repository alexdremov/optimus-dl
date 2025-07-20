# Optimus-DL

**Optimus-DL** is a modular, high-performance research framework for training Large Language Models (LLMs) and other deep learning models. It leverages modern PyTorch features (AMP, DDP, Compile) and a flexible, composition-based architecture.

## Key Features

*   **Modular "Recipe" Architecture**: Clean separation between model definitions, data pipelines, and training logic.
*   **Hydra-based Configuration**: Hierarchical, type-safe, and easily conveniently override-able configurations.
*   **Universal Metrics System**: Lazy evaluation and automatic distributed aggregation of metrics.
*   **Modern PyTorch**: Built-in support for Mixed Precision (AMP), Distributed Data Parallel (DDP), and `torch.compile`.
*   **Registry System**: easy dependency injection and component swapping via a centralized registry.

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
# Run with default configuration (Llama2 on TinyShakespeare)
python scripts/train.py

# Override specific parameters
python scripts/train.py model=gpt2 optimization.batch_size=64 common.use_gpu=true
```

## Project Structure

*   `optimus_dl/`: Main package source code.
    *   `core/`: Fundamental utilities (logging, registry, device management).
    *   `modules/`: Pluggable components (models, optimizers, data loaders).
    *   `recipe/`: Orchestration logic (training loops, evaluation).
*   `configs/`: Hierarchical Hydra configuration files.
*   `scripts/`: Entry points for training and evaluation.

## Development

The project enforces strict code quality standards.

```bash
# Run tests
pytest

# Format code
black .
isort .

# Type check
mypy .
```

## License

MIT License.