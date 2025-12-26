# Data Pipelines

Data handling in Optimus-DL is designed to be highly flexible and modular, allowing for complex data processing pipelines to be constructed from reusable components. The core components are located in `optimus_dl.modules.data`.

The key components are:
- **Sources**: Yield raw data items, like lines from a text file or examples from a Hugging Face dataset.
- **Transforms**: A chain of operations applied to the data, such as tokenization, chunking, shuffling, and batching.

For detailed information, see the [Data API Reference](../reference/modules/data/).

## Core Components

- [`datasets`](../reference/modules/data/datasets/): Contains various dataset implementations, including tokenized datasets, and utilities for handling different data formats.
- [`presets`](../reference/modules/data/presets/): Provides some predefined datasets for common use cases.
- [`transforms`](../reference/modules/data/transforms/): Includes a wide range of data transformations, from tokenization to batching and device placement.
