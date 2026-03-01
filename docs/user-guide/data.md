# Data Pipelines

Data handling in Optimus-DL is designed to be highly flexible and modular, allowing for complex data processing pipelines to be constructed from reusable components. The core components are located in `optimus_dl.modules.data`.

The key components are:
- **Sources**: Yield raw data items, like lines from a text file or examples from a Hugging Face dataset.
- **Transforms**: A chain of operations applied to the data, such as tokenization, chunking, shuffling, and batching.

For detailed information, see the [Data API Reference](../reference/modules/data/index.md).

## Core Components

- [`datasets`](../reference/modules/data/datasets/index.md): Contains various dataset implementations, including tokenized datasets, and utilities for handling different data formats.
- [`presets`](../reference/modules/data/presets/index.md): Provides some predefined datasets for common use cases.
- [`transforms`](../reference/modules/data/transforms/index.md): Includes a wide range of data transformations, from tokenization to batching and device placement.

## Pre-tokenized Datasets & Strategies

The `TokenizedDataset` is a high-performance dataset that streams tokens from memory-mapped numpy shards. It supports pluggable **Sampling Strategies** to control how documents are traversed:

- **`document`**: Yields full documents as they appear in the dataset.
- **`concat_random`**: Treats the entire dataset as a single concatenated stream of tokens, splits it into fixed-size chunks, and yields them in a globally random order. This is highly efficient for training as it ensures constant sequence lengths and full data utilization.

## Efficient Batching

Optimus-DL provides specialized transforms for efficient token batching:

- **`flat_batcher`**: Concatenates multiple variable-length sequences into a single flat tensor, accompanied by sequence length metadata. This avoids padding overhead and is compatible with FlashAttention and other kernel-level optimizations.
- **`basic_batcher`**: A standard batcher that pads sequences to a fixed length. Also supports flat batching to pack variable length sequences into a single tensor.
