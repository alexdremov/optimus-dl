# Data Preparation (Pre-tokenization)

Optimus-DL provides a robust pipeline for preparing large datasets for high-performance training. This process, often called **Pre-tokenization**, involves converting raw text data into sharded, memory-mapped numpy arrays.

## Why Pre-tokenize?

- **Zero Overhead**: Tokenization happens once, not during every training epoch.
- **Memory Mapping**: Shards are loaded using `mmap`, allowing for near-instant random access and minimal RAM usage.
- **Distributed Ready**: Data is sharded so each rank can efficiently read its own portion of the data.

## The Preparation Pipeline

Data preparation is orchestrated by `scripts/pretokenize.py`. The pipeline consists of:

1.  **Source**: Reads raw data (e.g., Hugging Face datasets, JSONL files).
2.  **Processor**: Tokenizes the text using a configured tokenizer (e.g., Llama 3, GPT-2).
3.  **Sharder**: Aggregates tokens and writes them into `.npy` files along with a global `index.json`.

### Example Configuration

Data preparation is configured via Hydra. A typical config (`configs/prepare_data/default.yaml`) looks like this:

```yaml
source:
  _name: hf_source
  path: "HuggingFaceFW/fineweb-edu"
  name: "sample-10BT"
  split: "train"
  streaming: true

tokenizer:
  _name: llama3
  path: "meta-llama/Meta-Llama-3-8B"

sharder:
  output_dir: "outputs/data/fineweb-edu"
  shard_size: 100_000_000 # tokens per shard
  dtype: "uint16"
```

## Running the Pipeline

To start the pre-tokenization process:

```bash
python scripts/pretokenize.py --config-name=default
```

You can override parameters from the command line:

```bash
python scripts/pretokenize.py \
    source.path="roneneldan/TinyStories" \
    sharder.output_dir="outputs/data/tinystories"
```

## Using Pre-tokenized Data

Once prepared, you can use the data in your training config by pointing a `tokenized_dataset` to the output directory:

```yaml
data:
  train_datasets:
    source:
      _name: tokenized_dataset
      data_dir: "outputs/data/fineweb-edu"
    transform: ${data.scratch.my_transform}
```

### Resume Support
The pre-tokenization script supports resuming from a checkpoint. If the process is interrupted, simply run the command again with the same `output_dir`, and it will pick up from the last saved shard.
