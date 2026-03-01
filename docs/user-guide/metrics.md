# Metrics Engine

Optimus-DL features a sophisticated **Metrics Engine** designed for complex, modular, and efficient metric calculation during training and evaluation. Unlike simple average meters, the `MetricEngine` coordinates data producers (**Sources**) and compute logic (**Metrics**) through a protocol-based handshake system.

## Key Concepts

The engine is built around three primary abstractions:

1.  **MetricSource**: Data producers that extract or transform information from the model, batch, or other sources.
2.  **Metric**: Stateless compute logic that takes data from sources and produces raw values for aggregation.
3.  **Accumulators**: Stateful objects (Average, Sum, Gather, Perplexity) that aggregate raw values across batches and ranks.

### Protocol Handshake

The `MetricEngine` uses a "protocol" system to decouple metrics from sources. A source declares what data it **provides** (e.g., `logits`, `loss`), and a metric declares what it **requires**. The engine automatically validates that all requirements are met before starting.

## Configuration

Metrics are configured in the `metrics` section of the training configuration. You can define multiple groups (e.g., `train`, `eval`).

```yaml
metrics:
  train:
    - _name: source_group
      prefix: "generation"
      sources:
        # Define sources for this group
        my_source:
          _name: causal_lm_source
      metrics:
        # Define metrics that use those sources
        - _name: perplexity
        - _name: accuracy
```

### Source Groups

A `source_group` allows you to run a set of metrics over a specific set of sources. This is useful for:
-   **Nesting**: Logging metrics under a specific prefix (e.g., `generation/accuracy`).
-   **Dependency Injection**: Reusing the same metric logic with different data sources.

## Built-in Components

### Standard Sources
-   **`causal_lm_source`**: Extracts `logits` and `loss` from a standard Causal Language Model forward pass.
-   **`classification_source`**: Prepares data for classification metrics (predictions and targets).

### Standard Metrics
-   **`loss`**: Reports the raw loss.
-   **`accuracy`**: Computes Top-1 accuracy.
-   **`perplexity`**: Computes perplexity ($e^{	ext{loss}}$).

## Advanced Features

### Lazy Evaluation & Caching
The engine ensures that each `MetricSource` is executed at most once per batch, even if multiple metrics depend on it. It uses a configuration-based hash to cache results across different source groups.

### External Protocols
If a model forward pass already computes some data (like `logits` or `loss`), these can be passed directly to the engine to avoid redundant computation. The engine identifies which protocols are "external" and expects them to be provided during the `update` call.

### Distributed Aggregation
The engine integrates with the framework's distributed system. Metrics can be aggregated across all ranks (e.g., global average accuracy) before final reporting.

## Implementing Custom Metrics

To implement a new metric, inherit from `Metric` and register it:

```python
from optimus_dl.modules.metrics.metrics import Metric, register_metric

@register_metric("my_custom_metric")
class MyCustomMetric(Metric):
    @property
    def requires(self):
        return {"my_protocol"}

    @property
    def accumulators(self):
        return {"my_val": "average"}

    def __call__(self, sources_data):
        data = sources_data["my_protocol"]
        # Compute...
        return {"my_val": {"value": 1.0, "weight": 1.0}}
```

## Usage in Recipes

The `MetricEngine` is typically initialized in the training recipe and updated in the main loop:

```python
# Initialization
engine = MetricEngine("train", cfg.metrics.train)

# In the loop
engine.update(
    data={"model": model, "batch": batch},
    computed_data={"loss": loss} # Optional: pass already computed data
)

# After aggregation
final_metrics = engine.compute(raw_aggregated_results)
```
