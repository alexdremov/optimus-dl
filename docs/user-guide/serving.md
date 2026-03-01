# Serving Models

Optimus-DL includes a high-performance serving script for deploying trained models as OpenAI-compatible API endpoints. This is useful for building applications that integrate with your trained models.

## 🚀 Serving Your Model

To start the server, use `scripts/serve.py` with a configured model:

```bash
# Serve a pre-trained model with default settings
python scripts/serve.py --config-name=tinyllama
```

The server will start by default at `http://localhost:8000`.

## 📦 API Endpoints

The server provides several OpenAI-compatible endpoints:

- `GET /v1/models`: List available models.
- `POST /v1/completions`: Generate text completions.
- `POST /v1/chat/completions`: Generate chat completions (requires model support).

### Example Request

```bash
curl -X POST http://localhost:8000/v1/completions \
  -d '{"prompt": "The future of AI is", "max_tokens": 50, "temperature": 0.7}'
```

## ⚙️ Configuration

Serving is configured via Hydra. You can customize the model architecture, checkpoint path, and server settings:

```yaml
# configs/serve/default.yaml
model:
  _name: llama2
  vocab_size: 32000
  n_layer: 12
  n_head: 12

server:
  host: "0.0.0.0"
  port: 8000
```

To serve your own model:

```bash
python scripts/serve.py
  common.checkpoint_path=outputs/my-run/checkpoint_latest
  model._name=llama2
```
