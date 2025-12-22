"""Serving recipe for LLM Baselines models."""

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn.functional as F
from pydantic import ValidationError

from optimus_dl.modules.distributed import build_best_collective
from optimus_dl.modules.tokenizer import build_tokenizer
from optimus_dl.recipe.mixins import ModelBuilder
from optimus_dl.recipe.serve.config import ServeConfig
from optimus_dl.recipe.serve.models import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    CompletionRequest,
    CompletionResponse,
    Delta,
)

logger = logging.getLogger(__name__)


class RequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, response_model):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(response_model.model_dump_json().encode("utf-8"))

    def _send_error(self, status_code: int, error_message: str):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": error_message}).encode("utf-8"))

    def _parse_request(self, model_class):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            return model_class(**data)
        except json.JSONDecodeError as err:
            self._send_error(400, "Invalid JSON")
            raise ValueError from err
        except ValidationError as err:
            self._send_error(422, str(err))
            raise ValueError from err

    def do_POST(self):
        routes = {
            "/v1/completions": self.handle_completions,
            "/v1/chat/completions": self.handle_chat_completions,
        }

        if self.path in routes:
            try:
                routes[self.path]()
            except ValueError:
                pass  # Handled in _parse_request
            except Exception as e:
                logger.error(f"Internal Error: {e}")
                self._send_error(500, str(e))
        else:
            self._send_error(404, "Not Found")

    def handle_completions(self):
        request = self._parse_request(CompletionRequest)

        # Non-streaming only for now for basic completions, or implement stream if needed
        # Assuming request.stream is supported later or ignored.
        # But generate_stream supports it.

        response_text = self.server.recipe.generate(
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_k,
        )

        response = CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    text=response_text,
                    finish_reason="length",  # Simplification
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        self._send_response(response)

    def handle_chat_completions(self):
        request = self._parse_request(ChatCompletionRequest)

        # Convert pydantic messages to dict list for tokenizer
        # request.messages is List[dict] already due to model definition flexibility
        # but pydantic validates it. If it was List[ChatMessage], we would need dump.
        # It is List[dict] in models.py now.
        messages_dicts = request.messages

        if request.stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            generator = self.server.recipe.generate_stream(
                messages_dicts,
                request.max_tokens,
                request.temperature,
                request.top_k,
            )

            id_ = f"chatcmpl-{int(time.time())}"
            created = int(time.time())

            for chunk_text in generator:
                chunk_resp = ChatCompletionChunk(
                    id=id_,
                    object="chat.completion.chunk",
                    created=created,
                    model=request.model,
                    choices=[ChatChunkChoice(index=0, delta=Delta(content=chunk_text))],
                )
                self.wfile.write(f"data: {chunk_resp.model_dump_json()}\n\n".encode())
                self.wfile.flush()

            # Finish chunk
            finish_resp = ChatCompletionChunk(
                id=id_,
                object="chat.completion.chunk",
                created=created,
                model=request.model,
                choices=[ChatChunkChoice(index=0, delta=Delta(), finish_reason="stop")],
            )
            self.wfile.write(f"data: {finish_resp.model_dump_json()}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        else:
            response_text = self.server.recipe.generate(
                messages_dicts,
                request.max_tokens,
                request.temperature,
                request.top_k,
            )

            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            self._send_response(response)


class ServeRecipe:
    """Recipe for serving LLM Baselines models via simple HTTP API."""

    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.device = None

        # Initialize builder with empty config as we load from checkpoint
        self.model_builder = ModelBuilder(None, [])

    def setup(self):
        """Setup model and tokenizer."""
        # Setup device
        if self.cfg.common.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.common.device)

        logger.info(f"Using device: {self.device}")

        # Build collective for potential distributed init
        collective = build_best_collective(
            None if self.device.type == "cuda" else torch.device("cpu")
        )

        assert (self.cfg.common.checkpoint_path is not None) ^ (
            self.cfg.common.model is not None
        ), "Either checkpoint_path or model must be specified, but not both"

        if self.cfg.common.checkpoint_path is not None:
            logger.info(
                f"Loading model from checkpoint: {self.cfg.common.checkpoint_path}"
            )
            self.model, _ = self.model_builder.build_model_from_checkpoint(
                checkpoint_path=self.cfg.common.checkpoint_path, device=self.device
            )
        else:
            logger.info("Building model from config")
            self.model = self.model_builder.build_model(
                model_config=self.cfg.common.model,
                collective=collective,
            )

        self.model.to(self.device)
        self.model.eval()

        # Build tokenizer
        self.tokenizer = build_tokenizer(self.cfg.common.tokenizer)
        logger.info("Model and tokenizer loaded")

    @torch.no_grad()
    def _debug_tokens_log(self, input_ids):
        """Log tokens for debugging."""
        tokens = []
        for token in input_ids.cpu().reshape(-1):
            token = token.item()
            tokens.append(f"{token}:'{self.tokenizer.decode([token])}'")
        logger.debug(f"Input tokens: {' '.join(tokens)}")

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_or_messages: str | list[dict],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """Generate text continuation yielding chunks."""
        if isinstance(prompt_or_messages, list):
            # Apply chat template
            input_ids_list = self.tokenizer.apply_chat_template(
                prompt_or_messages, tokenize=True, add_generation_prompt=True
            )
            input_ids = torch.tensor(
                input_ids_list, dtype=torch.long, device=self.device
            ).unsqueeze(0)
        else:
            if isinstance(prompt_or_messages, list):
                # Handle list of strings? Simple server assumes single string prompt
                prompt_or_messages = prompt_or_messages[0]

            input_ids = torch.tensor(
                self.tokenizer.encode(prompt_or_messages),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

        self._debug_tokens_log(input_ids)

        generated_ids = []
        last_text = ""

        for _ in range(max_new_tokens):
            # Crop context if needed
            if input_ids.size(1) > self.model.config.sequence_length:
                input_cond = input_ids[:, -self.model.config.sequence_length :]
            else:
                input_cond = input_ids

            outputs = self.model(input_cond)
            logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_ids.append(next_token.item())

            # Simple streaming: decode all and yield diff
            # This is inefficient but safe for bytes/utf-8 boundaries
            current_text = self.tokenizer.decode(generated_ids)
            new_text = current_text[len(last_text) :]

            if new_text:
                yield new_text
                last_text = current_text

            if (
                hasattr(self.cfg.common.tokenizer, "eos_token_id")
                and next_token.item() == self.cfg.common.tokenizer.eos_token_id
            ):
                break

    def generate(
        self,
        prompt_or_messages: str | list[dict],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Generate full text continuation."""
        return "".join(
            list(
                self.generate_stream(
                    prompt_or_messages, max_new_tokens, temperature, top_k
                )
            )
        )

    def run(self):
        """Run the server."""
        self.setup()

        server_address = (self.cfg.serve.host, self.cfg.serve.port)
        httpd = HTTPServer(server_address, RequestHandler)
        httpd.recipe = self

        logger.info(f"Serving at http://{self.cfg.serve.host}:{self.cfg.serve.port}")

        # Example payloads
        text_completion_ex = json.dumps(
            {
                "prompt": "Once upon a time",
                "max_tokens": 20,
                "temperature": 0.8,
            }
        )
        logger.info(
            f"Text Completion Example:\ncurl -X POST http://{self.cfg.serve.host}:{self.cfg.serve.port}/v1/completions -d '{text_completion_ex}'"
        )

        chat_completion_ex = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "max_tokens": 20,
                "temperature": 0.8,
                "stream": True,
            }
        )
        logger.info(
            f"Chat Streaming Example:\ncurl -X POST http://{self.cfg.serve.host}:{self.cfg.serve.port}/v1/chat/completions -d '{chat_completion_ex}'"
        )

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()
        logger.info("Server stopped")
