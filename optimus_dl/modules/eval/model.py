import logging

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm.auto import tqdm

from optimus_dl.modules.model.base import BaseModel

logger = logging.getLogger(__name__)


class LLMBaselinesModel(LM):
    """LLM Baselines evaluation model for llm_harness.

    This class implements the lm_eval interface using a pre-loaded model and tokenizer.
    All checkpoint loading logic is handled by the EvalRecipe.
    """

    def __init__(
        self,
        model: BaseModel,
        tokenizer,
        tokenizer_config,
        device: str | torch.device,
    ):
        """Initialize the model with pre-loaded components.

        Args:
            model: Pre-loaded BaseModel instance
            tokenizer: Pre-loaded tokenizer instance
            tokenizer_config: Tokenizer configuration for type detection
            device: Device the model is running on
        """
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.device = device

        logger.info(f"LLMBaselinesModel initialized on {self.device}")

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(text)
        return torch.as_tensor(encoded, dtype=torch.long)

    def _decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens to text using the model's tokenizer."""
        return self.tokenizer.decode(tokens)

    def _compute_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for input tokens."""
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            outputs = self.model(input_ids)
            logits = outputs["logits"]  # Shape: [batch, seq_len, vocab_size]

            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating continuations from contexts.

        Args:
            requests: List of Instance objects with (context, continuation) pairs

        Returns:
            List of (logprob, is_greedy) tuples
        """
        results = []

        for request in tqdm(requests, desc="Computing loglikelihood", leave=False):
            args = request.args
            if not isinstance(args, tuple) or len(args) != 2:
                raise ValueError(f"Expected 2 arguments for loglikelihood, got {args}")
            context, continuation = args

            # Tokenize context and continuation
            context_tokens = self._tokenize(context)
            continuation_tokens = self._tokenize(continuation)

            # Combine for full sequence
            full_tokens = torch.cat([context_tokens, continuation_tokens])

            # Get log probabilities
            log_probs = self._compute_logprobs(full_tokens)

            # Calculate log probability of continuation
            context_len = len(context_tokens)
            continuation_len = len(continuation_tokens)

            if continuation_len == 0:
                logprob = 0.0
                is_greedy = True
            else:
                # Get logprobs for continuation tokens
                relevant_logprobs = log_probs[
                    0, context_len - 1 : context_len + continuation_len - 1
                ]
                token_logprobs = relevant_logprobs[
                    torch.arange(continuation_len), continuation_tokens
                ]
                logprob = token_logprobs.sum().item()

                # Check if this would be the greedy choice
                greedy_tokens = log_probs[
                    0, context_len - 1 : context_len + continuation_len - 1
                ].argmax(dim=-1)
                is_greedy = torch.equal(
                    greedy_tokens, continuation_tokens.to(greedy_tokens.device)
                )

            results.append((logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Compute rolling log-likelihood for perplexity calculation.

        Args:
            requests: List of Instance objects with (text,) tuples

        Returns:
            List of log probabilities
        """
        results = []

        for request in requests:
            args = request.args
            if not isinstance(args, tuple) or len(args) != 1:
                raise ValueError(
                    f"Expected 1 argument for loglikelihood_rolling, got {args}"
                )
            text = args[0]

            # Tokenize text
            tokens = self._tokenize(text)

            if len(tokens) <= 1:
                results.append(0.0)
                continue

            # Get log probabilities
            log_probs = self._compute_logprobs(tokens)

            # Calculate total log probability (excluding first token)
            token_indices = tokens[1:]  # Target tokens
            context_logprobs = log_probs[0, :-1]  # Logprobs from context positions

            selected_logprobs = context_logprobs[
                torch.arange(len(token_indices)), token_indices
            ]
            total_logprob = selected_logprobs.sum().item()

            results.append(total_logprob)

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text until stopping criteria are met.

        Args:
            requests: List of Instance objects with (context, gen_kwargs) pairs

        Returns:
            List of generated text continuations
        """
        results = []

        for request in requests:
            args = request.args
            if not isinstance(args, tuple) or len(args) != 2:
                raise ValueError(f"Expected 2 arguments for generate_until, got {args}")
            context, gen_kwargs = args

            # Parse generation arguments
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            until = gen_kwargs.get("until", [])
            temperature = gen_kwargs.get("temperature", 0.0)  # Greedy by default

            # Tokenize context
            context_tokens = self._tokenize(context)
            input_ids = context_tokens.unsqueeze(0).to(self.device)

            # Generate tokens
            generated_tokens = []

            with torch.no_grad():
                for _ in range(max_gen_toks):
                    # Get next token logits
                    outputs = self.model(input_ids)
                    logits = outputs["logits"][:, -1, :]  # Last position

                    # Sample next token
                    if temperature == 0.0:
                        next_token = logits.argmax(dim=-1)
                    else:
                        probs = F.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1).squeeze(-1)

                    generated_tokens.append(next_token.item())

                    # Update input for next iteration
                    input_ids = torch.cat(
                        [input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
                    )

                    # Check stopping criteria
                    current_text = self._decode_tokens(generated_tokens)
                    if any(stop_seq in current_text for stop_seq in until):
                        # Truncate at first stopping sequence
                        for stop_seq in until:
                            if stop_seq in current_text:
                                current_text = current_text.split(stop_seq)[0]
                                break
                        break

            generated_text = (
                self._decode_tokens(generated_tokens) if generated_tokens else ""
            )

            # Remove stopping sequences from the end
            for stop_seq in until:
                if generated_text.endswith(stop_seq):
                    generated_text = generated_text[: -len(stop_seq)]

            results.append(generated_text)

        return results

    @property
    def tokenizer_name(self) -> str:
        """Return tokenizer name for caching."""
        return f"{self.tokenizer_config.type}:{self.tokenizer_config.name}"
