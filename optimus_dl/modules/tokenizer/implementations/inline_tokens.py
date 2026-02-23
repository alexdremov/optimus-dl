import re
from dataclasses import dataclass
from enum import StrEnum

from omegaconf import MISSING

from optimus_dl.modules.tokenizer import register_tokenizer
from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.modules.tokenizer.config import BaseTokenizerConfig


class UnkStrategy(StrEnum):
    IGNORE = "ignore"
    RAISE = "raise"
    UNK = "unk"


@dataclass
class InlineTokensTokenizerConfig(BaseTokenizerConfig):
    """Configuration for explicitly specified tokens tokenizer.

    Attributes:
        tokens: List of all tokens
        bos_token: Beginning-of-Sequence token.
        eos_token: End-of-Sequence token.
        unk_strategy: How to deal with unknown tokens. Can be ignore, raise or unk.
        Unk will replace the unknown token with the unk token.
    """

    tokens: list[str] = MISSING
    unk_strategy: UnkStrategy = UnkStrategy.RAISE


@register_tokenizer("inline_tokens_tokenizer", InlineTokensTokenizerConfig)
class InlineTokensTokenizer(BaseTokenizer):
    """Inline sequence tokenizer based on an explicitly provided list of tokens.

    Uses regex-based longest-match parsing to tokenize arbitrary strings
    without whitespace assumptions, handling unknown text chunks based on
    the configured strategy.

    Args:
        config: Tokenizer configuration containing the vocabulary and UNK strategy.
    """

    def __init__(self, config: InlineTokensTokenizerConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.tokens = config.tokens

        self._token_to_id = {token: i for i, token in enumerate(self.tokens)}

        # Sort tokens by length (longest first) for greedy regex matching.
        # This prevents partial matches if vocab contains both "A" and "AB".
        sorted_tokens = sorted(self.tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_tokens]

        # A capturing group in re.split() keeps the matched tokens and
        # returns the unmatched chunks between them.
        self._tokenizer_pattern = re.compile(f"({'|'.join(escaped_tokens)})")

        self._vocab_size = len(self.tokens)

        self._bos_token_id = self._vocab_size
        self._vocab_size += 1

        self._eos_token_id = self._vocab_size
        self._vocab_size += 1

        self._unk_token_id = None
        if config.unk_strategy == UnkStrategy.UNK:
            self._unk_token_id = self._vocab_size
            self._vocab_size += 1

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs and add special tokens using regex."""
        if self.config.add_bos:
            token_ids = [self._bos_token_id]
        else:
            token_ids = []

        # re.split() with a capturing group yields alternating un-matched and matched strings
        for chunk in self._tokenizer_pattern.split(text):
            if not chunk:
                continue  # Ignore empty string remnants from splits

            if chunk in self._token_to_id:
                token_ids.append(self._token_to_id[chunk])
            else:
                # If a chunk doesn't match a vocab token, it's treated as unknown.
                if self.config.unk_strategy == UnkStrategy.RAISE:
                    raise ValueError(f"Unknown token/characters encountered: '{chunk}'")
                elif self.config.unk_strategy == UnkStrategy.UNK:
                    assert self._unk_token_id is not None
                    token_ids.append(self._unk_token_id)
                elif self.config.unk_strategy == UnkStrategy.IGNORE:
                    continue

        if self.config.add_eos:
            token_ids.append(self._eos_token_id)
        return token_ids

    def decode(self, ids: list[int]) -> str:
        """Filter out special IDs and decode back to a string."""
        decoded_tokens = []

        for token_id in ids:
            # Filter out BOS and EOS
            if token_id in (self._bos_token_id, self._eos_token_id):
                continue

            # Handle UNK
            if self._unk_token_id is not None and token_id == self._unk_token_id:
                decoded_tokens.append("<UNK>")
                continue

            # Decode standard tokens
            if 0 <= token_id < len(self.tokens):
                decoded_tokens.append(self.tokens[token_id])
            else:
                raise ValueError(f"Invalid token ID: {token_id}")

        # Because we don't assume whitespace during tokenization, we just concatenate
        return "".join(decoded_tokens)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size including BOS/EOS/UNK tokens."""
        return self._vocab_size

    @property
    def bos_token_id(self):
        """BOS token ID from config."""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """EOS token ID from config."""
        return self._eos_token_id
