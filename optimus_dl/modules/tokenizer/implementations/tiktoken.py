from dataclasses import dataclass

import tiktoken

from optimus_dl.modules.tokenizer import register_tokenizer
from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.modules.tokenizer.config import BaseTokenizerConfig


@dataclass
class TiktokenConfig(BaseTokenizerConfig):
    """Configuration for Tiktoken tokenizers.

    Attributes:
        name: Name of the tiktoken encoding (e.g., 'gpt2', 'cl100k_base').
    """

    name: str = "gpt2"


@register_tokenizer("tiktoken", TiktokenConfig)
class TiktokenTokenizer(BaseTokenizer):
    """Wrapper for OpenAI's tiktoken library.

    Provides extremely fast Byte-Pair Encoding (BPE) for GPT-style models.

    Args:
        config: Tiktoken tokenizer configuration.
    """

    def __init__(self, config: TiktokenConfig, **kwargs):
        super().__init__(config)
        self.encoding = tiktoken.get_encoding(config.name)

    def encode(self, text: str) -> list[int]:
        """Convert text to IDs, allowing all special tokens."""
        # Using allowed_special="all" to permit special tokens in input text
        ids = self.encoding.encode(text, allowed_special="all")

        if self.config.add_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids

        if self.config.add_eos and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: list[int]) -> str:
        """Detokenize IDs into text."""
        return self.encoding.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the encoding."""
        return self.encoding.n_vocab

    @property
    def eos_token_id(self):
        """EOT token ID used as EOS."""
        return self.encoding.eot_token

    @property
    def bos_token_id(self):
        """EOT token ID used as BOS (tiktoken default)."""
        return self.encoding.eot_token
