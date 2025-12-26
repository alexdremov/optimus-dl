from dataclasses import dataclass

from optimus_dl.modules.tokenizer import register_tokenizer
from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.modules.tokenizer.config import BaseTokenizerConfig


@dataclass
class CharTokenizerConfig(BaseTokenizerConfig):
    """Configuration for character/byte-level tokenizer.

    Attributes:
        vocab_size: Number of unique byte values (usually 256).
        bos_token_id: ID for the Beginning-of-Sequence token.
        eos_token_id: ID for the End-of-Sequence token.
    """

    vocab_size: int = 256  # 0-255 bytes + special tokens
    bos_token_id: int = 256
    eos_token_id: int = 257


@register_tokenizer("char_tokenize", CharTokenizerConfig)
class CharTokenizer(BaseTokenizer):
    """Simple byte-level UTF-8 tokenizer.

    Converts text to raw UTF-8 bytes and adds optional BOS/EOS tokens.
    Detokenization skips the special token IDs and decodes the remainder
    as UTF-8.

    Args:
        config: Character tokenizer configuration.
    """

    def __init__(self, config: CharTokenizerConfig, **kwargs):
        super().__init__(config)

    def encode(self, text: str) -> list[int]:
        """Convert text to UTF-8 bytes and add special tokens."""
        input_ids = list(text.encode("utf-8"))

        if self.config.add_bos:
            if self.bos_token_id is None:
                raise ValueError(
                    "Tokenizer does not have a BOS token ID, but add_bos is True."
                )
            input_ids.insert(0, self.bos_token_id)
        if self.config.add_eos:
            if self.eos_token_id is None:
                raise ValueError(
                    "Tokenizer does not have an EOS token ID, but add_eos is True."
                )
            input_ids.append(self.eos_token_id)
        return input_ids

    def decode(self, ids: list[int]) -> str:
        """Filter out special IDs and decode bytes to UTF-8."""
        # Filter out special tokens
        bytes_list = []
        for id in ids:
            if 0 <= id < 256:
                bytes_list.append(id)
        return bytes(bytes_list).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        """Vocabulary size including BOS/EOS tokens."""
        return max(self.config.vocab_size, (self.config.eos_token_id or 0) + 1)

    @property
    def bos_token_id(self):
        """BOS token ID from config."""
        return self.config.bos_token_id

    @property
    def eos_token_id(self):
        """EOS token ID from config."""
        return self.config.eos_token_id
