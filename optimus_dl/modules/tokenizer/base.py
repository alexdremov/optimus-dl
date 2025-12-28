import pathlib
from abc import (
    ABC,
    abstractmethod,
)

import yaml

from .config import BaseTokenizerConfig


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers.

    Defines the standard interface for encoding strings to token IDs and
    decoding IDs back to text.

    Attributes:
        config: Configuration object for the tokenizer.
    """

    def __init__(self, config: BaseTokenizerConfig):
        self.config = config

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a text string into a list of integer token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back into a text string."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total size of the tokenizer's vocabulary."""
        pass

    @property
    def bos_token_id(self) -> int | None:
        """ID of the Beginning-of-Sequence token, if any."""
        return None

    @property
    def eos_token_id(self) -> int | None:
        """ID of the End-of-Sequence token, if any."""
        return None

    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration to a directory."""
        save_directory_path = pathlib.Path(save_directory)
        save_directory_path.mkdir(parents=True, exist_ok=True)
        with open(save_directory_path / "tokenizer_config.json", "w") as f:
            yaml.dump(self.config, f)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
        """Apply a chat template (e.g., Llama-2-chat) to a conversation history.

        Args:
            conversation: List of messages (e.g., [{"role": "user", "content": "..."}]).
            tokenize: Whether to return token IDs (True) or the raw string (False).
            add_generation_prompt: Whether to append the assistant's response prefix.

        Returns:
            Formatted string or list of token IDs.
        """
        raise NotImplementedError(
            "apply_chat_template not implemented for this tokenizer"
        )
