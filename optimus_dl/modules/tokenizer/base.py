import pathlib
from abc import ABC, abstractmethod

import yaml

from .config import BaseTokenizerConfig


class BaseTokenizer(ABC):
    def __init__(self, config: BaseTokenizerConfig):
        self.config = config

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int | None:
        return None

    def save_pretrained(self, save_directory: str):
        """Save tokenizer artifacts to directory."""
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
        """Apply chat template to conversation history.

        Args:
            conversation: List of messages (e.g., [{"role": "user", "content": "Hello"}])
            tokenize: Whether to return token ids (True) or string (False)
            add_generation_prompt: Whether to append generation prompt

        Returns:
            Formatted string or list of token ids
        """
        raise NotImplementedError(
            "apply_chat_template not implemented for this tokenizer"
        )
