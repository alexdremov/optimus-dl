from dataclasses import dataclass

from transformers import AutoTokenizer

from optimus_dl.modules.tokenizer import register_tokenizer
from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.modules.tokenizer.config import BaseTokenizerConfig


@dataclass
class HFTokenizerConfig(BaseTokenizerConfig):
    """Configuration for Hugging Face tokenizers.

    Attributes:
        name: Name or path of the pretrained tokenizer on Hugging Face Hub.
        trust_remote_code: If True, allows executing code from the model repo.
    """

    name: str = "gpt2"
    trust_remote_code: bool = False


@register_tokenizer("transformers", HFTokenizerConfig)
class HFTokenizer(BaseTokenizer):
    """Wrapper for Hugging Face AutoTokenizer.

    Integrates standard Hub tokenizers into the framework. It handles:

    - **Pretrained Loading**: Automatically downloads and caches tokenizers.
    - **Special Tokens**: Manages BOS/EOS injection based on config.
    - **Chat Templates**: Supports generating formatted conversation strings.

    Args:
        config: Hugging Face tokenizer configuration.
    """

    def __init__(self, config: HFTokenizerConfig, **kwargs):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name, trust_remote_code=config.trust_remote_code
        )

    def encode(self, text: str) -> list[int]:
        """Convert text to IDs using the Hub tokenizer."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        if self.config.add_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if self.config.add_eos and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: list[int]) -> str:
        """Detokenize IDs into text."""
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size from Hub tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def bos_token_id(self):
        """BOS ID from Hub tokenizer."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        """EOS ID from Hub tokenizer."""
        return self.tokenizer.eos_token_id

    def save_pretrained(self, save_directory: str):
        """Delegate saving to the underlying transformers tokenizer."""
        self.tokenizer.save_pretrained(save_directory)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
        """Apply the Hub tokenizer's chat template to a conversation."""
        if (
            not hasattr(self.tokenizer, "apply_chat_template")
            or not self.tokenizer.chat_template
        ):
            raise ValueError("Tokenizer does not support chat template")

        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
