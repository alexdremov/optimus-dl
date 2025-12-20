from dataclasses import dataclass

from transformers import AutoTokenizer

from optimus_dl.modules.tokenizer import register_tokenizer
from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.modules.tokenizer.config import BaseTokenizerConfig


@dataclass
class HFTokenizerConfig(BaseTokenizerConfig):
    name: str = "gpt2"
    trust_remote_code: bool = False


@register_tokenizer("transformers", HFTokenizerConfig)
class HFTokenizer(BaseTokenizer):
    def __init__(self, config: HFTokenizerConfig, **kwargs):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name, trust_remote_code=config.trust_remote_code
        )

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        if self.config.add_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if self.config.add_eos and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def save_pretrained(self, save_directory: str):
        self.tokenizer.save_pretrained(save_directory)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
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
