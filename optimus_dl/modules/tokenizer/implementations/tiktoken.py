from dataclasses import dataclass

import tiktoken

from optimus_dl.modules.tokenizer import register_tokenizer
from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.modules.tokenizer.config import BaseTokenizerConfig


@dataclass
class TiktokenConfig(BaseTokenizerConfig):
    name: str = "gpt2"


@register_tokenizer("tiktoken", TiktokenConfig)
class TiktokenTokenizer(BaseTokenizer):
    def __init__(self, config: TiktokenConfig, **kwargs):
        super().__init__(config)
        self.encoding = tiktoken.get_encoding(config.name)

    def encode(self, text: str) -> list[int]:
        # Using allowed_special="all" to permit special tokens in input text
        ids = self.encoding.encode(text, allowed_special="all")

        if self.config.add_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids

        if self.config.add_eos and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    @property
    def eos_token_id(self):
        return self.encoding.eot_token

    @property
    def bos_token_id(self):
        return self.encoding.eot_token
