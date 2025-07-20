from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import BaseTokenizer
from .config import BaseTokenizerConfig

_TOKENIZER_REGISTRY, register_tokenizer, build_tokenizer = make_registry(
    "tokenizer", BaseTokenizer
)
bootstrap_module(__name__)
