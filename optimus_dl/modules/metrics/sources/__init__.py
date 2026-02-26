from optimus_dl.core.bootstrap import bootstrap_module

from .causal_lm import (
    CausalLMSource,
    CausalLMSourceConfig,
)
from .generation import (
    GenerationSource,
    GenerationSourceConfig,
)

bootstrap_module(__name__)
