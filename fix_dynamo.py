import torch
import torch._dynamo

try:
    from triton.compiler import compiler

    torch._dynamo.disallow_in_graph(compiler.ASTSource.make_ir)
except Exception:
    pass

try:
    from liger_kernel.transformers.functional import liger_rms_norm

    torch._dynamo.disallow_in_graph(liger_rms_norm)
except Exception:
    pass
