import torch

print(torch.__version__)
import torch._dynamo

for k in dir(torch._dynamo.config):
    if "compile" in k.lower():
        print(k)
