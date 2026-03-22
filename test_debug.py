from omegaconf import OmegaConf

from optimus_dl.core.omegaconf import (
    non_resolving_instantiate,
)

print("Starting debug script")
a = OmegaConf.create({"a": {"_target_": "math.floor", "_args_": [3.2]}, "b": "${.a}"})
print("Original a:", a)
b = non_resolving_instantiate(a)
print("Returned b:", b)
print("B.a type:", type(b.a))
print("B.a value:", b.a)
