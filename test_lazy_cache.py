from omegaconf import OmegaConf

from optimus_dl.core.omegaconf import non_resolving_instantiate

config = OmegaConf.create(
    {
        "p": 16,
        "outer": {
            "_target_": "builtins.dict",
            "inner": {"_target_": "math.sqrt", "_args_": ["${p}"]},
        },
    }
)

inst = non_resolving_instantiate(config)
print("Initial inst.p:", inst.p)
print("Initial inst.outer.inner:", inst.outer.inner)
inst.p = 25
print("Updated inst.p:", inst.p)
print("Updated inst.outer.inner:", inst.outer.inner)
