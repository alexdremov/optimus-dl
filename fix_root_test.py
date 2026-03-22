import omegaconf

from optimus_dl.core.omegaconf import non_resolving_instantiate

c = omegaconf.OmegaConf.create({"_target_": "builtins.dict", "key": "val"})

res = non_resolving_instantiate(c, lazy=True)
print("Type:", type(res))
print("Val:", res)
