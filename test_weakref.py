import weakref

from omegaconf import OmegaConf

cfg = OmegaConf.create({"a": 1})
try:
    wr = weakref.ref(cfg)
    print("OmegaConf objects are weakly referenceable")
except TypeError:
    print("OmegaConf objects are NOT weakly referenceable")
