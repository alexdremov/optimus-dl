import gc

from omegaconf import OmegaConf

from optimus_dl.core.omegaconf import non_resolving_instantiate


def test_collision():
    # a.b and a_b will both result in ghost_key "a_b"
    config = OmegaConf.create(
        {
            "a": {"b": {"_target_": "math.ceil", "_args_": [1.1]}},
            "a_b": {"_target_": "math.floor", "_args_": [1.9]},
        }
    )

    inst = non_resolving_instantiate(config)
    # If they collide, one will overwrite the other in the global store
    print(f"a.b (expected 2): {inst.a.b}")
    print(f"a_b (expected 1): {inst.a_b}")


def test_id_reuse():
    def create_and_instantiate(val):
        cfg = OmegaConf.create({"obj": {"_target_": "builtins.dict", "v": val}})
        inst = non_resolving_instantiate(cfg)
        _id = id(inst)
        # Access to instantiate
        _v = inst.obj.v
        return _id

    id1 = create_and_instantiate(1)
    # Force GC to increase chance of ID reuse
    gc.collect()

    # We try many times to hit the same ID
    for _i in range(100):
        cfg2 = OmegaConf.create({"obj": {"_target_": "builtins.dict", "v": 2}})
        if id(cfg2) == id1:
            print(f"Hit ID reuse: {id1}")
            inst2 = non_resolving_instantiate(cfg2)
            if inst2.obj.v == 1:
                print("FAILURE: Picked up stale state from ID reuse!")
            else:
                print("SUCCESS: Correct value despite ID reuse.")
            return
    print("Could not trigger ID reuse in this run.")


print("--- Collision Test ---")
test_collision()
print("\n--- ID Reuse Test ---")
test_id_reuse()
